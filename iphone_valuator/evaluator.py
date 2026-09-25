"""Decide whether an iPhone listing is worth its asking price.

Predicts the fair market value with the trained model, computes the price delta
``(asking - fair) / fair * 100%`` and explains the verdict in plain language.

Usage::

    python -m iphone_valuator.evaluator -m "iPhone 13 Pro" -s 256 -c used -b 78 -p 65000
    python -m iphone_valuator.evaluator --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Final, TypeVar, assert_never

import pandas as pd

from iphone_valuator.config import MODEL_PATH, NEW_DEVICE_BATTERY_HEALTH, configure_logging
from iphone_valuator.domain import (
    VALID_STORAGE_GB,
    Condition,
    PhoneModel,
    format_storage,
    get_model,
    months_since_release,
)
from iphone_valuator.features import build_feature_frame
from iphone_valuator.modeling import ModelBundle, load_bundle
from iphone_valuator.text_parsing import (
    normalize_region,
    parse_condition,
    parse_model,
    parse_price,
    parse_storage,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_PCT: Final = 10.0
SLIGHT_DEVIATION_PCT: Final = 5.0
LOW_BATTERY_PCT: Final = 80
HEALTHY_BATTERY_PCT: Final = 90
DISPLAY_ROUNDING_RUB: Final = 500
SUPPORTED_CONDITIONS: Final = (Condition.NEW, Condition.USED, Condition.REFURBISHED)
_THOUSANDS_PRICE_RE: Final = re.compile(r"\s*(\d+(?:[.,]\d+)?)\s*(?:k|к|тыс\.?)\s*", re.IGNORECASE)

T = TypeVar("T")


class Verdict(StrEnum):
    GREAT_DEAL = "Great Deal / Underpriced"
    FAIR = "Fair Market Price"
    OVERPRICED = "Overpriced / Not Worth It"


class InvalidListingError(ValueError):
    """The listing parameters are malformed, inconsistent or unsupported by the model."""


@dataclass(frozen=True, slots=True)
class ListingQuery:
    """Parameters of the listing being judged."""

    model: str
    storage_gb: int
    condition: Condition
    asking_price: int
    battery_health: int | None = None
    region: str | None = None
    has_box: bool = False
    has_receipt: bool = False


@dataclass(frozen=True, slots=True)
class Valuation:
    query: ListingQuery
    fair_price: float
    fair_range: tuple[float, float]
    delta_pct: float
    verdict: Verdict
    summary: str
    region: str
    battery_assumed: float | None
    typical_error_pct: float

    def to_dict(self) -> dict[str, object]:
        query = self.query
        return {
            "model": query.model,
            "storage": format_storage(query.storage_gb),
            "condition": query.condition.value,
            "battery_health": query.battery_health,
            "battery_assumed": self.battery_assumed,
            "region": self.region,
            "has_box": query.has_box,
            "has_receipt": query.has_receipt,
            "asking_price": query.asking_price,
            "fair_price": round(self.fair_price),
            "fair_range": [round(bound) for bound in self.fair_range],
            "delta_pct": round(self.delta_pct, 2),
            "verdict": self.verdict.value,
            "summary": self.summary,
            "typical_error_pct": round(self.typical_error_pct, 2),
        }


def compute_delta_pct(asking_price: float, fair_price: float) -> float:
    """Relative deviation of the asking price from the fair price, in percent."""
    if fair_price <= 0:
        raise ValueError("fair price must be positive")
    return (asking_price - fair_price) / fair_price * 100.0


def classify(delta_pct: float, threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> Verdict:
    """More than ``threshold_pct`` below market is a deal, more than above is overpriced."""
    if delta_pct < -threshold_pct:
        return Verdict.GREAT_DEAL
    if delta_pct > threshold_pct:
        return Verdict.OVERPRICED
    return Verdict.FAIR


def format_rub(value: float) -> str:
    return f"{value:,.0f} RUB"


def _rounded(value: float) -> float:
    return round(value / DISPLAY_ROUNDING_RUB) * DISPLAY_ROUNDING_RUB


def _battery_phrase(query: ListingQuery) -> str:
    battery = query.battery_health
    if battery is None:
        return "a new battery" if query.condition is Condition.NEW else "unknown battery health"
    if battery < LOW_BATTERY_PCT:
        return f"only {battery}% battery health"
    if battery < HEALTHY_BATTERY_PCT:
        return f"{battery}% battery health"
    return f"a healthy {battery}% battery"


def _extras_phrase(query: ListingQuery) -> str:
    extras = [
        label
        for present, label in ((query.has_box, "original box"), (query.has_receipt, "receipt"))
        if present
    ]
    return f" ({' and '.join(extras)} included)" if extras else ""


def build_summary(
    query: ListingQuery, fair_price: float, delta_pct: float, verdict: Verdict
) -> str:
    """One-sentence explanation of the verdict for humans."""
    fair = f"~{format_rub(_rounded(fair_price))}"
    asking = f"{format_rub(query.asking_price)} ({delta_pct:+.0f}%)"
    device = f"{query.condition.label} {query.model} {format_storage(query.storage_gb)}"
    details = f"for a {device} with {_battery_phrase(query)}{_extras_phrase(query)}"
    match verdict:
        case Verdict.OVERPRICED:
            return (
                f"Fair price is {fair}, but this seller is asking {asking} {details} — overpriced."
            )
        case Verdict.GREAT_DEAL:
            return (
                f"Fair price is {fair}, and this seller is asking only {asking} {details} — "
                "a great deal. Prices this far below market can hide defects or a scam, "
                "so inspect the phone before paying."
            )
        case Verdict.FAIR:
            if delta_pct > SLIGHT_DEVIATION_PCT:
                tail = "slightly above market, so there is room to negotiate"
            elif delta_pct < -SLIGHT_DEVIATION_PCT:
                tail = "slightly below market, a solid offer"
            else:
                tail = "right in line with the market"
            return f"Fair price is {fair} and this seller is asking {asking} {details} — {tail}."
        case _:
            assert_never(verdict)


def validate_query(query: ListingQuery) -> PhoneModel:
    """Check the query against the catalog and the model's training scope."""
    model = get_model(query.model)
    if model is None:
        raise InvalidListingError(f"Unknown iPhone model: {query.model!r}")
    if not model.supports_storage(query.storage_gb):
        options = ", ".join(format_storage(size) for size in model.storage_options)
        raise InvalidListingError(
            f"{model.name} was never sold with {format_storage(query.storage_gb)} "
            f"(available: {options})"
        )
    if query.condition not in SUPPORTED_CONDITIONS:
        raise InvalidListingError(
            "For-parts/broken phones are excluded from the training data, "
            "so the model cannot value them reliably"
        )
    if query.battery_health is not None and not 1 <= query.battery_health <= 100:
        raise InvalidListingError("Battery health must be between 1 and 100%")
    if query.asking_price <= 0:
        raise InvalidListingError("Asking price must be positive")
    return model


class Valuator:
    """Scores listings against the fair market price learned by a trained bundle."""

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        today: date | None = None,
    ) -> None:
        if threshold_pct <= 0:
            raise ValueError("threshold_pct must be positive")
        self._bundle = bundle
        self._threshold_pct = threshold_pct
        self._today = today

    @classmethod
    def from_path(
        cls, path: Path, *, threshold_pct: float = DEFAULT_THRESHOLD_PCT, today: date | None = None
    ) -> Valuator:
        return cls(load_bundle(path), threshold_pct=threshold_pct, today=today)

    @property
    def threshold_pct(self) -> float:
        return self._threshold_pct

    def evaluate(self, query: ListingQuery) -> Valuation:
        model = validate_query(query)
        battery_assumed = (
            None
            if query.battery_health is not None
            else self._assumed_battery(model, query.condition)
        )
        region = normalize_region(query.region) if query.region else self._bundle.default_region
        fair_price = self._predict(query, model, region, battery_assumed)
        delta_pct = compute_delta_pct(query.asking_price, fair_price)
        verdict = classify(delta_pct, self._threshold_pct)
        band = self._threshold_pct / 100.0
        return Valuation(
            query=query,
            fair_price=fair_price,
            fair_range=(fair_price * (1 - band), fair_price * (1 + band)),
            delta_pct=delta_pct,
            verdict=verdict,
            summary=build_summary(query, fair_price, delta_pct, verdict),
            region=region,
            battery_assumed=battery_assumed,
            typical_error_pct=self._bundle.typical_error_pct,
        )

    def predict_fair_price(self, query: ListingQuery) -> float:
        return self.evaluate(query).fair_price

    def _assumed_battery(self, model: PhoneModel, condition: Condition) -> float:
        if condition is Condition.NEW:
            return float(NEW_DEVICE_BATTERY_HEALTH)
        return round(self._bundle.impute_battery(model.release_year))

    def _predict(
        self, query: ListingQuery, model: PhoneModel, region: str, battery_assumed: float | None
    ) -> float:
        battery = query.battery_health if query.battery_health is not None else battery_assumed
        row = {
            "model": model.name,
            "storage_gb": query.storage_gb,
            "condition": query.condition.value,
            "region": region,
            "battery_health": battery,
            "battery_known": int(query.battery_health is not None),
            "phone_age_months": months_since_release(model, self._today or date.today()),
            "has_box": int(query.has_box),
            "has_receipt": int(query.has_receipt),
        }
        features = build_feature_frame(pd.DataFrame([row]))
        return float(self._bundle.estimator.predict(features)[0])


def parse_model_input(text: str) -> str:
    model = parse_model(text, require_prefix=False)
    if model is None:
        raise InvalidListingError(
            f"Could not recognise an iPhone model in {text!r} "
            "(try 'iPhone 13 Pro' or '14 pro max')"
        )
    return model


def parse_storage_input(text: str) -> int:
    storage = parse_storage(text, allow_bare=True)
    if storage is None and text.strip().isdigit() and int(text) in VALID_STORAGE_GB:
        storage = int(text)
    if storage is None:
        raise InvalidListingError(f"Could not read a storage size from {text!r} (try 128, 1TB)")
    return storage


def parse_condition_input(text: str) -> Condition:
    normalized = text.strip().lower()
    try:
        condition: Condition | None = Condition(normalized)
    except ValueError:
        condition = parse_condition(normalized)
    if condition is None:
        raise InvalidListingError(
            f"Unknown condition {text!r}; use one of: new, used, refurbished"
        )
    return condition


def parse_battery_input(text: str) -> int | None:
    cleaned = text.strip().rstrip("%").strip()
    if not cleaned:
        return None
    if not cleaned.isdigit() or not 1 <= int(cleaned) <= 100:
        raise InvalidListingError(f"Battery health must be a whole percentage 1-100, got {text!r}")
    return int(cleaned)


def parse_price_input(text: str) -> int:
    thousands = _THOUSANDS_PRICE_RE.fullmatch(text)
    price = (
        round(float(thousands.group(1).replace(",", ".")) * 1000)
        if thousands
        else parse_price(text)
    )
    if price is None or price <= 0:
        raise InvalidListingError(f"Could not read a price from {text!r} (try 65000 or 65k)")
    return price


def _ask(
    prompt: str,
    parser: Callable[[str], T],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> T:
    while True:
        try:
            return parser(input_fn(prompt))
        except InvalidListingError as error:
            output_fn(f"  ! {error}")


def _storage_for(model_name: str) -> Callable[[str], int]:
    model = get_model(model_name)

    def parse(text: str) -> int:
        storage = parse_storage_input(text)
        if model is not None and not model.supports_storage(storage):
            options = ", ".join(format_storage(size) for size in model.storage_options)
            raise InvalidListingError(f"{model.name} comes with {options}")
        return storage

    return parse


def _supported_condition(text: str) -> Condition:
    condition = parse_condition_input(text)
    if condition not in SUPPORTED_CONDITIONS:
        raise InvalidListingError(
            "For-parts/broken phones cannot be valued; pick new/used/refurbished"
        )
    return condition


def prompt_query(
    input_fn: Callable[[str], str] | None = None, output_fn: Callable[[str], None] | None = None
) -> ListingQuery:
    """Ask for listing parameters one by one, re-prompting on invalid input."""
    read = input_fn or input
    write = output_fn or print
    model = _ask("Model (e.g. iPhone 13 Pro, 14 pro max): ", parse_model_input, read, write)
    options = "/".join(format_storage(size) for size in _storage_options(model))
    storage = _ask(f"Storage [{options}]: ", _storage_for(model), read, write)
    condition = _ask("Condition [new/used/refurbished]: ", _supported_condition, read, write)
    battery = _ask("Battery health, % (Enter if unknown): ", parse_battery_input, read, write)
    price = _ask("Asking price, RUB: ", parse_price_input, read, write)
    return ListingQuery(
        model=model,
        storage_gb=storage,
        condition=condition,
        asking_price=price,
        battery_health=battery,
    )


def _storage_options(model_name: str) -> tuple[int, ...]:
    model = get_model(model_name)
    return model.storage_options if model is not None else VALID_STORAGE_GB


def format_report(valuation: Valuation) -> str:
    query = valuation.query
    if query.battery_health is not None:
        battery = f"battery {query.battery_health}%"
    else:
        battery = f"battery unknown (assumed ~{valuation.battery_assumed:.0f}%)"
    low, high = valuation.fair_range
    lines = [
        f"{query.model} {format_storage(query.storage_gb)} | {query.condition.label} | {battery} "
        f"| {valuation.region} | asking {format_rub(query.asking_price)}",
        f"Fair market value : ~{format_rub(_rounded(valuation.fair_price))} "
        f"(fair range {format_rub(_rounded(low))} – {format_rub(_rounded(high))})",
        f"Price delta       : {valuation.delta_pct:+.1f}%",
        f"Verdict           : {valuation.verdict.value}",
        f"Summary           : {valuation.summary}",
        f"Model uncertainty : typical error ±{valuation.typical_error_pct:.1f}% (cross-validated)",
    ]
    return "\n".join(lines)


def run_interactive(
    valuator: Valuator,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> int:
    """Interactive loop; returns an exit code."""
    read = input_fn or input
    write = output_fn or print
    write("iPhone listing valuator. Press Ctrl+C to quit.")
    try:
        while True:
            valuation = valuator.evaluate(prompt_query(read, write))
            write("")
            write(format_report(valuation))
            write("")
            answer = read("Evaluate another listing? [y/N]: ").strip().lower()
            if not answer.startswith(("y", "д")):
                return 0
    except (EOFError, KeyboardInterrupt):
        write("")
        return 0


def query_from_args(args: argparse.Namespace) -> ListingQuery:
    return ListingQuery(
        model=parse_model_input(args.model),
        storage_gb=parse_storage_input(args.storage),
        condition=parse_condition_input(args.condition),
        asking_price=parse_price_input(args.price),
        battery_health=parse_battery_input(args.battery or ""),
        region=args.region,
        has_box=args.box,
        has_receipt=args.receipt,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iphone-evaluate",
        description="Decide whether an Avito iPhone listing is worth its asking price.",
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("-m", "--model", help="iPhone model, e.g. 'iPhone 13 Pro' or '14 pro max'")
    parser.add_argument("-s", "--storage", help="storage, e.g. 128, 256GB, 1TB")
    parser.add_argument("-c", "--condition", help="new, used or refurbished")
    parser.add_argument("-b", "--battery", help="battery health in percent (omit if unknown)")
    parser.add_argument("-p", "--price", help="asking price in RUB, e.g. 65000 or 65k")
    parser.add_argument("--region", help="city/region (default: most common in training data)")
    parser.add_argument("--box", action="store_true", help="original box included")
    parser.add_argument("--receipt", action="store_true", help="receipt/documents included")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_PCT,
        help="half-width of the fair band in percent (default 10)",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("-i", "--interactive", action="store_true", help="prompt for inputs")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    required = {
        "--model": args.model,
        "--storage": args.storage,
        "--condition": args.condition,
        "--price": args.price,
    }
    interactive = args.interactive or all(value is None for value in required.values())
    missing = [flag for flag, value in required.items() if value is None]
    if not interactive and missing:
        parser.error(f"missing {', '.join(missing)} (or run with --interactive)")
    try:
        valuator = Valuator.from_path(args.model_path, threshold_pct=args.threshold)
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if interactive:
        return run_interactive(valuator)
    try:
        valuation = valuator.evaluate(query_from_args(args))
    except InvalidListingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(valuation.to_dict(), ensure_ascii=False, indent=2)
        if args.json
        else format_report(valuation)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
