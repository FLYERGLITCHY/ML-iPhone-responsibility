"""Cleaning, anti-garbage filtering and normalization of raw Avito iPhone listings.

Usage::

    python -m iphone_valuator.cleaner --input data/raw/iphones_raw.csv \
        --output data/processed/iphones_clean.csv --outlier-method both
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from iphone_valuator.config import (
    CLEAN_DATA_PATH,
    FALLBACK_BATTERY_HEALTH,
    MAX_BATTERY_HEALTH,
    MAX_PRICE_RUB,
    MIN_BATTERY_HEALTH,
    MIN_PRICE_RUB,
    NEW_DEVICE_BATTERY_HEALTH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    configure_logging,
)
from iphone_valuator.domain import (
    VALID_STORAGE_GB,
    Condition,
    get_model,
    months_since_release,
    release_year_of,
    require_model,
)
from iphone_valuator.schemas import CLEAN_COLUMNS
from iphone_valuator.text_parsing import (
    JunkReason,
    detect_junk,
    find_param,
    has_original_box,
    has_receipt,
    mentions_installment,
    normalize_region,
    normalize_text,
    parse_battery_health,
    parse_condition,
    parse_model,
    parse_price,
    parse_storage,
)

logger = logging.getLogger(__name__)

MIN_LOG_IQR: Final = 0.05
MIN_ISOLATION_FOREST_ROWS: Final = 50
FINGERPRINT_CHARS: Final = 200
MIN_FINGERPRINT_CHARS: Final = 30
_CONDITION_CODES: Final[dict[str, int]] = {
    Condition.USED.value: 0,
    Condition.REFURBISHED.value: 1,
    Condition.NEW.value: 2,
}
_NORMALIZED_COLUMNS: Final[tuple[str, ...]] = (
    "item_id",
    "listing_url",
    "title",
    "model",
    "storage_gb",
    "condition",
    "battery_health",
    "price",
    "location",
    "region",
    "has_box",
    "has_receipt",
    "installment",
    "junk_reason",
    "description",
    "scraped_at",
)
_OUTLIER_LEVELS: Final = (("model", "storage_gb", "condition"), ("model", "condition"), ("model",))
_PRICE_LEVELS: Final = (("model", "storage_gb"), ("model",))


class OutlierMethod(StrEnum):
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    BOTH = "both"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Thresholds of the anti-garbage heuristics."""

    min_price: int = MIN_PRICE_RUB
    max_price: int = MAX_PRICE_RUB
    relative_price_floor: float = 0.3
    installment_price_ratio: float = 0.5
    outlier_method: OutlierMethod = OutlierMethod.IQR
    iqr_multiplier: float = 2.0
    min_group_size: int = 10
    isolation_contamination: float = 0.02
    default_condition: Condition = Condition.USED
    random_state: int = RANDOM_STATE
    reference_date: date | None = None


@dataclass(slots=True)
class CleaningReport:
    """How many rows each rule removed, in pipeline order."""

    input_rows: int = 0
    output_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + count

    def format(self) -> str:
        lines = [f"input rows: {self.input_rows}"]
        lines += [f"  -{count:<6} {reason}" for reason, count in self.dropped.items()]
        lines.append(f"output rows: {self.output_rows}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CleaningResult:
    data: pd.DataFrame
    report: CleaningReport


def _as_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _load_params(value: object) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in decoded.items()} if isinstance(decoded, dict) else {}


def _as_int(value: object) -> int | None:
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _known_storage(value: object) -> int | None:
    number = _as_int(value)
    return number if number in VALID_STORAGE_GB else None


def _valid_battery(value: object) -> int | None:
    number = _as_int(value)
    if number is None or not MIN_BATTERY_HEALTH <= number <= MAX_BATTERY_HEALTH:
        return None
    return number


def normalize_record(
    record: Mapping[str, object], default_condition: Condition = Condition.USED
) -> dict[str, object]:
    """Resolve one raw row into typed attributes, re-parsing text wherever fields are missing."""
    title = _as_text(record.get("title"))
    description = _as_text(record.get("description"))
    params = _load_params(record.get("params"))
    text = f"{title}\n{description}"
    params_text = "\n".join(f"{key}: {value}" for key, value in params.items())
    raw_condition = record.get("condition")
    condition_source = (
        raw_condition if _as_text(raw_condition) else find_param(params, "состояние")
    )
    junk = detect_junk(title, description)
    return {
        "item_id": _as_text(record.get("item_id")),
        "listing_url": _as_text(record.get("listing_url")),
        "title": title,
        "model": parse_model(record.get("model"), require_prefix=False)
        or parse_model(find_param(params, "модель"), require_prefix=False)
        or parse_model(title)
        or parse_model(description),
        "storage_gb": _known_storage(record.get("storage_gb"))
        or parse_storage(
            find_param(params, "встроенная память", "объем встроенной памяти", "встроен")
        )
        or parse_storage(title, allow_bare=True)
        or parse_storage(description),
        "condition": (parse_condition(condition_source, text) or default_condition).value,
        "battery_health": _valid_battery(record.get("battery_health"))
        or parse_battery_health(params_text)
        or parse_battery_health(description)
        or parse_battery_health(title),
        "price": parse_price(record.get("price")),
        "location": _as_text(record.get("location")),
        "region": normalize_region(record.get("location")),
        "has_box": has_original_box(text),
        "has_receipt": has_receipt(text),
        "installment": mentions_installment(text),
        "junk_reason": junk.value if junk is not None else None,
        "description": description,
        "scraped_at": _as_text(record.get("scraped_at")),
    }


def _release_years(models: pd.Series) -> pd.Series:
    return models.map(release_year_of).astype(float)


def _supports_storage(name: object, storage_gb: int) -> bool:
    model = get_model(name) if isinstance(name, str) else None
    return model is not None and model.supports_storage(storage_gb)


def _description_fingerprint(text: object) -> str:
    return re.sub(r"\W+", " ", normalize_text(text))[:FINGERPRINT_CHARS].strip()


def _group_statistics(
    df: pd.DataFrame, values: pd.Series, levels: Sequence[Sequence[str]], min_size: int
) -> pd.DataFrame:
    """Per-row median/quartiles from the most specific grouping level with enough rows."""
    stats = pd.DataFrame(np.nan, index=df.index, columns=["median", "q1", "q3", "count"])
    for keys in levels:
        grouped = values.groupby([df[key] for key in keys])
        count = grouped.transform("size")
        eligible = stats["count"].isna() & (count >= min_size)
        if not eligible.any():
            continue
        stats.loc[eligible, "median"] = grouped.transform("median")[eligible]
        stats.loc[eligible, "q1"] = grouped.transform("quantile", 0.25)[eligible]
        stats.loc[eligible, "q3"] = grouped.transform("quantile", 0.75)[eligible]
        stats.loc[eligible, "count"] = count[eligible]
    return stats


class ListingCleaner:
    """Turns raw scraped rows into a strict, de-duplicated, model-ready dataset."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self._config = config or CleaningConfig()

    def clean(self, raw: pd.DataFrame) -> CleaningResult:
        config = self._config
        report = CleaningReport(input_rows=len(raw))
        records = [
            normalize_record(row, config.default_condition) for row in raw.to_dict("records")
        ]
        df = pd.DataFrame.from_records(records, columns=list(_NORMALIZED_COLUMNS))
        df = self._drop(df, df["price"].isna(), "missing_price", report)
        df = self._drop(df, df["model"].isna(), "unknown_model", report)
        df = self._drop(df, df["storage_gb"].isna(), "missing_storage", report)
        df = df.astype({"price": "int64", "storage_gb": "int64"})
        supported = [
            _supports_storage(m, int(s))
            for m, s in zip(df["model"], df["storage_gb"], strict=True)
        ]
        df = self._drop(
            df, ~pd.Series(supported, index=df.index, dtype=bool), "storage_not_offered", report
        )
        for reason in JunkReason:
            df = self._drop(df, df["junk_reason"] == reason.value, f"junk_{reason.value}", report)
        df = self._drop(
            df, df["condition"] == Condition.FOR_PARTS.value, "condition_for_parts", report
        )
        df = self._drop(df, df["price"] < config.min_price, "price_below_minimum", report)
        df = self._drop(df, df["price"] > config.max_price, "price_above_maximum", report)
        df = self._deduplicate(df, report)
        df = self._impute_battery(df)
        df = self._drop(df, self._suspiciously_cheap(df), "suspiciously_cheap", report)
        df = self._remove_outliers(df, report)
        df = self._add_derived_columns(df)
        report.output_rows = len(df)
        clean = df.loc[:, list(CLEAN_COLUMNS)].reset_index(drop=True)
        return CleaningResult(data=clean, report=report)

    @staticmethod
    def _drop(
        df: pd.DataFrame, mask: pd.Series, reason: str, report: CleaningReport
    ) -> pd.DataFrame:
        flags = mask.fillna(False).astype(bool)
        report.record(reason, int(flags.sum()))
        return df.loc[~flags]

    def _deduplicate(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        same_id = (df["item_id"] != "") & df.duplicated(subset=["item_id"], keep="first")
        df = self._drop(df, same_id, "duplicate_item_id", report)
        same_url = (df["listing_url"] != "") & df.duplicated(subset=["listing_url"], keep="first")
        df = self._drop(df, same_url, "duplicate_listing_url", report)
        fingerprint = df["description"].map(_description_fingerprint)
        keys = df[["model", "storage_gb", "price"]].assign(fingerprint=fingerprint)
        reposted = (fingerprint.str.len() >= MIN_FINGERPRINT_CHARS) & keys.duplicated(keep="first")
        return self._drop(df, reposted, "duplicate_content", report)

    @staticmethod
    def _impute_battery(df: pd.DataFrame) -> pd.DataFrame:
        known = df["battery_health"].notna()
        is_new = df["condition"] == Condition.NEW.value
        release_year = _release_years(df["model"])
        reference = known & ~is_new
        medians = df.loc[reference, "battery_health"].groupby(release_year[reference]).median()
        global_median = (
            float(df.loc[reference, "battery_health"].median())
            if reference.any()
            else FALLBACK_BATTERY_HEALTH
        )
        imputed = release_year.map(medians).fillna(global_median)
        imputed = imputed.where(~is_new, float(NEW_DEVICE_BATTERY_HEALTH))
        return df.assign(
            battery_known=known.astype("int64"),
            battery_health=df["battery_health"].fillna(imputed).round().astype("int64"),
        )

    def _suspiciously_cheap(self, df: pd.DataFrame) -> pd.Series:
        log_price = np.log(df["price"].astype(float))
        stats = _group_statistics(df, log_price, _PRICE_LEVELS, self._config.min_group_size)
        ratio = np.exp(log_price - stats["median"])
        below_floor = ratio < self._config.relative_price_floor
        installment_bait = df["installment"].astype(bool) & (
            ratio < self._config.installment_price_ratio
        )
        return (below_floor | installment_bait) & stats["median"].notna()

    def _iqr_outliers(self, df: pd.DataFrame) -> pd.Series:
        log_price = np.log(df["price"].astype(float))
        stats = _group_statistics(df, log_price, _OUTLIER_LEVELS, self._config.min_group_size)
        iqr = (stats["q3"] - stats["q1"]).clip(lower=MIN_LOG_IQR)
        k = self._config.iqr_multiplier
        outside = (log_price < stats["q1"] - k * iqr) | (log_price > stats["q3"] + k * iqr)
        return outside & stats["count"].notna()

    def _isolation_forest_outliers(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < MIN_ISOLATION_FOREST_ROWS:
            logger.warning("Skipping Isolation Forest: only %d rows", len(df))
            return pd.Series(False, index=df.index)
        log_price = np.log(df["price"].astype(float))
        stats = _group_statistics(df, log_price, _PRICE_LEVELS, min_size=1)
        features = pd.DataFrame(
            {
                "log_price_deviation": log_price - stats["median"],
                "battery_health": df["battery_health"].astype(float),
                "condition": df["condition"].map(_CONDITION_CODES).astype(float),
                "log_storage": np.log2(df["storage_gb"].astype(float)),
                "release_year": _release_years(df["model"]),
            }
        )
        forest = IsolationForest(
            n_estimators=200,
            contamination=self._config.isolation_contamination,
            random_state=self._config.random_state,
        )
        return pd.Series(forest.fit_predict(features.to_numpy()) == -1, index=df.index)

    def _remove_outliers(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        method = self._config.outlier_method
        match method:
            case OutlierMethod.NONE:
                return df
            case OutlierMethod.IQR:
                return self._drop(df, self._iqr_outliers(df), "price_outlier_iqr", report)
            case OutlierMethod.ISOLATION_FOREST:
                flags = self._isolation_forest_outliers(df)
                return self._drop(df, flags, "price_outlier_isolation_forest", report)
            case OutlierMethod.BOTH:
                df = self._drop(df, self._iqr_outliers(df), "price_outlier_iqr", report)
                flags = self._isolation_forest_outliers(df)
                return self._drop(df, flags, "price_outlier_isolation_forest", report)
            case _:
                assert_never(method)

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        fallback = self._config.reference_date or datetime.now(UTC).date()
        scraped = pd.to_datetime(df["scraped_at"], format="ISO8601", errors="coerce", utc=True)
        as_of = [fallback if pd.isna(ts) else ts.date() for ts in scraped]
        models = [require_model(name) for name in df["model"]]
        return df.assign(
            release_year=[model.release_year for model in models],
            phone_age_months=[
                months_since_release(model, day) for model, day in zip(models, as_of, strict=True)
            ],
            has_box=df["has_box"].astype("int64"),
            has_receipt=df["has_receipt"].astype("int64"),
        )


def load_raw_listings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"item_id": str, "listing_url": str})


def save_clean_listings(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iphone-clean", description="Filter junk and normalize raw Avito iPhone listings."
    )
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=CLEAN_DATA_PATH)
    parser.add_argument(
        "--outlier-method",
        choices=[method.value for method in OutlierMethod],
        default=OutlierMethod.IQR.value,
    )
    parser.add_argument("--iqr-multiplier", type=float, default=2.0, help="fence width in IQRs")
    parser.add_argument("--contamination", type=float, default=0.02, help="Isolation Forest share")
    parser.add_argument("--min-price", type=int, default=MIN_PRICE_RUB)
    parser.add_argument("--max-price", type=int, default=MAX_PRICE_RUB)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    if not args.input.exists():
        logger.error("Raw data file not found: %s", args.input)
        return 1
    config = CleaningConfig(
        min_price=args.min_price,
        max_price=args.max_price,
        outlier_method=OutlierMethod(args.outlier_method),
        iqr_multiplier=args.iqr_multiplier,
        isolation_contamination=args.contamination,
    )
    result = ListingCleaner(config).clean(load_raw_listings(args.input))
    save_clean_listings(result.data, args.output)
    logger.info("Cleaning report:\n%s", result.report.format())
    if result.data.empty:
        logger.warning("No listings survived cleaning; check the raw data and thresholds")
    logger.info("Saved %d clean listings to %s", len(result.data), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
