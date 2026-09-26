"""Synthetic Avito-like raw listings for offline demos and tests. NOT real market data.

The generator mimics the scraper's CSV schema, Russian listing text, missing attributes,
duplicates and the junk the cleaner must remove (locked, broken, accessories, placeholders).

Usage::

    python -m iphone_valuator.synthetic --rows 3000 --output data/raw/iphones_raw.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from iphone_valuator.config import RANDOM_STATE, RAW_DATA_PATH, configure_logging
from iphone_valuator.domain import (
    IPHONE_MODELS,
    Condition,
    PhoneModel,
    months_since_release,
)
from iphone_valuator.schemas import RAW_COLUMNS

logger = logging.getLogger(__name__)

BASE_PRICES_RUB: Final[Mapping[str, int]] = MappingProxyType(
    {
        "iPhone 8": 7_000,
        "iPhone 8 Plus": 9_000,
        "iPhone X": 11_000,
        "iPhone XR": 12_500,
        "iPhone XS": 13_500,
        "iPhone XS Max": 16_000,
        "iPhone 11": 18_000,
        "iPhone 11 Pro": 22_000,
        "iPhone 11 Pro Max": 26_000,
        "iPhone SE 2020": 9_500,
        "iPhone 12 mini": 19_000,
        "iPhone 12": 24_000,
        "iPhone 12 Pro": 30_000,
        "iPhone 12 Pro Max": 36_000,
        "iPhone 13 mini": 28_000,
        "iPhone 13": 34_000,
        "iPhone 13 Pro": 44_000,
        "iPhone 13 Pro Max": 52_000,
        "iPhone SE 2022": 15_000,
        "iPhone 14": 42_000,
        "iPhone 14 Plus": 46_000,
        "iPhone 14 Pro": 56_000,
        "iPhone 14 Pro Max": 64_000,
        "iPhone 15": 55_000,
        "iPhone 15 Plus": 60_000,
        "iPhone 15 Pro": 72_000,
        "iPhone 15 Pro Max": 85_000,
        "iPhone 16": 68_000,
        "iPhone 16 Plus": 75_000,
        "iPhone 16 Pro": 90_000,
        "iPhone 16 Pro Max": 105_000,
        "iPhone 16e": 50_000,
        "iPhone 17": 80_000,
        "iPhone Air": 90_000,
        "iPhone 17 Pro": 115_000,
        "iPhone 17 Pro Max": 130_000,
    }
)
POPULAR_MODELS: Final = frozenset(
    {
        "iPhone 11",
        "iPhone 12",
        "iPhone 13",
        "iPhone 13 Pro",
        "iPhone 14",
        "iPhone 14 Pro",
        "iPhone 14 Pro Max",
        "iPhone 15",
        "iPhone 15 Pro",
        "iPhone 15 Pro Max",
        "iPhone 16 Pro",
    }
)
STORAGE_STEP_MULTIPLIER: Final = 1.12
CONDITION_MULTIPLIERS: Final[Mapping[Condition, float]] = MappingProxyType(
    {Condition.USED: 1.0, Condition.NEW: 1.22, Condition.REFURBISHED: 0.9}
)
BATTERY_SLOPE: Final = 0.007
REFERENCE_BATTERY: Final = 88
BOX_MULTIPLIER: Final = 1.03
RECEIPT_MULTIPLIER: Final = 1.02
REGIONS: Final[tuple[tuple[str, str, float, float], ...]] = (
    ("Москва", "moskva", 0.34, 1.04),
    ("Санкт-Петербург", "sankt-peterburg", 0.16, 1.02),
    ("Московская область", "moskovskaya_oblast", 0.10, 1.01),
    ("Екатеринбург", "ekaterinburg", 0.07, 0.98),
    ("Новосибирск", "novosibirsk", 0.06, 0.97),
    ("Казань", "kazan", 0.06, 0.98),
    ("Краснодар", "krasnodar", 0.06, 0.99),
    ("Нижний Новгород", "nizhniy_novgorod", 0.05, 0.97),
    ("Ростов-на-Дону", "rostov-na-donu", 0.05, 0.98),
    ("Самара", "samara", 0.05, 0.97),
)
STREETS: Final = ("ул. Ленина", "пр-т Мира", "ул. Гагарина", "Центральный р-н", "ул. Советская")
CONDITION_PARAMS: Final[Mapping[Condition, tuple[str, ...]]] = MappingProxyType(
    {
        Condition.USED: ("Отличное", "Хорошее", "Удовлетворительное"),
        Condition.NEW: ("Новое",),
        Condition.REFURBISHED: ("Восстановленное",),
    }
)
CONDITION_PHRASES: Final[Mapping[Condition, tuple[str, ...]]] = MappingProxyType(
    {
        Condition.USED: (
            "Состояние отличное, без сколов и царапин.",
            "Есть мелкие потертости на корпусе, экран идеальный.",
            "Пользовался аккуратно, всегда в чехле и со стеклом.",
            "Состояние хорошее, следы использования минимальные.",
        ),
        Condition.NEW: (
            "Новый, запечатан, не активирован.",
            "Абсолютно новый, не активирован, гарантия магазина.",
        ),
        Condition.REFURBISHED: (
            "Восстановленный, выглядит как новый.",
            "Официально восстановленный (refurbished), корпус заменен.",
        ),
    }
)
BATTERY_PHRASES: Final = (
    "АКБ {b}%.",
    "Аккумулятор {b}%.",
    "Ёмкость аккумулятора {b} %.",
    "{b}% акб.",
    "Battery health {b}%.",
    "Состояние батареи {b}%.",
)
KIT_PHRASES: Final[Mapping[tuple[bool, bool], tuple[str, ...]]] = MappingProxyType(
    {
        (True, True): ("Полный комплект: коробка, чек.", "Коробка, чек, документы — все есть."),
        (True, False): ("Коробка есть, чека нет.", "Есть родная коробка."),
        (False, True): ("Без коробки, но есть чек.",),
        (False, False): ("Только телефон, без коробки и документов.", "Коробки и чека нет."),
    }
)
EXTRA_PHRASES: Final = (
    "Face ID работает.",
    "Не битый, не утопленник.",
    "Торг уместен.",
    "Без торга.",
    "С перекупами не работаю.",
    "Никогда не вскрывался.",
    "Отправлю Авито Доставкой.",
)
JUNK_TEMPLATES: Final[tuple[tuple[str, str, float], ...]] = (
    ("{title}", "На запчасти, не включается.", 0.25),
    ("{title}", "Заблокирован на iCloud, пароль не помню.", 0.3),
    ("{title}", "Экран разбит, остальное работает.", 0.55),
    ("{title}", "Утопленник, после воды не заряжается.", 0.3),
    ("Чехол для {model}", "Новый силиконовый чехол, подходит идеально.", 0.02),
    ("{title}", "Реплика, точная копия, работает шустро.", 0.2),
    ("Куплю {model}", "Куплю дорого в любом состоянии.", 0.5),
)
PLACEHOLDER_PRICES: Final = (1, 100, 1_000, 999_999, 1_111_111)


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    rows: int = 3000
    junk_share: float = 0.08
    anomaly_share: float = 0.04
    duplicate_share: float = 0.03
    missing_battery_share: float = 0.3
    unparsed_share: float = 0.2
    noise_sigma: float = 0.06
    seed: int = RANDOM_STATE
    scraped_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def storage_label_ru(storage_gb: int) -> str:
    return f"{storage_gb // 1024} ТБ" if storage_gb >= 1024 else f"{storage_gb} ГБ"


def expected_price(
    model: PhoneModel,
    storage_gb: int,
    condition: Condition,
    battery_health: int,
    has_box: bool,
    has_receipt: bool,
    region_multiplier: float = 1.0,
) -> float:
    """The noise-free "true" market price used to generate synthetic listings."""
    price = BASE_PRICES_RUB[model.name] * STORAGE_STEP_MULTIPLIER ** model.storage_options.index(
        storage_gb
    )
    price *= CONDITION_MULTIPLIERS[condition]
    if condition is Condition.USED:
        price *= 1 + BATTERY_SLOPE * (battery_health - REFERENCE_BATTERY)
    if has_box:
        price *= BOX_MULTIPLIER
    if has_receipt:
        price *= RECEIPT_MULTIPLIER
    return price * region_multiplier


class _ListingFactory:
    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._scraped_at = config.scraped_at.isoformat(timespec="seconds")
        weights = np.array([3.0 if m.name in POPULAR_MODELS else 1.0 for m in IPHONE_MODELS])
        self._model_weights = weights / weights.sum()
        region_weights = np.array([region[2] for region in REGIONS])
        self._region_weights = region_weights / region_weights.sum()
        self._next_id = 4_000_000_000

    def listing(self) -> dict[str, object]:
        rng = self._rng
        model = IPHONE_MODELS[rng.choice(len(IPHONE_MODELS), p=self._model_weights)]
        storage_weights = np.array([0.45, 0.33, 0.16, 0.06][: len(model.storage_options)])
        storage = int(rng.choice(model.storage_options, p=storage_weights / storage_weights.sum()))
        condition = [Condition.USED, Condition.NEW, Condition.REFURBISHED][
            rng.choice(3, p=[0.78, 0.12, 0.10])
        ]
        battery = self._battery(model, condition)
        has_box = bool(rng.random() < (0.95 if condition is Condition.NEW else 0.55))
        has_receipt = bool(rng.random() < (0.7 if condition is Condition.NEW else 0.3))
        region, slug, _, region_multiplier = REGIONS[
            rng.choice(len(REGIONS), p=self._region_weights)
        ]
        price = expected_price(
            model, storage, condition, battery, has_box, has_receipt, region_multiplier
        ) * rng.lognormal(0.0, self._config.noise_sigma)
        battery_known = rng.random() >= self._config.missing_battery_share
        parsed = rng.random() >= self._config.unparsed_share
        title = self._title(model, storage)
        item_id = self._new_id()
        return {
            "item_id": item_id,
            "listing_url": f"https://www.avito.ru/{slug}/telefony/iphone_{item_id}",
            "title": title,
            "model": model.name if parsed else None,
            "storage_gb": storage if parsed else None,
            "condition": condition.value if parsed else None,
            "battery_health": battery if parsed and battery_known else None,
            "price": self._round_price(price),
            "description": self._description(
                condition, battery, battery_known, has_box, has_receipt
            ),
            "location": f"{region}, {STREETS[rng.integers(len(STREETS))]}",
            "params": json.dumps(
                {
                    "Производитель": "Apple",
                    "Модель": model.name,
                    "Встроенная память": storage_label_ru(storage),
                    "Состояние": str(rng.choice(CONDITION_PARAMS[condition])),
                },
                ensure_ascii=False,
            ),
            "scraped_at": self._scraped_at,
        }

    def junk(self) -> dict[str, object]:
        row = self.listing()
        title_template, description, price_factor = JUNK_TEMPLATES[
            self._rng.integers(len(JUNK_TEMPLATES))
        ]
        model_name = str(json.loads(str(row["params"]))["Модель"])
        row["title"] = title_template.format(title=row["title"], model=model_name)
        row["description"] = description
        row["price"] = self._round_price(max(float(str(row["price"])) * price_factor, 300.0))
        return row

    def anomaly(self) -> dict[str, object]:
        row = self.listing()
        price = float(str(row["price"]))
        kind = self._rng.integers(4)
        if kind == 0:
            row["price"] = int(self._rng.choice(PLACEHOLDER_PRICES))
        elif kind == 1:
            row["price"] = self._round_price(price * 0.15)
            row["description"] = f"{row['description']} Первый взнос, остальное в рассрочку."
        elif kind == 2:
            row["price"] = self._round_price(price * 2.8)
        else:
            row["price"] = self._round_price(price * 0.45)
        return row

    def duplicate(self, source: Mapping[str, object]) -> dict[str, object]:
        row = dict(source)
        if self._rng.random() < 0.5:
            item_id = self._new_id()
            row["item_id"] = item_id
            row["listing_url"] = f"https://www.avito.ru/moskva/telefony/iphone_{item_id}"
        return row

    def _battery(self, model: PhoneModel, condition: Condition) -> int:
        if condition is Condition.NEW:
            return 100
        if condition is Condition.REFURBISHED:
            return int(np.clip(100 - abs(self._rng.normal(0, 3)), 85, 100))
        age = months_since_release(model, self._config.scraped_at.date())
        return int(np.clip(100 - 0.35 * age + self._rng.normal(0, 4), 72, 100))

    def _title(self, model: PhoneModel, storage: int) -> str:
        style = self._rng.integers(4)
        if style == 0:
            return f"Apple {model.name} {storage_label_ru(storage).replace(' ', '')}"
        if style == 1 and storage < 1024:
            return f"{model.name} {storage}"
        return f"{model.name}, {storage_label_ru(storage)}"

    def _description(
        self,
        condition: Condition,
        battery: int,
        battery_known: bool,
        has_box: bool,
        has_receipt: bool,
    ) -> str:
        rng = self._rng
        parts = [str(rng.choice(CONDITION_PHRASES[condition]))]
        if battery_known:
            parts.append(str(rng.choice(BATTERY_PHRASES)).format(b=battery))
        parts.append(str(rng.choice(KIT_PHRASES[(has_box, has_receipt)])))
        extras = rng.choice(len(EXTRA_PHRASES), size=rng.integers(1, 4), replace=False)
        parts.extend(EXTRA_PHRASES[index] for index in extras)
        return " ".join(parts)

    def _round_price(self, price: float) -> int:
        rounded = max(round(price / 500) * 500, 1)
        if rounded >= 10_000 and self._rng.random() < 0.3:
            return rounded - 10
        return rounded

    def _new_id(self) -> str:
        self._next_id += int(self._rng.integers(1, 50))
        return str(self._next_id)


def generate_listings(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Build a raw-listings frame with the scraper's columns."""
    config = config or SyntheticConfig()
    factory = _ListingFactory(config)
    rng = np.random.default_rng(config.seed + 1)
    n_junk = round(config.rows * config.junk_share)
    n_anomalies = round(config.rows * config.anomaly_share)
    n_duplicates = round(config.rows * config.duplicate_share)
    n_clean = max(config.rows - n_junk - n_anomalies - n_duplicates, 0)
    rows = [factory.listing() for _ in range(n_clean)]
    rows += [factory.junk() for _ in range(n_junk)]
    rows += [factory.anomaly() for _ in range(n_anomalies)]
    if rows:
        rows += [factory.duplicate(rows[rng.integers(len(rows))]) for _ in range(n_duplicates)]
    order = rng.permutation(len(rows))
    return pd.DataFrame([rows[index] for index in order], columns=list(RAW_COLUMNS))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iphone-synthetic", description="Generate synthetic raw listings (demo data only)."
    )
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    listings = generate_listings(SyntheticConfig(rows=args.rows, seed=args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    listings.to_csv(args.output, index=False)
    logger.info("Wrote %d synthetic listings to %s", len(listings), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
