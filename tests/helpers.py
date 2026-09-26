from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from iphone_valuator.features import FEATURE_COLUMNS
from iphone_valuator.modeling import ModelBundle, ModelName

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRAPED_AT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
QUICK_MODEL_PARAMS = {
    ModelName.CATBOOST: {"iterations": 300},
    ModelName.LIGHTGBM: {"n_estimators": 300},
}


class StubEstimator:
    """Returns a fixed fair price and records every feature frame it receives."""

    def __init__(self, price: float) -> None:
        self.price = price
        self.frames: list[pd.DataFrame] = []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.frames.append(features.copy())
        return np.full(len(features), self.price, dtype=float)


def make_stub_bundle(price: float = 55_000.0) -> ModelBundle:
    return ModelBundle(
        estimator=StubEstimator(price),
        model_name=ModelName.CATBOOST,
        feature_columns=FEATURE_COLUMNS,
        battery_median_by_year={2021: 86.0},
        battery_global_median=88.0,
        default_region="Москва",
        known_regions=("Москва", "Санкт-Петербург"),
        cv_metrics={},
        typical_error_pct=6.0,
        n_train_samples=100,
        trained_at="2026-09-01T00:00:00+00:00",
    )


def search_page_html(cards: Sequence[tuple[str, str, int]], *, has_next: bool) -> str:
    """Minimal Avito-like search page; ``cards`` are ``(item_id, title, price)`` tuples."""
    items = "".join(
        f'<div data-marker="item" data-item-id="{item_id}">'
        f'<a data-marker="item-title" href="/moskva/telefony/iphone_{item_id}">'
        f"<h3>{title}</h3></a>"
        f'<meta itemprop="price" content="{price}">'
        f'<div data-marker="item-address">Москва</div>'
        "</div>"
        for item_id, title, price in cards
    )
    if has_next:
        nav = '<a data-marker="pagination-button/nextPage" href="?p=2">next</a>'
    else:
        nav = '<span data-marker="pagination-button/nextPage" aria-disabled="true">next</span>'
    return f'<html><body>{items}<div data-marker="pagination-button">{nav}</div></body></html>'


def item_page_html(
    title: str, price: int, description: str, params: Mapping[str, str] | None = None
) -> str:
    rows = "".join(
        f"<li><span>{key}: </span>{value}</li>" for key, value in (params or {}).items()
    )
    return (
        "<html><body>"
        f'<h1 data-marker="item-view/title-info">{title}</h1>'
        f'<span data-marker="item-view/item-price" content="{price}">{price} ₽</span>'
        f'<div data-marker="item-view/item-params"><ul>{rows}</ul></div>'
        f'<div data-marker="item-view/item-description"><p>{description}</p></div>'
        '<div data-marker="item-view/item-address">Санкт-Петербург, Невский пр-т</div>'
        "</body></html>"
    )


def raw_row(**overrides: object) -> dict[str, object]:
    """A valid raw listing row (scraper schema) that survives cleaning unless overridden."""
    row: dict[str, object] = {
        "item_id": "1000000001",
        "listing_url": "https://www.avito.ru/moskva/telefony/iphone_1000000001",
        "title": "iPhone 13 Pro, 256 ГБ",
        "model": "iPhone 13 Pro",
        "storage_gb": 256,
        "condition": "used",
        "battery_health": 88,
        "price": 50_000,
        "description": "Отличное состояние, АКБ 88%. Коробка есть.",
        "location": "Москва, ул. Тверская",
        "params": json.dumps({"Модель": "iPhone 13 Pro"}, ensure_ascii=False),
        "scraped_at": SCRAPED_AT.isoformat(),
    }
    row.update(overrides)
    return row
