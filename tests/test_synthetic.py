from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from iphone_valuator import synthetic
from iphone_valuator.domain import Condition, require_model
from iphone_valuator.schemas import RAW_COLUMNS
from iphone_valuator.synthetic import SyntheticConfig, expected_price, generate_listings
from tests.helpers import SCRAPED_AT


def test_generate_listings_shape_and_determinism() -> None:
    config = SyntheticConfig(rows=400, seed=3, scraped_at=SCRAPED_AT)
    first = generate_listings(config)
    second = generate_listings(config)
    assert len(first) == 400
    assert list(first.columns) == list(RAW_COLUMNS)
    pd.testing.assert_frame_equal(first, second)


def test_generate_listings_contains_realistic_noise(synthetic_raw: pd.DataFrame) -> None:
    titles = synthetic_raw["title"].astype(str)
    assert titles.str.startswith("Чехол для").any()
    assert titles.str.startswith("Куплю").any()
    assert synthetic_raw["model"].isna().any()
    assert synthetic_raw["battery_health"].isna().any()
    assert synthetic_raw["item_id"].duplicated().any()
    assert (synthetic_raw["price"] < 5_000).any()
    params = synthetic_raw["params"].map(json.loads)
    assert params.map(lambda value: value["Производитель"] == "Apple").all()


def test_expected_price_structure() -> None:
    model = require_model("iPhone 13 Pro")
    base = expected_price(model, 128, Condition.USED, 88, False, False)
    assert base == pytest.approx(44_000)
    assert expected_price(model, 256, Condition.USED, 88, False, False) > base
    assert expected_price(model, 128, Condition.NEW, 88, False, False) > base
    assert expected_price(model, 128, Condition.USED, 75, False, False) < base
    assert expected_price(model, 128, Condition.USED, 88, True, True) > base
    assert expected_price(model, 128, Condition.USED, 88, False, False, 1.04) > base


def test_empty_generation() -> None:
    assert generate_listings(SyntheticConfig(rows=0, scraped_at=SCRAPED_AT)).empty


def test_cli_writes_csv(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "listings.csv"
    assert synthetic.main(["--rows", "50", "--output", str(output)]) == 0
    assert len(pd.read_csv(output)) == 50
