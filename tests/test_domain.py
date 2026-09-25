from __future__ import annotations

from datetime import date

import pytest

from iphone_valuator.domain import (
    IPHONE_MODELS,
    VALID_STORAGE_GB,
    Condition,
    ModelTier,
    format_storage,
    get_model,
    months_since_release,
    release_year_of,
    require_model,
)


def test_catalog_names_are_unique() -> None:
    names = [model.name for model in IPHONE_MODELS]
    assert len(names) == len(set(names))


def test_catalog_storage_options_are_sorted_and_known() -> None:
    for model in IPHONE_MODELS:
        assert list(model.storage_options) == sorted(model.storage_options)
        assert set(model.storage_options) <= set(VALID_STORAGE_GB)


def test_valid_storage_covers_64gb_to_2tb() -> None:
    assert VALID_STORAGE_GB == (64, 128, 256, 512, 1024, 2048)


@pytest.mark.parametrize("name", ["iPhone 13 Pro", "iphone 13 pro", "  IPHONE   13  PRO "])
def test_get_model_is_case_and_whitespace_insensitive(name: str) -> None:
    model = get_model(name)
    assert model is not None
    assert model.name == "iPhone 13 Pro"
    assert model.tier is ModelTier.PRO


@pytest.mark.parametrize("name", [None, "", "iPhone 99", "Galaxy S24"])
def test_get_model_returns_none_for_unknown(name: str | None) -> None:
    assert get_model(name) is None


def test_require_model_raises_for_unknown() -> None:
    with pytest.raises(KeyError, match="iPhone 99"):
        require_model("iPhone 99")


def test_supports_storage() -> None:
    model = require_model("iPhone 15 Pro Max")
    assert model.supports_storage(256)
    assert not model.supports_storage(128)
    assert require_model("iPhone 17 Pro Max").supports_storage(2048)


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (date(2021, 9, 24), 0),
        (date(2021, 10, 23), 0),
        (date(2021, 10, 24), 1),
        (date(2026, 9, 1), 59),
        (date(2020, 1, 1), 0),
    ],
)
def test_months_since_release(as_of: date, expected: int) -> None:
    assert months_since_release(require_model("iPhone 13"), as_of) == expected


@pytest.mark.parametrize(
    ("storage", "label"), [(64, "64GB"), (512, "512GB"), (1024, "1TB"), (2048, "2TB")]
)
def test_format_storage(storage: int, label: str) -> None:
    assert format_storage(storage) == label


def test_release_year_of_handles_non_strings() -> None:
    assert release_year_of("iPhone 12 mini") == 2020
    assert release_year_of(float("nan")) is None
    assert release_year_of("unknown") is None


def test_every_condition_has_a_label() -> None:
    assert {condition.label for condition in Condition} == {
        "brand-new",
        "used",
        "refurbished",
        "for-parts/broken",
    }
