from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iphone_valuator import cleaner
from iphone_valuator.cleaner import (
    CleaningConfig,
    CleaningReport,
    ListingCleaner,
    OutlierMethod,
    normalize_record,
)
from iphone_valuator.config import FALLBACK_BATTERY_HEALTH
from iphone_valuator.schemas import CLEAN_COLUMNS, RAW_COLUMNS
from iphone_valuator.text_parsing import detect_junk
from tests.helpers import SCRAPED_AT, raw_row


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(RAW_COLUMNS))


def group(
    count: int, price: int = 50_000, start_id: int = 0, **overrides: object
) -> list[dict[str, object]]:
    """``count`` distinct but comparable listings around ``price``."""
    rows = []
    for index in range(count):
        item_id = str(2_000_000_000 + start_id + index)
        wiggle = (index % 5 - 2) * 500
        rows.append(
            raw_row(
                item_id=item_id,
                listing_url=f"https://www.avito.ru/moskva/telefony/iphone_{item_id}",
                price=price + wiggle,
                description=f"Состояние хорошее, АКБ 88%. Объявление номер {item_id}.",
                **overrides,
            )
        )
    return rows


def varied_rows(count: int, seed: int = 0) -> list[dict[str, object]]:
    """Listings with distinct prices and batteries, so anomaly scores do not tie."""
    rng = np.random.default_rng(seed)
    return [
        raw_row(
            item_id=f"v{index}",
            listing_url=f"https://www.avito.ru/moskva/telefony/iphone_v{index}",
            price=int(rng.normal(50_000, 2_500)),
            battery_health=int(rng.integers(75, 101)),
            description=f"Описание объявления номер {index}, все отлично.",
        )
        for index in range(count)
    ]


def clean(*rows: dict[str, object], **config: object) -> tuple[pd.DataFrame, CleaningReport]:
    settings: dict[str, object] = {"reference_date": SCRAPED_AT.date()}
    settings.update(config)
    result = ListingCleaner(CleaningConfig(**settings)).clean(frame(*rows))
    return result.data, result.report


def test_normalize_record_parses_missing_fields_from_text() -> None:
    record = raw_row(
        model=None,
        storage_gb=None,
        condition=None,
        battery_health=None,
        title="Айфон 12 мини",
        description="Б/у. Коробки нет, чек есть.",
        params=json.dumps(
            {"Встроенная память": "128 ГБ", "Ёмкость аккумулятора": "81%"}, ensure_ascii=False
        ),
        price="21 500 ₽",
        location="г. Санкт-Петербург, Невский",
    )
    normalized = normalize_record(record)
    assert normalized["model"] == "iPhone 12 mini"
    assert normalized["storage_gb"] == 128
    assert normalized["condition"] == "used"
    assert normalized["battery_health"] == 81
    assert normalized["price"] == 21500
    assert normalized["region"] == "Санкт-Петербург"
    assert normalized["has_box"] is False
    assert normalized["has_receipt"] is True
    assert normalized["junk_reason"] is None


def test_normalize_record_trusts_valid_fields_and_repairs_invalid_ones() -> None:
    normalized = normalize_record(
        raw_row(storage_gb=100, battery_health=150, title="iPhone 13 Pro 512GB", description="")
    )
    assert normalized["storage_gb"] == 512
    assert normalized["battery_health"] is None
    assert normalize_record(raw_row(storage_gb="256.0"))["storage_gb"] == 256


def test_normalize_record_tolerates_missing_and_broken_columns() -> None:
    normalized = normalize_record({"title": "iPhone 15, 128 ГБ", "price": 60000, "params": "{bad"})
    assert normalized["model"] == "iPhone 15"
    assert normalized["storage_gb"] == 128
    assert normalized["condition"] == "used"
    assert normalized["item_id"] == ""


def test_clean_output_schema_and_derived_columns() -> None:
    data, report = clean(raw_row())
    assert list(data.columns) == list(CLEAN_COLUMNS)
    row = data.iloc[0]
    assert row["model"] == "iPhone 13 Pro"
    assert row["region"] == "Москва"
    assert row["release_year"] == 2021
    assert row["phone_age_months"] == 59
    assert row["battery_known"] == 1
    assert row["has_box"] == 1
    assert row["has_receipt"] == 0
    assert report.input_rows == report.output_rows == 1
    assert report.dropped == {}


def test_missing_scrape_date_uses_reference_date() -> None:
    data, _ = clean(raw_row(scraped_at=None), reference_date=date(2022, 9, 24))
    assert data.iloc[0]["phone_age_months"] == 12


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"price": None}, "missing_price"),
        ({"price": "договорная"}, "missing_price"),
        ({"model": None, "title": "Samsung Galaxy S23", "params": "{}"}, "unknown_model"),
        (
            {"storage_gb": None, "title": "iPhone 13 Pro", "params": "{}", "description": "ok"},
            "missing_storage",
        ),
        ({"model": "iPhone 13", "storage_gb": 64}, "storage_not_offered"),
        ({"description": "Заблокирован на iCloud"}, "junk_locked"),
        ({"description": "На запчасти"}, "junk_for_parts"),
        ({"description": "Экран разбит"}, "junk_broken_screen"),
        ({"description": "Утопленник"}, "junk_water_damage"),
        ({"description": "Реплика"}, "junk_fake"),
        ({"title": "Чехол для iPhone 13 Pro"}, "junk_accessory"),
        ({"title": "Куплю iPhone 13 Pro 256"}, "junk_not_for_sale"),
        ({"condition": "for_parts"}, "condition_for_parts"),
        ({"price": 4_999}, "price_below_minimum"),
        ({"price": 999_999}, "price_above_maximum"),
    ],
)
def test_hard_filters(overrides: dict[str, object], reason: str) -> None:
    data, report = clean(raw_row(**overrides))
    assert data.empty
    assert report.dropped == {reason: 1}


def test_deduplication_by_id_url_and_reposted_content() -> None:
    base = raw_row(
        item_id="1", listing_url="u1", description="Уникальное описание телефона, АКБ 88%."
    )
    same_id = raw_row(item_id="1", listing_url="u2", description="другое описание объявления")
    same_url = raw_row(item_id="3", listing_url="u1", description="третье описание объявления")
    reposted = raw_row(item_id="4", listing_url="u4", description=base["description"])
    short_text = raw_row(item_id="5", listing_url="u5", description="ok")
    short_twin = raw_row(item_id="6", listing_url="u6", description="ok")
    data, report = clean(base, same_id, same_url, reposted, short_text, short_twin)
    assert sorted(data["item_id"]) == ["1", "5", "6"]
    assert report.dropped == {
        "duplicate_item_id": 1,
        "duplicate_listing_url": 1,
        "duplicate_content": 1,
    }


def test_battery_imputation_by_release_year_and_condition() -> None:
    rows = [
        raw_row(item_id="1", listing_url="u1", battery_health=80, description="a" * 40),
        raw_row(item_id="2", listing_url="u2", battery_health=90, description="b" * 40),
        raw_row(item_id="3", listing_url="u3", battery_health=None, description="c" * 40),
        raw_row(
            item_id="4",
            listing_url="u4",
            battery_health=None,
            condition="new",
            description="d" * 40,
        ),
        raw_row(
            item_id="5",
            listing_url="u5",
            model="iPhone 15",
            storage_gb=128,
            title="iPhone 15",
            battery_health=None,
            description="e" * 40,
        ),
    ]
    data, _ = clean(*rows)
    by_id = data.set_index("item_id")
    assert by_id.loc["3", "battery_health"] == 85
    assert by_id.loc["3", "battery_known"] == 0
    assert by_id.loc["4", "battery_health"] == 100
    assert by_id.loc["5", "battery_health"] == 85
    assert by_id.loc["1", "battery_known"] == 1


def test_battery_imputation_without_any_known_values_uses_fallback() -> None:
    data, _ = clean(raw_row(battery_health=None, description="Хорошее состояние"))
    assert data.iloc[0]["battery_health"] == round(FALLBACK_BATTERY_HEALTH)
    assert data.iloc[0]["battery_known"] == 0


def test_suspiciously_cheap_and_installment_bait_are_removed() -> None:
    rows = group(12)
    rows.append(raw_row(item_id="cheap", listing_url="cheap", price=12_000, description="x" * 40))
    rows.append(
        raw_row(
            item_id="bait",
            listing_url="bait",
            price=22_000,
            description="Первый взнос, остальное в рассрочку",
        )
    )
    data, report = clean(*rows, outlier_method=OutlierMethod.NONE)
    assert "cheap" not in set(data["item_id"])
    assert "bait" not in set(data["item_id"])
    assert report.dropped["suspiciously_cheap"] == 2
    assert len(data) == 12


def test_iqr_removes_extreme_prices_within_comparable_group() -> None:
    rows = group(15)
    rows.append(raw_row(item_id="high", listing_url="high", price=150_000, description="y" * 40))
    rows.append(raw_row(item_id="low", listing_url="low", price=30_000, description="z" * 40))
    data, report = clean(*rows, outlier_method=OutlierMethod.IQR)
    assert {"high", "low"}.isdisjoint(set(data["item_id"]))
    assert report.dropped["price_outlier_iqr"] == 2
    assert len(data) == 15


def test_iqr_keeps_small_groups_it_cannot_judge() -> None:
    rows = group(3)
    rows.append(raw_row(item_id="high", listing_url="high", price=150_000, description="y" * 40))
    data, report = clean(*rows, outlier_method=OutlierMethod.IQR)
    assert len(data) == 4
    assert "price_outlier_iqr" not in report.dropped


def test_iqr_falls_back_to_coarser_groups() -> None:
    rows = group(8, storage_gb=256, start_id=0)
    rows += group(8, storage_gb=512, title="iPhone 13 Pro, 512 ГБ", start_id=100)
    rows.append(
        raw_row(
            item_id="odd",
            listing_url="odd",
            storage_gb=128,
            title="iPhone 13 Pro, 128 ГБ",
            price=200_000,
            description="q" * 40,
        )
    )
    data, report = clean(*rows, outlier_method=OutlierMethod.IQR)
    assert "odd" not in set(data["item_id"])
    assert report.dropped["price_outlier_iqr"] == 1


def test_isolation_forest_flags_roughly_the_contamination_share() -> None:
    data, report = clean(
        *varied_rows(100),
        outlier_method=OutlierMethod.ISOLATION_FOREST,
        isolation_contamination=0.05,
    )
    flagged = report.dropped["price_outlier_isolation_forest"]
    assert 3 <= flagged <= 7
    assert len(data) == 100 - flagged


def test_isolation_forest_is_skipped_for_small_datasets() -> None:
    data, report = clean(*group(10), outlier_method=OutlierMethod.ISOLATION_FOREST)
    assert len(data) == 10
    assert report.dropped == {}


def test_outlier_method_none_and_both() -> None:
    rows = varied_rows(60)
    rows.append(raw_row(item_id="high", listing_url="high", price=150_000, description="y" * 40))
    kept, _ = clean(*rows, outlier_method=OutlierMethod.NONE)
    assert len(kept) == 61
    data, report = clean(*rows, outlier_method=OutlierMethod.BOTH)
    assert "high" not in set(data["item_id"])
    assert report.dropped["price_outlier_iqr"] >= 1
    assert report.dropped["price_outlier_isolation_forest"] >= 1


def test_empty_input_produces_empty_clean_frame() -> None:
    data, report = clean()
    assert data.empty
    assert list(data.columns) == list(CLEAN_COLUMNS)
    assert report.input_rows == report.output_rows == 0


def test_report_accounts_for_every_row(synthetic_raw: pd.DataFrame) -> None:
    result = ListingCleaner(CleaningConfig(reference_date=SCRAPED_AT.date())).clean(synthetic_raw)
    report = result.report
    assert report.input_rows - sum(report.dropped.values()) == report.output_rows
    assert report.output_rows == len(result.data)
    assert "input rows: 1500" in report.format()


def test_synthetic_junk_never_survives(synthetic_clean: pd.DataFrame) -> None:
    assert not synthetic_clean.empty
    reasons = [
        detect_junk(title, description)
        for title, description in zip(
            synthetic_clean["title"], synthetic_clean["description"].fillna(""), strict=True
        )
    ]
    assert all(reason is None for reason in reasons)
    assert synthetic_clean["price"].between(5_000, 400_000).all()
    assert synthetic_clean["item_id"].is_unique
    assert synthetic_clean["battery_health"].between(40, 100).all()
    assert set(synthetic_clean["condition"]) <= {"new", "used", "refurbished"}
    assert np.issubdtype(synthetic_clean["price"].dtype, np.integer)


def test_cli_cleans_csv(tmp_path: Path, synthetic_raw: pd.DataFrame) -> None:
    raw_path = tmp_path / "raw.csv"
    output = tmp_path / "processed" / "iphones_clean.csv"
    synthetic_raw.to_csv(raw_path, index=False)
    assert cleaner.main(["--input", str(raw_path), "--output", str(output)]) == 0
    written = pd.read_csv(output)
    assert list(written.columns) == list(CLEAN_COLUMNS)
    assert len(written) > 1000


def test_cli_reports_missing_input(tmp_path: Path) -> None:
    assert cleaner.main(["--input", str(tmp_path / "absent.csv")]) == 1
