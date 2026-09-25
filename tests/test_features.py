from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from iphone_valuator.config import UNKNOWN_REGION
from iphone_valuator.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    OTHER_CATEGORY,
    CategoryEncoder,
    CategoryOutput,
    build_feature_frame,
)


def base_frame(**overrides: list[object]) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "model": ["iPhone 13 Pro", "iPhone 17 Pro Max"],
        "storage_gb": [256, 2048],
        "condition": ["used", "new"],
        "region": ["Москва", None],
        "battery_health": [87, 100],
        "battery_known": [1, 0],
        "phone_age_months": [59, 11],
        "has_box": [1, 0],
        "has_receipt": [0, 1],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_build_feature_frame_columns_and_types() -> None:
    features = build_feature_frame(base_frame())
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "model_tier"] == "pro"
    assert features.loc[1, "model_tier"] == "pro_max"
    assert list(features["storage"]) == ["256GB", "2TB"]
    assert features.loc[1, "region"] == UNKNOWN_REGION
    for column in NUMERIC_FEATURES:
        assert features[column].dtype == np.float64
    for column in CATEGORICAL_FEATURES:
        assert all(isinstance(value, str) for value in features[column])


def test_build_feature_frame_marks_unknown_models() -> None:
    features = build_feature_frame(base_frame(model=["iPhone 99", "iPhone 13"]))
    assert list(features["model_tier"]) == [OTHER_CATEGORY, "base"]


def test_build_feature_frame_requires_base_columns() -> None:
    with pytest.raises(ValueError, match="battery_health"):
        build_feature_frame(base_frame().drop(columns=["battery_health"]))


def categorical_frame(regions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["iPhone 13"] * len(regions),
            "model_tier": ["base"] * len(regions),
            "storage": ["128GB"] * len(regions),
            "condition": ["used"] * len(regions),
            "region": regions,
            "battery_health": [90.0] * len(regions),
        }
    )


def test_category_encoder_freezes_vocabulary_and_groups_rare_values() -> None:
    train = categorical_frame(["Москва"] * 3 + ["Казань"])
    encoder = CategoryEncoder(min_frequency={"region": 2}).fit(train)
    assert encoder.categories_["region"] == ["Москва", OTHER_CATEGORY]
    encoded = encoder.transform(categorical_frame(["Москва", "Казань", "Тверь"]))
    assert isinstance(encoded["region"].dtype, pd.CategoricalDtype)
    assert list(encoded["region"].cat.categories) == ["Москва", OTHER_CATEGORY]
    assert list(encoded["region"].astype(str)) == ["Москва", OTHER_CATEGORY, OTHER_CATEGORY]
    assert encoded["battery_health"].tolist() == [90.0, 90.0, 90.0]


def test_category_encoder_object_output_for_catboost() -> None:
    encoder = CategoryEncoder(output=CategoryOutput.OBJECT.value).fit(
        categorical_frame(["Москва"])
    )
    encoded = encoder.transform(categorical_frame(["Сочи"]))
    assert encoded["region"].dtype == object
    assert encoded.loc[0, "region"] == OTHER_CATEGORY


def test_category_encoder_does_not_duplicate_other_category() -> None:
    encoder = CategoryEncoder().fit(categorical_frame([OTHER_CATEGORY, "Москва"]))
    assert encoder.categories_["region"] == ["Москва", OTHER_CATEGORY]


def test_category_encoder_requires_fit() -> None:
    with pytest.raises(NotFittedError):
        CategoryEncoder().transform(categorical_frame(["Москва"]))


def test_category_encoder_does_not_mutate_input() -> None:
    frame = categorical_frame(["Москва", "Тверь"])
    CategoryEncoder().fit(frame).transform(frame)
    assert list(frame["region"]) == ["Москва", "Тверь"]
