"""Feature engineering shared by training and inference, so both see identical model inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self, assert_never

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from iphone_valuator.config import UNKNOWN_REGION
from iphone_valuator.domain import format_storage, get_model

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "model",
    "model_tier",
    "storage",
    "condition",
    "region",
)
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "storage_gb",
    "battery_health",
    "battery_known",
    "phone_age_months",
    "has_box",
    "has_receipt",
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
BASE_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "storage_gb",
    "condition",
    "region",
    "battery_health",
    "battery_known",
    "phone_age_months",
    "has_box",
    "has_receipt",
)
TARGET_COLUMN: Final = "price"
OTHER_CATEGORY: Final = "__other__"
MONOTONE_DIRECTIONS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "storage_gb": 1,
        "battery_health": 1,
        "phone_age_months": -1,
        "has_box": 1,
        "has_receipt": 1,
    }
)


def _model_tier(name: object) -> str:
    model = get_model(name) if isinstance(name, str) else None
    return model.tier.value if model is not None else OTHER_CATEGORY


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map clean listing columns (``BASE_COLUMNS``) to the model matrix (``FEATURE_COLUMNS``)."""
    missing = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {missing}")
    storage_gb = df["storage_gb"].astype("int64")
    features = pd.DataFrame(
        {
            "model": df["model"].astype(str),
            "model_tier": df["model"].map(_model_tier).astype(str),
            "storage": storage_gb.map(format_storage).astype(str),
            "condition": df["condition"].astype(str),
            "region": df["region"].fillna(UNKNOWN_REGION).astype(str),
            "storage_gb": storage_gb.astype(float),
            "battery_health": df["battery_health"].astype(float),
            "battery_known": df["battery_known"].astype(float),
            "phone_age_months": df["phone_age_months"].astype(float),
            "has_box": df["has_box"].astype(float),
            "has_receipt": df["has_receipt"].astype(float),
        },
        index=df.index,
    )
    return features.loc[:, list(FEATURE_COLUMNS)]


class CategoryOutput(StrEnum):
    CATEGORY = "category"
    OBJECT = "object"


class CategoryEncoder(TransformerMixin, BaseEstimator):
    """Freezes categorical vocabularies at fit time; unseen or rare values become ``__other__``.

    ``output="category"`` yields pandas categoricals with fixed categories (LightGBM), while
    ``output="object"`` yields plain strings (CatBoost's native categorical handling).
    """

    def __init__(
        self,
        columns: Sequence[str] = CATEGORICAL_FEATURES,
        min_frequency: Mapping[str, int] | None = None,
        output: str = CategoryOutput.CATEGORY.value,
    ) -> None:
        self.columns = columns
        self.min_frequency = min_frequency
        self.output = output

    def fit(self, frame: pd.DataFrame, y: object = None) -> Self:
        thresholds = dict(self.min_frequency or {})
        self.categories_: dict[str, list[str]] = {}
        for column in self.columns:
            counts = frame[column].astype(str).value_counts()
            frequent = counts[counts >= thresholds.get(column, 1)].index
            vocabulary = sorted(str(value) for value in frequent if value != OTHER_CATEGORY)
            self.categories_[column] = [*vocabulary, OTHER_CATEGORY]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "categories_")
        output = CategoryOutput(self.output)
        encoded = frame.copy()
        for column in self.columns:
            vocabulary = self.categories_[column]
            values = encoded[column].astype(str)
            values = values.where(values.isin(vocabulary), OTHER_CATEGORY)
            match output:
                case CategoryOutput.CATEGORY:
                    encoded[column] = pd.Categorical(values, categories=vocabulary)
                case CategoryOutput.OBJECT:
                    encoded[column] = values.astype(object)
                case _:
                    assert_never(output)
        return encoded
