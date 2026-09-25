"""Estimator factory, log-price wrapper and the persisted model bundle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Self, assert_never

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike, NDArray
from sklearn.pipeline import Pipeline

from iphone_valuator.config import RANDOM_STATE
from iphone_valuator.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MONOTONE_DIRECTIONS,
    CategoryEncoder,
    CategoryOutput,
)

BUNDLE_SCHEMA_VERSION: Final = 1


class ModelName(StrEnum):
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"


DEFAULT_CATBOOST_PARAMS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "iterations": 1500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
    }
)
DEFAULT_LIGHTGBM_PARAMS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_data_per_group": 5,
        "cat_smooth": 10.0,
    }
)


class LogTargetRegressor:
    """Fits a pipeline on log-prices and predicts in RUB.

    Working in log space turns errors multiplicative (a 5 000 RUB miss matters more on a
    20 000 RUB phone than on a 120 000 RUB one) and guarantees positive predictions.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def fit(self, features: pd.DataFrame, prices: ArrayLike) -> Self:
        target = np.asarray(prices, dtype=float)
        if target.size == 0 or not np.all(np.isfinite(target)) or np.any(target <= 0):
            raise ValueError("Prices must be finite and strictly positive")
        self.pipeline.fit(features, np.log(target))
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        return np.exp(np.asarray(self.pipeline.predict(features), dtype=float))

    def feature_importances(self) -> dict[str, float]:
        importances = self.pipeline.named_steps["model"].feature_importances_
        return {
            name: float(value) for name, value in zip(FEATURE_COLUMNS, importances, strict=True)
        }


def make_estimator(
    name: ModelName,
    *,
    params: Mapping[str, object] | None = None,
    region_min_count: int = 10,
    random_state: int = RANDOM_STATE,
) -> LogTargetRegressor:
    """Build an unfitted CatBoost or LightGBM price model.

    LightGBM gets monotonic business constraints (better battery or more storage never lowers
    the price). CatBoost does not: enabling any monotone constraint there degrades its
    categorical handling and roughly doubles the error on this task.
    """
    min_frequency = {"region": region_min_count}
    overrides = dict(params or {})
    settings: dict[str, Any]
    match name:
        case ModelName.CATBOOST:
            encoder = CategoryEncoder(min_frequency=min_frequency, output=CategoryOutput.OBJECT)
            settings = {
                **DEFAULT_CATBOOST_PARAMS,
                "cat_features": list(CATEGORICAL_FEATURES),
                "random_seed": random_state,
                "verbose": 0,
                "allow_writing_files": False,
                **overrides,
            }
            regressor = CatBoostRegressor(**settings)
        case ModelName.LIGHTGBM:
            encoder = CategoryEncoder(min_frequency=min_frequency, output=CategoryOutput.CATEGORY)
            settings = {
                **DEFAULT_LIGHTGBM_PARAMS,
                "monotone_constraints": [MONOTONE_DIRECTIONS.get(c, 0) for c in FEATURE_COLUMNS],
                "random_state": random_state,
                "verbose": -1,
                **overrides,
            }
            regressor = LGBMRegressor(**settings)
        case _:
            assert_never(name)
    return LogTargetRegressor(Pipeline([("encode", encoder), ("model", regressor)]))


@dataclass(slots=True)
class ModelBundle:
    """Everything inference needs: the fitted pipeline plus training-time statistics."""

    estimator: LogTargetRegressor
    model_name: ModelName
    feature_columns: tuple[str, ...]
    battery_median_by_year: dict[int, float]
    battery_global_median: float
    default_region: str
    known_regions: tuple[str, ...]
    cv_metrics: dict[str, dict[str, float]]
    typical_error_pct: float
    n_train_samples: int
    trained_at: str
    library_versions: dict[str, str] = field(default_factory=dict)
    schema_version: int = BUNDLE_SCHEMA_VERSION

    def impute_battery(self, release_year: int) -> float:
        return self.battery_median_by_year.get(release_year, self.battery_global_median)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    """Load a bundle saved by :func:`save_bundle`.

    joblib files are pickles: only load artifacts you produced or trust.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Train one first: python -m iphone_valuator.train"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has bundle schema v{bundle.schema_version}, expected "
            f"v{BUNDLE_SCHEMA_VERSION}; retrain the model"
        )
    return bundle
