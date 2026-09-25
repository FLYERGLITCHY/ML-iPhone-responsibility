"""Cross-validated training of CatBoost and LightGBM fair-price models.

The better model (by cross-validated MAPE by default) is refit on all clean listings and saved as
a joblib bundle together with a JSON metrics report.

Usage::

    python -m iphone_valuator.train --input data/processed/iphones_clean.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, assert_never

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from iphone_valuator.config import (
    CLEAN_DATA_PATH,
    FALLBACK_BATTERY_HEALTH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    UNKNOWN_REGION,
    configure_logging,
)
from iphone_valuator.domain import Condition, release_year_of
from iphone_valuator.features import (
    BASE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_frame,
)
from iphone_valuator.modeling import ModelBundle, ModelName, make_estimator, save_bundle

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS: Final = 30
TRACKED_PACKAGES: Final = ("catboost", "lightgbm", "numpy", "pandas", "scikit-learn")


class Metric(StrEnum):
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Errors in RUB (RMSE, MAE) and percent (MAPE) on the original price scale."""

    rmse: float
    mae: float
    mape: float

    @classmethod
    def compute(
        cls, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> RegressionMetrics:
        errors = y_pred - y_true
        return cls(
            rmse=float(np.sqrt(np.mean(errors**2))),
            mae=float(np.mean(np.abs(errors))),
            mape=float(np.mean(np.abs(errors) / y_true) * 100.0),
        )

    def get(self, metric: Metric) -> float:
        match metric:
            case Metric.RMSE:
                return self.rmse
            case Metric.MAE:
                return self.mae
            case Metric.MAPE:
                return self.mape
            case _:
                assert_never(metric)


@dataclass(frozen=True, slots=True, eq=False)
class CrossValidationResult:
    model_name: ModelName
    folds: tuple[RegressionMetrics, ...]
    oof_predictions: NDArray[np.float64]
    overall: RegressionMetrics

    def mean(self, metric: Metric) -> float:
        return float(np.mean([fold.get(metric) for fold in self.folds]))

    def std(self, metric: Metric) -> float:
        return float(np.std([fold.get(metric) for fold in self.folds]))

    def summary(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for metric in Metric:
            stats[f"{metric.value}_mean"] = round(self.mean(metric), 4)
            stats[f"{metric.value}_std"] = round(self.std(metric), 4)
        return stats


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    models: tuple[ModelName, ...] = (ModelName.CATBOOST, ModelName.LIGHTGBM)
    n_splits: int = 5
    select_by: Metric = Metric.MAPE
    region_min_count: int = 10
    random_state: int = RANDOM_STATE
    model_params: Mapping[ModelName, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("at least one model must be evaluated")
        if self.n_splits < 2:
            raise ValueError("cross-validation needs at least 2 folds")


@dataclass(frozen=True, slots=True, eq=False)
class TrainingResult:
    bundle: ModelBundle
    cv_results: dict[ModelName, CrossValidationResult]
    feature_importances: dict[str, float]
    select_by: Metric


def prepare_training_data(clean: pd.DataFrame) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Validate the clean dataset and split it into the feature matrix and RUB prices."""
    missing = [column for column in (*BASE_COLUMNS, TARGET_COLUMN) if column not in clean.columns]
    if missing:
        raise ValueError(f"Clean dataset is missing columns: {missing}")
    data = clean.dropna(subset=[*BASE_COLUMNS, TARGET_COLUMN])
    data = data.loc[data[TARGET_COLUMN] > 0]
    if len(data) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Need at least {MIN_TRAINING_ROWS} clean listings to train, got {len(data)}"
        )
    features = build_feature_frame(data).reset_index(drop=True)
    return features, data[TARGET_COLUMN].to_numpy(dtype=float)


def cross_validate(
    name: ModelName,
    features: pd.DataFrame,
    prices: NDArray[np.float64],
    config: TrainingConfig,
) -> CrossValidationResult:
    """K-fold CV producing per-fold metrics and out-of-fold predictions."""
    splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    oof = np.zeros_like(prices)
    folds: list[RegressionMetrics] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(features), start=1):
        estimator = make_estimator(
            name,
            params=config.model_params.get(name),
            region_min_count=config.region_min_count,
            random_state=config.random_state,
        )
        estimator.fit(features.iloc[train_idx], prices[train_idx])
        predictions = estimator.predict(features.iloc[valid_idx])
        oof[valid_idx] = predictions
        metrics = RegressionMetrics.compute(prices[valid_idx], predictions)
        folds.append(metrics)
        logger.info(
            "%s fold %d/%d: RMSE=%.0f MAE=%.0f MAPE=%.2f%%",
            name.value,
            fold,
            config.n_splits,
            metrics.rmse,
            metrics.mae,
            metrics.mape,
        )
    return CrossValidationResult(
        model_name=name,
        folds=tuple(folds),
        oof_predictions=oof,
        overall=RegressionMetrics.compute(prices, oof),
    )


def _battery_medians(clean: pd.DataFrame) -> tuple[dict[int, float], float]:
    known = clean.loc[(clean["battery_known"] == 1) & (clean["condition"] != Condition.NEW.value)]
    if known.empty:
        return {}, FALLBACK_BATTERY_HEALTH
    years = known["model"].map(release_year_of).astype(float)
    by_year = known["battery_health"].groupby(years).median()
    medians = {int(year): float(value) for year, value in by_year.items()}
    return medians, float(known["battery_health"].median())


def _region_statistics(features: pd.DataFrame, min_count: int) -> tuple[str, tuple[str, ...]]:
    counts = features["region"].value_counts()
    known = tuple(sorted(str(region) for region in counts[counts >= min_count].index))
    named = counts.drop(labels=[UNKNOWN_REGION], errors="ignore")
    default = str(named.index[0]) if not named.empty else UNKNOWN_REGION
    return default, known


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def train_models(clean: pd.DataFrame, config: TrainingConfig | None = None) -> TrainingResult:
    """Cross-validate every configured model, then refit the best one on all data."""
    config = config or TrainingConfig()
    features, prices = prepare_training_data(clean)
    results = {name: cross_validate(name, features, prices, config) for name in config.models}
    best = min(results.values(), key=lambda result: result.mean(config.select_by))
    logger.info("Best model by CV %s: %s", config.select_by.value.upper(), best.model_name.value)
    estimator = make_estimator(
        best.model_name,
        params=config.model_params.get(best.model_name),
        region_min_count=config.region_min_count,
        random_state=config.random_state,
    ).fit(features, prices)
    battery_by_year, battery_median = _battery_medians(clean)
    default_region, known_regions = _region_statistics(features, config.region_min_count)
    bundle = ModelBundle(
        estimator=estimator,
        model_name=best.model_name,
        feature_columns=FEATURE_COLUMNS,
        battery_median_by_year=battery_by_year,
        battery_global_median=battery_median,
        default_region=default_region,
        known_regions=known_regions,
        cv_metrics={name.value: result.summary() for name, result in results.items()},
        typical_error_pct=best.overall.mape,
        n_train_samples=len(features),
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        library_versions=_library_versions(),
    )
    return TrainingResult(
        bundle=bundle,
        cv_results=results,
        feature_importances=estimator.feature_importances(),
        select_by=config.select_by,
    )


def format_cv_table(result: TrainingResult) -> str:
    header = f"{'model':<10} {'RMSE, RUB':>18} {'MAE, RUB':>18} {'MAPE, %':>14}"
    rows = [header, "-" * len(header)]
    for name, cv in result.cv_results.items():
        marker = "  <- best" if name == result.bundle.model_name else ""
        rows.append(
            f"{name.value:<10} "
            f"{cv.mean(Metric.RMSE):>9,.0f} ± {cv.std(Metric.RMSE):<6,.0f} "
            f"{cv.mean(Metric.MAE):>9,.0f} ± {cv.std(Metric.MAE):<6,.0f} "
            f"{cv.mean(Metric.MAPE):>6.2f} ± {cv.std(Metric.MAPE):<5.2f}{marker}"
        )
    return "\n".join(rows)


def write_metrics(result: TrainingResult, path: Path) -> None:
    bundle = result.bundle
    importances = sorted(result.feature_importances.items(), key=lambda kv: kv[1], reverse=True)
    payload = {
        "best_model": bundle.model_name.value,
        "selected_by": result.select_by.value,
        "n_train_samples": bundle.n_train_samples,
        "trained_at": bundle.trained_at,
        "typical_error_pct": round(bundle.typical_error_pct, 3),
        "cross_validation": bundle.cv_metrics,
        "feature_importances": {name: round(value, 4) for name, value in importances},
        "library_versions": bundle.library_versions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iphone-train", description="Train CatBoost/LightGBM fair-price models with CV."
    )
    parser.add_argument("--input", type=Path, default=CLEAN_DATA_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_PATH)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[name.value for name in ModelName],
        default=[name.value for name in ModelName],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--select-by", choices=[metric.value for metric in Metric], default=Metric.MAPE.value
    )
    parser.add_argument("--region-min-count", type=int, default=10)
    parser.add_argument("--catboost-iterations", type=int, help="override CatBoost iterations")
    parser.add_argument("--lightgbm-estimators", type=int, help="override LightGBM n_estimators")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if not args.input.exists():
        logger.error("Clean data file not found: %s", args.input)
        return 1
    model_params: dict[ModelName, Mapping[str, object]] = {}
    if args.catboost_iterations:
        model_params[ModelName.CATBOOST] = {"iterations": args.catboost_iterations}
    if args.lightgbm_estimators:
        model_params[ModelName.LIGHTGBM] = {"n_estimators": args.lightgbm_estimators}
    try:
        config = TrainingConfig(
            models=tuple(ModelName(name) for name in args.models),
            n_splits=args.folds,
            select_by=Metric(args.select_by),
            region_min_count=args.region_min_count,
            model_params=model_params,
        )
        result = train_models(pd.read_csv(args.input), config)
    except ValueError as error:
        logger.error("Training failed: %s", error)
        return 1
    save_bundle(result.bundle, args.output)
    write_metrics(result, args.metrics_output)
    logger.info("Cross-validation results:\n%s", format_cv_table(result))
    logger.info("Saved model to %s and metrics to %s", args.output, args.metrics_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
