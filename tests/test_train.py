from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from iphone_valuator import train
from iphone_valuator.features import FEATURE_COLUMNS, build_feature_frame
from iphone_valuator.modeling import (
    BUNDLE_SCHEMA_VERSION,
    LogTargetRegressor,
    ModelName,
    load_bundle,
    make_estimator,
    save_bundle,
)
from iphone_valuator.train import (
    MIN_TRAINING_ROWS,
    Metric,
    RegressionMetrics,
    TrainingConfig,
    TrainingResult,
    format_cv_table,
    prepare_training_data,
    write_metrics,
)


def test_regression_metrics() -> None:
    metrics = RegressionMetrics.compute(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert metrics.rmse == pytest.approx(np.sqrt((100 + 400) / 2))
    assert metrics.mae == pytest.approx(15.0)
    assert metrics.mape == pytest.approx(10.0)
    assert metrics.get(Metric.MAPE) == metrics.mape
    assert metrics.get(Metric.RMSE) == metrics.rmse
    assert metrics.get(Metric.MAE) == metrics.mae


def test_training_config_validation() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        TrainingConfig(models=())
    with pytest.raises(ValueError, match="2 folds"):
        TrainingConfig(n_splits=1)


def test_prepare_training_data_validates_input(synthetic_clean: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        prepare_training_data(synthetic_clean.drop(columns=["region"]))
    with pytest.raises(ValueError, match=f"at least {MIN_TRAINING_ROWS}"):
        prepare_training_data(synthetic_clean.head(MIN_TRAINING_ROWS - 1))
    with_gaps = synthetic_clean.copy()
    with_gaps.loc[with_gaps.index[:5], "battery_health"] = np.nan
    features, prices = prepare_training_data(with_gaps)
    assert len(features) == len(prices) == len(synthetic_clean) - 5
    assert tuple(features.columns) == FEATURE_COLUMNS


def test_log_target_regressor_rejects_non_positive_prices(synthetic_clean: pd.DataFrame) -> None:
    features = build_feature_frame(synthetic_clean.head(10))
    estimator = make_estimator(ModelName.LIGHTGBM, params={"n_estimators": 5})
    with pytest.raises(ValueError, match="strictly positive"):
        estimator.fit(features, np.zeros(10))


def test_training_selects_best_model_and_reaches_useful_accuracy(
    training_result: TrainingResult,
) -> None:
    results = training_result.cv_results
    assert set(results) == {ModelName.CATBOOST, ModelName.LIGHTGBM}
    best = training_result.bundle.model_name
    assert results[best].mean(Metric.MAPE) == min(r.mean(Metric.MAPE) for r in results.values())
    for result in results.values():
        assert len(result.folds) == 3
        assert result.mean(Metric.MAPE) < 15.0
        assert np.all(result.oof_predictions > 0)
    assert training_result.bundle.typical_error_pct == pytest.approx(results[best].overall.mape)


def test_bundle_contains_inference_statistics(training_result: TrainingResult) -> None:
    bundle = training_result.bundle
    assert bundle.feature_columns == FEATURE_COLUMNS
    assert bundle.default_region == "Москва"
    assert "Москва" in bundle.known_regions
    assert 70 <= bundle.battery_global_median <= 100
    assert all(70 <= value <= 100 for value in bundle.battery_median_by_year.values())
    assert bundle.impute_battery(1990) == bundle.battery_global_median
    assert set(bundle.cv_metrics) == {"catboost", "lightgbm"}
    assert {"mape_mean", "rmse_std"} <= set(bundle.cv_metrics["catboost"])
    assert bundle.library_versions["python"]
    assert set(training_result.feature_importances) == set(FEATURE_COLUMNS)


def test_trained_model_learns_price_structure(training_result: TrainingResult) -> None:
    rows = pd.DataFrame(
        {
            "model": ["iPhone 13", "iPhone 15 Pro Max", "iPhone 13", "iPhone 13"],
            "storage_gb": [128, 256, 128, 128],
            "condition": ["used", "used", "new", "used"],
            "region": ["Москва"] * 4,
            "battery_health": [88, 88, 100, 75],
            "battery_known": [1] * 4,
            "phone_age_months": [59, 35, 59, 59],
            "has_box": [0] * 4,
            "has_receipt": [0] * 4,
        }
    )
    base, flagship, new, worn = training_result.bundle.estimator.predict(build_feature_frame(rows))
    assert flagship > base * 1.8
    assert new > base
    assert worn < base


def test_lightgbm_is_monotonic_in_battery_health(synthetic_clean: pd.DataFrame) -> None:
    features, prices = prepare_training_data(synthetic_clean)
    estimator = make_estimator(ModelName.LIGHTGBM, params={"n_estimators": 200})
    estimator.fit(features, prices)
    probe = pd.concat([features.iloc[[0]]] * 61, ignore_index=True)
    probe["battery_health"] = np.arange(40, 101, dtype=float)
    predictions = estimator.predict(probe)
    assert np.all(np.diff(predictions) >= -1e-6)


def test_bundle_round_trip(training_result: TrainingResult, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "model.joblib"
    save_bundle(training_result.bundle, path)
    loaded = load_bundle(path)
    assert isinstance(loaded.estimator, LogTargetRegressor)
    assert loaded.model_name == training_result.bundle.model_name


def test_load_bundle_errors(training_result: TrainingResult, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Train one first"):
        load_bundle(tmp_path / "absent.joblib")
    joblib.dump({"not": "a bundle"}, tmp_path / "dict.joblib")
    with pytest.raises(TypeError, match="ModelBundle"):
        load_bundle(tmp_path / "dict.joblib")
    stale = replace(training_result.bundle, schema_version=BUNDLE_SCHEMA_VERSION + 1)
    joblib.dump(stale, tmp_path / "stale.joblib")
    with pytest.raises(ValueError, match="retrain"):
        load_bundle(tmp_path / "stale.joblib")


def test_metrics_report_and_table(training_result: TrainingResult, tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    write_metrics(training_result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["best_model"] == training_result.bundle.model_name.value
    assert payload["selected_by"] == "mape"
    assert set(payload["cross_validation"]) == {"catboost", "lightgbm"}
    importances = list(payload["feature_importances"].values())
    assert importances == sorted(importances, reverse=True)
    table = format_cv_table(training_result)
    assert "catboost" in table
    assert "lightgbm" in table
    assert table.count("<- best") == 1


def test_cli_trains_and_saves_artifacts(tmp_path: Path, synthetic_clean: pd.DataFrame) -> None:
    data_path = tmp_path / "clean.csv"
    synthetic_clean.head(400).to_csv(data_path, index=False)
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    exit_code = train.main(
        [
            "--input",
            str(data_path),
            "--output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--models",
            "lightgbm",
            "--folds",
            "3",
            "--select-by",
            "rmse",
            "--lightgbm-estimators",
            "50",
        ]
    )
    assert exit_code == 0
    assert load_bundle(model_path).model_name is ModelName.LIGHTGBM
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["selected_by"] == "rmse"


def test_cli_catboost_override(tmp_path: Path, synthetic_clean: pd.DataFrame) -> None:
    data_path = tmp_path / "clean.csv"
    synthetic_clean.head(200).to_csv(data_path, index=False)
    exit_code = train.main(
        [
            "--input",
            str(data_path),
            "--output",
            str(tmp_path / "model.joblib"),
            "--metrics-output",
            str(tmp_path / "metrics.json"),
            "--models",
            "catboost",
            "--folds",
            "2",
            "--catboost-iterations",
            "30",
        ]
    )
    assert exit_code == 0
    assert load_bundle(tmp_path / "model.joblib").model_name is ModelName.CATBOOST


def test_cli_failures(tmp_path: Path, synthetic_clean: pd.DataFrame) -> None:
    assert train.main(["--input", str(tmp_path / "absent.csv")]) == 1
    tiny = tmp_path / "tiny.csv"
    synthetic_clean.head(5).to_csv(tiny, index=False)
    assert train.main(["--input", str(tiny), "--output", str(tmp_path / "m.joblib")]) == 1
    assert not (tmp_path / "m.joblib").exists()
