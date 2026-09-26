from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from iphone_valuator.cleaner import CleaningConfig, ListingCleaner
from iphone_valuator.modeling import save_bundle
from iphone_valuator.synthetic import SyntheticConfig, generate_listings
from iphone_valuator.train import TrainingConfig, TrainingResult, train_models
from tests.helpers import FIXTURES_DIR, QUICK_MODEL_PARAMS, SCRAPED_AT


@pytest.fixture
def fixture_html() -> Callable[[str], str]:
    def load(name: str) -> str:
        return (FIXTURES_DIR / name).read_text(encoding="utf-8")

    return load


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return generate_listings(SyntheticConfig(rows=1500, seed=7, scraped_at=SCRAPED_AT))


@pytest.fixture(scope="session")
def synthetic_clean(synthetic_raw: pd.DataFrame) -> pd.DataFrame:
    config = CleaningConfig(reference_date=SCRAPED_AT.date())
    return ListingCleaner(config).clean(synthetic_raw).data


@pytest.fixture(scope="session")
def training_result(synthetic_clean: pd.DataFrame) -> TrainingResult:
    config = TrainingConfig(n_splits=3, model_params=QUICK_MODEL_PARAMS)
    return train_models(synthetic_clean, config)


@pytest.fixture(scope="session")
def model_path(training_result: TrainingResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    save_bundle(training_result.bundle, path)
    return path
