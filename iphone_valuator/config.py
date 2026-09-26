"""Project-wide paths, business thresholds and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

DATA_DIR: Final = Path("data")
RAW_DATA_PATH: Final = DATA_DIR / "raw" / "iphones_raw.csv"
CLEAN_DATA_PATH: Final = DATA_DIR / "processed" / "iphones_clean.csv"
ARTIFACTS_DIR: Final = Path("artifacts")
MODEL_PATH: Final = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH: Final = ARTIFACTS_DIR / "metrics.json"
BROWSER_STATE_PATH: Final = Path(".browser_state") / "avito_storage_state.json"

AVITO_BASE_URL: Final = "https://www.avito.ru"
AVITO_PHONES_PATH: Final = "telefony/mobilnye_telefony"
DEFAULT_REGION_SLUG: Final = "rossiya"
DEFAULT_SEARCH_QUERY: Final = "iphone"

MIN_PRICE_RUB: Final = 5_000
MAX_PRICE_RUB: Final = 400_000
MIN_BATTERY_HEALTH: Final = 40
MAX_BATTERY_HEALTH: Final = 100
NEW_DEVICE_BATTERY_HEALTH: Final = 100
FALLBACK_BATTERY_HEALTH: Final = 90.0
UNKNOWN_REGION: Final = "Неизвестно"
RANDOM_STATE: Final = 42

LOG_FORMAT: Final = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for CLI entry points."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
