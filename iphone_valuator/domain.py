"""Domain vocabulary: listing conditions, the iPhone model catalog and release-date arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final, assert_never


class Condition(StrEnum):
    """Normalized physical condition of a listed device."""

    NEW = "new"
    USED = "used"
    REFURBISHED = "refurbished"
    FOR_PARTS = "for_parts"

    @property
    def label(self) -> str:
        match self:
            case Condition.NEW:
                return "brand-new"
            case Condition.USED:
                return "used"
            case Condition.REFURBISHED:
                return "refurbished"
            case Condition.FOR_PARTS:
                return "for-parts/broken"
            case _:
                assert_never(self)


class ModelTier(StrEnum):
    """Product line tier, shared across generations to let models borrow strength."""

    BASE = "base"
    MINI = "mini"
    PLUS = "plus"
    PRO = "pro"
    PRO_MAX = "pro_max"
    SE = "se"
    E = "e"
    AIR = "air"


@dataclass(frozen=True, slots=True)
class PhoneModel:
    """Static facts about a single iPhone model."""

    name: str
    tier: ModelTier
    release_date: date
    storage_options: tuple[int, ...]

    @property
    def release_year(self) -> int:
        return self.release_date.year

    def supports_storage(self, storage_gb: int) -> bool:
        return storage_gb in self.storage_options


IPHONE_MODELS: Final[tuple[PhoneModel, ...]] = (
    PhoneModel("iPhone 8", ModelTier.BASE, date(2017, 9, 22), (64, 128, 256)),
    PhoneModel("iPhone 8 Plus", ModelTier.PLUS, date(2017, 9, 22), (64, 128, 256)),
    PhoneModel("iPhone X", ModelTier.PRO, date(2017, 11, 3), (64, 256)),
    PhoneModel("iPhone XR", ModelTier.BASE, date(2018, 10, 26), (64, 128, 256)),
    PhoneModel("iPhone XS", ModelTier.PRO, date(2018, 9, 21), (64, 256, 512)),
    PhoneModel("iPhone XS Max", ModelTier.PRO_MAX, date(2018, 9, 21), (64, 256, 512)),
    PhoneModel("iPhone 11", ModelTier.BASE, date(2019, 9, 20), (64, 128, 256)),
    PhoneModel("iPhone 11 Pro", ModelTier.PRO, date(2019, 9, 20), (64, 256, 512)),
    PhoneModel("iPhone 11 Pro Max", ModelTier.PRO_MAX, date(2019, 9, 20), (64, 256, 512)),
    PhoneModel("iPhone SE 2020", ModelTier.SE, date(2020, 4, 24), (64, 128, 256)),
    PhoneModel("iPhone 12 mini", ModelTier.MINI, date(2020, 11, 13), (64, 128, 256)),
    PhoneModel("iPhone 12", ModelTier.BASE, date(2020, 10, 23), (64, 128, 256)),
    PhoneModel("iPhone 12 Pro", ModelTier.PRO, date(2020, 10, 23), (128, 256, 512)),
    PhoneModel("iPhone 12 Pro Max", ModelTier.PRO_MAX, date(2020, 11, 13), (128, 256, 512)),
    PhoneModel("iPhone 13 mini", ModelTier.MINI, date(2021, 9, 24), (128, 256, 512)),
    PhoneModel("iPhone 13", ModelTier.BASE, date(2021, 9, 24), (128, 256, 512)),
    PhoneModel("iPhone 13 Pro", ModelTier.PRO, date(2021, 9, 24), (128, 256, 512, 1024)),
    PhoneModel("iPhone 13 Pro Max", ModelTier.PRO_MAX, date(2021, 9, 24), (128, 256, 512, 1024)),
    PhoneModel("iPhone SE 2022", ModelTier.SE, date(2022, 3, 18), (64, 128, 256)),
    PhoneModel("iPhone 14", ModelTier.BASE, date(2022, 9, 16), (128, 256, 512)),
    PhoneModel("iPhone 14 Plus", ModelTier.PLUS, date(2022, 10, 7), (128, 256, 512)),
    PhoneModel("iPhone 14 Pro", ModelTier.PRO, date(2022, 9, 16), (128, 256, 512, 1024)),
    PhoneModel("iPhone 14 Pro Max", ModelTier.PRO_MAX, date(2022, 9, 16), (128, 256, 512, 1024)),
    PhoneModel("iPhone 15", ModelTier.BASE, date(2023, 9, 22), (128, 256, 512)),
    PhoneModel("iPhone 15 Plus", ModelTier.PLUS, date(2023, 9, 22), (128, 256, 512)),
    PhoneModel("iPhone 15 Pro", ModelTier.PRO, date(2023, 9, 22), (128, 256, 512, 1024)),
    PhoneModel("iPhone 15 Pro Max", ModelTier.PRO_MAX, date(2023, 9, 22), (256, 512, 1024)),
    PhoneModel("iPhone 16", ModelTier.BASE, date(2024, 9, 20), (128, 256, 512)),
    PhoneModel("iPhone 16 Plus", ModelTier.PLUS, date(2024, 9, 20), (128, 256, 512)),
    PhoneModel("iPhone 16 Pro", ModelTier.PRO, date(2024, 9, 20), (128, 256, 512, 1024)),
    PhoneModel("iPhone 16 Pro Max", ModelTier.PRO_MAX, date(2024, 9, 20), (256, 512, 1024)),
    PhoneModel("iPhone 16e", ModelTier.E, date(2025, 2, 28), (128, 256, 512)),
    PhoneModel("iPhone 17", ModelTier.BASE, date(2025, 9, 19), (256, 512)),
    PhoneModel("iPhone Air", ModelTier.AIR, date(2025, 9, 19), (256, 512, 1024)),
    PhoneModel("iPhone 17 Pro", ModelTier.PRO, date(2025, 9, 19), (256, 512, 1024)),
    PhoneModel("iPhone 17 Pro Max", ModelTier.PRO_MAX, date(2025, 9, 19), (256, 512, 1024, 2048)),
)

VALID_STORAGE_GB: Final[tuple[int, ...]] = tuple(
    sorted({storage for model in IPHONE_MODELS for storage in model.storage_options})
)

_MODELS_BY_KEY: Final[dict[str, PhoneModel]] = {
    model.name.casefold(): model for model in IPHONE_MODELS
}


def get_model(name: str | None) -> PhoneModel | None:
    """Look up a catalog model by its canonical name (case- and whitespace-insensitive)."""
    if not name:
        return None
    return _MODELS_BY_KEY.get(" ".join(name.split()).casefold())


def release_year_of(name: object) -> int | None:
    """Release year for a catalog model name, ``None`` for anything else (including ``NaN``)."""
    model = get_model(name) if isinstance(name, str) else None
    return model.release_year if model is not None else None


def require_model(name: str) -> PhoneModel:
    """Like :func:`get_model` but raises ``KeyError`` for names outside the catalog."""
    model = get_model(name)
    if model is None:
        raise KeyError(f"Unknown iPhone model: {name!r}")
    return model


def months_since_release(model: PhoneModel, as_of: date) -> int:
    """Whole months elapsed between the model's release and ``as_of`` (never negative)."""
    released = model.release_date
    months = (as_of.year - released.year) * 12 + (as_of.month - released.month)
    if as_of.day < released.day:
        months -= 1
    return max(months, 0)


def format_storage(storage_gb: int) -> str:
    """Render a capacity in gigabytes as a marketing label such as ``128GB`` or ``1TB``."""
    if storage_gb >= 1024 and storage_gb % 1024 == 0:
        return f"{storage_gb // 1024}TB"
    return f"{storage_gb}GB"
