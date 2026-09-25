"""Tabular schemas shared between pipeline stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Final


@dataclass(frozen=True, slots=True)
class RawListing:
    """One scraped listing with best-effort parsed attributes (``None`` when unknown)."""

    item_id: str
    listing_url: str
    title: str
    model: str | None
    storage_gb: int | None
    condition: str | None
    battery_health: int | None
    price: int | None
    description: str
    location: str
    params: dict[str, str] = field(default_factory=dict)
    scraped_at: str = ""

    def to_record(self) -> dict[str, str | int | None]:
        record: dict[str, str | int | None] = {
            key: value for key, value in asdict(self).items() if key != "params"
        }
        record["params"] = json.dumps(self.params, ensure_ascii=False)
        return record


RAW_COLUMNS: Final[tuple[str, ...]] = tuple(f.name for f in fields(RawListing))

CLEAN_COLUMNS: Final[tuple[str, ...]] = (
    "item_id",
    "listing_url",
    "title",
    "model",
    "storage_gb",
    "condition",
    "battery_health",
    "battery_known",
    "price",
    "location",
    "region",
    "has_box",
    "has_receipt",
    "release_year",
    "phone_age_months",
    "description",
    "scraped_at",
)
