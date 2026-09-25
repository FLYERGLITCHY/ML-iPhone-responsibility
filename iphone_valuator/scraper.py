"""Avito iPhone scraper: search pagination, listing pages, attribute extraction and CSV export.

Usage::

    python -m iphone_valuator.scraper --region moskva --max-pages 5
    python -m iphone_valuator.scraper --search-url "https://www.avito.ru/..." \
        --headful --manual-captcha
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from bs4.element import Tag

from iphone_valuator.browser import (
    BlockedError,
    BrowserOptions,
    FetchError,
    PageFetcher,
    PlaywrightFetcher,
)
from iphone_valuator.config import (
    AVITO_BASE_URL,
    AVITO_PHONES_PATH,
    BROWSER_STATE_PATH,
    DEFAULT_REGION_SLUG,
    DEFAULT_SEARCH_QUERY,
    RAW_DATA_PATH,
    configure_logging,
)
from iphone_valuator.schemas import RAW_COLUMNS, RawListing
from iphone_valuator.text_parsing import (
    find_param,
    parse_battery_health,
    parse_condition,
    parse_model,
    parse_price,
    parse_storage,
)

logger = logging.getLogger(__name__)

SEARCH_READY_SELECTOR: Final = '[data-marker="item"]'
ITEM_READY_SELECTOR: Final = '[data-marker="item-view/title-info"], h1'

_CARD_SELECTOR: Final = '[data-marker="item"]'
_CARD_LINK_SELECTORS: Final = ('a[data-marker="item-title"]', 'a[itemprop="url"]')
_CARD_PRICE_SELECTORS: Final = ('[data-marker="item-price"]',)
_CARD_LOCATION_SELECTORS: Final = (
    '[data-marker="item-address"]',
    '[data-marker="item-location"]',
    '[class*="geo-root"]',
    '[class*="geo-address"]',
)
_CARD_SNIPPET_SELECTORS: Final = (
    '[data-marker="item-description"]',
    '[class*="descriptionStep"]',
    '[class*="item-description"]',
)
_PAGINATION_SELECTOR: Final = (
    '[data-marker="pagination-button"], nav[aria-label*="агинац"], [class*="pagination-root"]'
)
_NEXT_PAGE_SELECTOR: Final = (
    '[data-marker="pagination-button/nextPage"], '
    '[data-marker="pagination-button/next"], a[rel="next"]'
)
_ITEM_TITLE_SELECTORS: Final = (
    '[data-marker="item-view/title-info"]',
    'h1[itemprop="name"]',
    "h1",
)
_ITEM_PRICE_SELECTORS: Final = ('[data-marker="item-view/item-price"]', '[itemprop="price"]')
_ITEM_DESCRIPTION_SELECTORS: Final = (
    '[data-marker="item-view/item-description"]',
    '[itemprop="description"]',
)
_ITEM_ADDRESS_SELECTORS: Final = (
    '[data-marker="item-view/item-address"]',
    '[itemprop="address"]',
    '[class*="item-address__string"]',
    '[class*="style-item-address"]',
)
_ITEM_PARAM_SELECTORS: Final = (
    '[data-marker="item-view/item-params"] li',
    '[class*="params-paramsList"] li',
    "#bx_item-params li",
)
_ITEM_ID_FROM_URL_RE: Final = re.compile(r"_(\d{6,})/?$")


@dataclass(frozen=True, slots=True)
class SearchCard:
    """A listing preview as rendered on a search results page."""

    item_id: str
    url: str
    title: str
    price: int | None
    location: str
    snippet: str


@dataclass(frozen=True, slots=True)
class SearchPage:
    cards: list[SearchCard]
    has_next: bool


@dataclass(frozen=True, slots=True)
class ItemDetails:
    """Fields available only on the listing page itself."""

    title: str
    price: int | None
    description: str
    location: str
    params: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScrapeConfig:
    """What to crawl and how politely."""

    search_url: str
    max_pages: int = 5
    max_listings: int | None = None
    fetch_details: bool = True
    min_delay_s: float = 3.0
    max_delay_s: float = 8.0
    long_pause_every: int = 15
    long_pause_range_s: tuple[float, float] = (20.0, 45.0)

    def __post_init__(self) -> None:
        if self.max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if self.max_listings is not None and self.max_listings < 1:
            raise ValueError("max_listings must be at least 1")
        if not 0 <= self.min_delay_s <= self.max_delay_s:
            raise ValueError("delays must satisfy 0 <= min_delay <= max_delay")


def build_search_url(region: str = DEFAULT_REGION_SLUG, query: str = DEFAULT_SEARCH_QUERY) -> str:
    """Avito mobile-phones search URL for a region slug, sorted by publication date."""
    params = urlencode({"q": query, "s": "104"})
    return f"{AVITO_BASE_URL}/{region.strip('/')}/{AVITO_PHONES_PATH}?{params}"


def build_page_url(search_url: str, page: int) -> str:
    """Set Avito's ``p`` pagination parameter while preserving every other filter."""
    if page < 1:
        raise ValueError("page numbers start at 1")
    parts = urlsplit(search_url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "p"]
    if page > 1:
        query.append(("p", str(page)))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _attr(tag: Tag | None, name: str) -> str | None:
    if tag is None:
        return None
    value = tag.get(name)
    if isinstance(value, list):
        value = " ".join(value)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _text(tag: Tag | None) -> str:
    return " ".join(tag.get_text(" ", strip=True).split()) if tag is not None else ""


def _select_first(root: Tag, selectors: Sequence[str]) -> Tag | None:
    for selector in selectors:
        found = root.select_one(selector)
        if found is not None:
            return found
    return None


def _first_text(root: Tag, selectors: Sequence[str]) -> str:
    for selector in selectors:
        text = _text(root.select_one(selector))
        if text:
            return text
    return ""


def _item_id_from_url(url: str) -> str | None:
    match = _ITEM_ID_FROM_URL_RE.search(urlsplit(url).path)
    return match.group(1) if match else None


def _parse_card(node: Tag, base_url: str) -> SearchCard | None:
    link = _select_first(node, _CARD_LINK_SELECTORS)
    href = _attr(link, "href")
    if href is None:
        return None
    url = urljoin(base_url, href).split("#", 1)[0].split("?", 1)[0]
    item_id = _attr(node, "data-item-id") or _item_id_from_url(url)
    if item_id is None:
        return None
    price = parse_price(_attr(node.select_one('meta[itemprop="price"]'), "content"))
    return SearchCard(
        item_id=item_id,
        url=url,
        title=_text(link) or _attr(link, "title") or _first_text(node, ('[itemprop="name"]',)),
        price=price
        if price is not None
        else parse_price(_first_text(node, _CARD_PRICE_SELECTORS)),
        location=_first_text(node, _CARD_LOCATION_SELECTORS),
        snippet=_attr(node.select_one('meta[itemprop="description"]'), "content")
        or _first_text(node, _CARD_SNIPPET_SELECTORS),
    )


def _has_next_page(soup: BeautifulSoup) -> bool:
    pagination = soup.select_one(_PAGINATION_SELECTOR)
    if pagination is None:
        return True
    next_button = pagination.select_one(_NEXT_PAGE_SELECTOR) or soup.select_one(
        _NEXT_PAGE_SELECTOR
    )
    if next_button is None:
        return False
    return _attr(next_button, "aria-disabled") != "true" and not next_button.has_attr("disabled")


def parse_search_page(html: str, base_url: str = AVITO_BASE_URL) -> SearchPage:
    """Extract listing cards and pagination state from a search results page."""
    soup = BeautifulSoup(html, "lxml")
    cards: dict[str, SearchCard] = {}
    for node in soup.select(_CARD_SELECTOR):
        card = _parse_card(node, base_url)
        if card is not None and card.item_id not in cards:
            cards[card.item_id] = card
    return SearchPage(cards=list(cards.values()), has_next=_has_next_page(soup))


def _json_ld_product(soup: BeautifulSoup) -> dict[str, object]:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        for candidate in payload if isinstance(payload, list) else [payload]:
            if isinstance(candidate, dict) and candidate.get("@type") == "Product":
                return candidate
    return {}


def _json_ld_price(product: dict[str, object]) -> object:
    offers = product.get("offers")
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    return offers.get("price") if isinstance(offers, dict) else None


def _parse_params(soup: BeautifulSoup) -> dict[str, str]:
    params: dict[str, str] = {}
    for selector in _ITEM_PARAM_SELECTORS:
        for item in soup.select(selector):
            key, separator, value = _text(item).partition(":")
            if separator and key.strip() and value.strip():
                params.setdefault(key.strip(), value.strip())
        if params:
            break
    return params


def parse_item_page(html: str) -> ItemDetails:
    """Extract title, price, full description, address and attributes from a listing page."""
    soup = BeautifulSoup(html, "lxml")
    product = _json_ld_product(soup)
    price_node = _select_first(soup, _ITEM_PRICE_SELECTORS)
    price = parse_price(_attr(price_node, "content"))
    if price is None:
        price = parse_price(_text(price_node)) or parse_price(_json_ld_price(product))
    description_node = _select_first(soup, _ITEM_DESCRIPTION_SELECTORS)
    description = (
        description_node.get_text("\n", strip=True)
        if description_node is not None
        else str(product.get("description") or "")
    )
    return ItemDetails(
        title=_first_text(soup, _ITEM_TITLE_SELECTORS) or str(product.get("name") or ""),
        price=price,
        description=description,
        location=_first_text(soup, _ITEM_ADDRESS_SELECTORS),
        params=_parse_params(soup),
    )


def build_listing(
    card: SearchCard, details: ItemDetails | None, scraped_at: datetime
) -> RawListing:
    """Merge card and page data, preferring structured attributes over free text."""
    title = (details.title if details else "") or card.title
    description = (details.description if details else "") or card.snippet
    location = (details.location if details else "") or card.location
    price = details.price if details is not None and details.price is not None else card.price
    params = dict(details.params) if details else {}
    params_text = "\n".join(f"{key}: {value}" for key, value in params.items())
    storage_param = find_param(params, "встроенная память", "объем встроенной памяти", "встроен")
    condition = parse_condition(find_param(params, "состояние"), f"{title}\n{description}")
    return RawListing(
        item_id=card.item_id,
        listing_url=card.url,
        title=title,
        model=parse_model(find_param(params, "модель"), require_prefix=False)
        or parse_model(title)
        or parse_model(description),
        storage_gb=parse_storage(storage_param)
        or parse_storage(title, allow_bare=True)
        or parse_storage(description),
        condition=condition.value if condition is not None else None,
        battery_health=parse_battery_health(params_text)
        or parse_battery_health(description)
        or parse_battery_health(title),
        price=price,
        description=description,
        location=location,
        params=params,
        scraped_at=scraped_at.isoformat(timespec="seconds"),
    )


class CsvListingWriter:
    """Append-only CSV sink; already-saved item IDs are skipped so interrupted runs can resume."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._known_ids = self._read_known_ids(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def known_ids(self) -> frozenset[str]:
        return frozenset(self._known_ids)

    def write(self, listing: RawListing) -> bool:
        if listing.item_id in self._known_ids:
            return False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(listing.to_record())
        self._known_ids.add(listing.item_id)
        return True

    @staticmethod
    def _read_known_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["item_id"] for row in csv.DictReader(handle) if row.get("item_id")}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else AVITO_BASE_URL


class AvitoScraper:
    """Walks search pages and listing pages through any :class:`PageFetcher`."""

    def __init__(
        self,
        fetcher: PageFetcher,
        config: ScrapeConfig,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._fetcher = fetcher
        self._config = config
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._clock = clock
        self._requests = 0
        self._base_url = _origin(config.search_url)

    async def scrape(self, writer: CsvListingWriter | None = None) -> list[RawListing]:
        """Collect new listings; stops on the last page, repeated pages, limits or hard blocks."""
        known = set(writer.known_ids) if writer is not None else set()
        seen_this_run: set[str] = set()
        collected: list[RawListing] = []
        try:
            for page_number in range(1, self._config.max_pages + 1):
                page = await self._load_search_page(page_number)
                if page is None:
                    break
                fresh = [card for card in page.cards if card.item_id not in seen_this_run]
                if not fresh:
                    logger.info("Search page %d repeats earlier results; stopping", page_number)
                    break
                seen_this_run.update(card.item_id for card in fresh)
                for card in fresh:
                    if card.item_id in known:
                        continue
                    listing = await self._scrape_card(card)
                    known.add(card.item_id)
                    collected.append(listing)
                    if writer is not None:
                        writer.write(listing)
                    if self._reached_limit(len(collected)):
                        return collected
                if not page.has_next:
                    logger.info("Reached the last search page (%d)", page_number)
                    break
        except BlockedError as error:
            logger.error("Stopping early, persistently blocked by anti-bot protection: %s", error)
        return collected

    async def _load_search_page(self, page_number: int) -> SearchPage | None:
        url = build_page_url(self._config.search_url, page_number)
        logger.info("Search page %d: %s", page_number, url)
        try:
            html = await self._fetch(url, SEARCH_READY_SELECTOR)
        except BlockedError:
            raise
        except FetchError as error:
            logger.error("Could not load search page %d: %s", page_number, error)
            return None
        page = parse_search_page(html, self._base_url)
        if not page.cards:
            logger.info("No listings on search page %d; stopping", page_number)
            return None
        return page

    async def _scrape_card(self, card: SearchCard) -> RawListing:
        details: ItemDetails | None = None
        if self._config.fetch_details:
            try:
                details = parse_item_page(await self._fetch(card.url, ITEM_READY_SELECTOR))
            except BlockedError:
                raise
            except FetchError as error:
                logger.warning("Keeping search-card data only for %s: %s", card.url, error)
        return build_listing(card, details, self._clock())

    async def _fetch(self, url: str, wait_for: str) -> str:
        if self._requests:
            await self._sleep(self._next_delay())
        self._requests += 1
        return await self._fetcher.fetch(url, wait_for=wait_for)

    def _next_delay(self) -> float:
        delay = self._rng.uniform(self._config.min_delay_s, self._config.max_delay_s)
        every = self._config.long_pause_every
        if every > 0 and self._requests % every == 0:
            delay += self._rng.uniform(*self._config.long_pause_range_s)
        return delay

    def _reached_limit(self, count: int) -> bool:
        return self._config.max_listings is not None and count >= self._config.max_listings


async def run_scraper(config: ScrapeConfig, options: BrowserOptions, output: Path) -> int:
    """Scrape with a real browser and append new listings to ``output``; returns the count."""
    writer = CsvListingWriter(output)
    async with PlaywrightFetcher(options) as fetcher:
        listings = await AvitoScraper(fetcher, config).scrape(writer)
    return len(listings)


def _read_proxy_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iphone-scrape", description="Scrape iPhone listings from Avito into a raw CSV."
    )
    parser.add_argument("--search-url", help="full Avito search URL (overrides --region/--query)")
    parser.add_argument(
        "--region", default=DEFAULT_REGION_SLUG, help="Avito region slug: moskva, rossiya, ..."
    )
    parser.add_argument("--query", default=DEFAULT_SEARCH_QUERY, help="search query")
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-listings", type=int, default=None)
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="skip listing pages (faster, but descriptions are truncated)",
    )
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--min-delay", type=float, default=3.0, help="seconds between requests")
    parser.add_argument("--max-delay", type=float, default=8.0, help="seconds between requests")
    parser.add_argument("--headful", action="store_true", help="show the browser window")
    parser.add_argument(
        "--manual-captcha",
        action="store_true",
        help="with --headful, wait for you to solve captchas in the browser",
    )
    parser.add_argument("--channel", default=None, help="browser channel, e.g. 'chrome'")
    parser.add_argument(
        "--proxy", action="append", default=[], help="proxy URL (repeatable, rotated on blocks)"
    )
    parser.add_argument("--proxy-file", type=Path, help="file with one proxy URL per line")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=BROWSER_STATE_PATH,
        help="cookie/localStorage file reused between runs",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = ScrapeConfig(
            search_url=args.search_url or build_search_url(args.region, args.query),
            max_pages=args.max_pages,
            max_listings=args.max_listings,
            fetch_details=not args.no_details,
            min_delay_s=args.min_delay,
            max_delay_s=args.max_delay,
        )
        options = BrowserOptions(
            headless=not args.headful,
            channel=args.channel,
            proxies=(*args.proxy, *_read_proxy_file(args.proxy_file)),
            storage_state_path=args.state_file,
            max_retries=args.max_retries,
            manual_captcha=args.manual_captcha,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        count = asyncio.run(run_scraper(config, options, args.output))
    except KeyboardInterrupt:
        logger.warning("Interrupted; everything scraped so far is saved in %s", args.output)
        return 130
    logger.info("Saved %d new listings to %s", count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
