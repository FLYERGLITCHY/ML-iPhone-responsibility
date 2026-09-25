from __future__ import annotations

import asyncio
import csv
import json
import random
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

import pytest

from iphone_valuator import scraper
from iphone_valuator.browser import BlockedError, BrowserOptions, FetchError
from iphone_valuator.schemas import RAW_COLUMNS, RawListing
from iphone_valuator.scraper import (
    ITEM_READY_SELECTOR,
    SEARCH_READY_SELECTOR,
    AvitoScraper,
    CsvListingWriter,
    ItemDetails,
    ScrapeConfig,
    SearchCard,
    build_listing,
    build_page_url,
    build_search_url,
    parse_item_page,
    parse_search_page,
)
from tests.helpers import SCRAPED_AT, item_page_html, search_page_html

SEARCH_URL = "https://www.avito.ru/moskva/telefony/mobilnye_telefony?q=iphone&s=104"


class FakeFetcher:
    """Serves canned HTML per URL and records every request."""

    def __init__(self, pages: Mapping[str, str | Exception]) -> None:
        self.pages = dict(pages)
        self.calls: list[tuple[str, str | None]] = []

    async def fetch(self, url: str, *, wait_for: str | None = None) -> str:
        self.calls.append((url, wait_for))
        result = self.pages.get(url)
        if result is None:
            raise FetchError(f"unexpected url {url}")
        if isinstance(result, Exception):
            raise result
        return result


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def page_url(page: int) -> str:
    return build_page_url(SEARCH_URL, page)


def item_url(item_id: str) -> str:
    return f"https://www.avito.ru/moskva/telefony/iphone_{item_id}"


def make_scraper(
    pages: Mapping[str, str | Exception], **config: object
) -> tuple[AvitoScraper, FakeFetcher, SleepRecorder]:
    fetcher = FakeFetcher(pages)
    sleep = SleepRecorder()
    settings: dict[str, object] = {
        "search_url": SEARCH_URL,
        "min_delay_s": 1.0,
        "max_delay_s": 2.0,
    }
    settings.update(config)
    instance = AvitoScraper(
        fetcher,
        ScrapeConfig(**settings),
        sleep=sleep,
        rng=random.Random(0),
        clock=lambda: SCRAPED_AT,
    )
    return instance, fetcher, sleep


def test_build_search_url() -> None:
    assert build_search_url("moskva", "iphone 13") == (
        "https://www.avito.ru/moskva/telefony/mobilnye_telefony?q=iphone+13&s=104"
    )


def test_build_page_url_sets_replaces_and_removes_page_param() -> None:
    assert build_page_url(SEARCH_URL, 1) == SEARCH_URL
    assert build_page_url(SEARCH_URL, 3) == f"{SEARCH_URL}&p=3"
    assert build_page_url(f"{SEARCH_URL}&p=7", 2) == f"{SEARCH_URL}&p=2"
    assert build_page_url(f"{SEARCH_URL}&p=7", 1) == SEARCH_URL
    with pytest.raises(ValueError, match="start at 1"):
        build_page_url(SEARCH_URL, 0)


def test_parse_search_page(fixture_html: Callable[[str], str]) -> None:
    page = parse_search_page(fixture_html("avito_search_page.html"))
    assert page.has_next is True
    assert [card.item_id for card in page.cards] == ["4312345678", "4312345999"]
    first, second = page.cards
    assert first == SearchCard(
        item_id="4312345678",
        url="https://www.avito.ru/moskva/telefony/iphone_13_pro_256_gb_4312345678",
        title="iPhone 13 Pro, 256 ГБ",
        price=54990,
        location="Москва",
        snippet="Отличное состояние, АКБ 89%, полный комплект.",
    )
    assert second.url.endswith("iphone_12_128gb_4312345999")
    assert second.price == 32000
    assert second.location == "Москва, м. Тверская"
    assert second.snippet == "Б/у, аккумулятор 83%, без коробки."


def test_parse_search_page_detects_last_page(fixture_html: Callable[[str], str]) -> None:
    page = parse_search_page(fixture_html("avito_search_last_page.html"))
    assert page.has_next is False
    assert [card.price for card in page.cards] == [79000]


def test_parse_search_page_without_pagination_assumes_more_pages() -> None:
    page = parse_search_page("<html><body><p>layout changed</p></body></html>")
    assert page.cards == []
    assert page.has_next is True


def test_parse_item_page(fixture_html: Callable[[str], str]) -> None:
    details = parse_item_page(fixture_html("avito_item_page.html"))
    assert details.title == "iPhone 13 Pro, 256 ГБ"
    assert details.price == 54990
    assert details.location == "Москва, Пресненская наб., 12"
    assert details.params == {
        "Производитель": "Apple",
        "Модель": "iPhone 13 Pro",
        "Встроенная память": "256 ГБ",
        "Цвет": "графитовый",
        "Состояние": "Отличное",
    }
    assert details.description.splitlines() == [
        "Продаю iPhone 13 Pro в отличном состоянии.",
        "АКБ 89%, Face ID работает.",
        "Полный комплект: коробка, чек. Не битый, не утопленник.",
    ]


def test_parse_item_page_falls_back_to_json_ld() -> None:
    payload = {
        "@type": "Product",
        "name": "iPhone 14, 128 ГБ",
        "description": "АКБ 90%",
        "offers": [{"price": "41000", "priceCurrency": "RUB"}],
    }
    html = (
        '<html><head><script type="application/ld+json">not json</script>'
        f'<script type="application/ld+json">{json.dumps(payload, ensure_ascii=False)}</script>'
        "</head><body></body></html>"
    )
    details = parse_item_page(html)
    assert details == ItemDetails(
        title="iPhone 14, 128 ГБ", price=41000, description="АКБ 90%", location="", params={}
    )


def test_build_listing_prefers_structured_params() -> None:
    card = SearchCard("1", "https://www.avito.ru/x_1", "Телефон", 30000, "м. Тверская", "snippet")
    details = ItemDetails(
        title="Продаю айфон",
        price=31000,
        description="Состояние хорошее. Коробка есть.",
        location="Москва, ул. Тверская",
        params={
            "Модель": "iPhone 13 Pro",
            "Встроенная память": "256 ГБ",
            "Состояние": "Хорошее",
            "Ёмкость аккумулятора": "84%",
        },
    )
    listing = build_listing(card, details, SCRAPED_AT)
    assert listing == RawListing(
        item_id="1",
        listing_url="https://www.avito.ru/x_1",
        title="Продаю айфон",
        model="iPhone 13 Pro",
        storage_gb=256,
        condition="used",
        battery_health=84,
        price=31000,
        description="Состояние хорошее. Коробка есть.",
        location="Москва, ул. Тверская",
        params=details.params,
        scraped_at="2026-09-01T12:00:00+00:00",
    )


def test_build_listing_from_card_only_parses_title_and_snippet() -> None:
    card = SearchCard(
        "2",
        "https://www.avito.ru/x_2",
        "Айфон 12 мини 128",
        25000,
        "Казань",
        "На запчасти, акб 70%",
    )
    listing = build_listing(card, None, SCRAPED_AT)
    assert listing.model == "iPhone 12 mini"
    assert listing.storage_gb == 128
    assert listing.condition == "for_parts"
    assert listing.battery_health == 70
    assert listing.params == {}
    assert listing.price == 25000


def test_raw_listing_record_serializes_params_as_json() -> None:
    listing = build_listing(SearchCard("3", "u", "iPhone 15", 1, "", ""), None, SCRAPED_AT)
    record = listing.to_record()
    assert set(record) == set(RAW_COLUMNS)
    assert record["params"] == "{}"


def test_csv_writer_appends_dedupes_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "listings.csv"
    first = build_listing(
        SearchCard("10", "u10", "iPhone 13, 128 ГБ", 1, "", ""), None, SCRAPED_AT
    )
    second = build_listing(
        SearchCard("11", "u11", "iPhone 14, 128 ГБ", 2, "", ""), None, SCRAPED_AT
    )
    writer = CsvListingWriter(path)
    assert writer.write(first) is True
    assert writer.write(first) is False
    resumed = CsvListingWriter(path)
    assert resumed.known_ids == frozenset({"10"})
    assert resumed.write(second) is True
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["item_id"] for row in rows] == ["10", "11"]
    assert rows[1]["model"] == "iPhone 14"


def test_scrape_config_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        ScrapeConfig(search_url=SEARCH_URL, max_pages=0)
    with pytest.raises(ValueError, match="max_listings"):
        ScrapeConfig(search_url=SEARCH_URL, max_listings=0)
    with pytest.raises(ValueError, match="delays"):
        ScrapeConfig(search_url=SEARCH_URL, min_delay_s=5, max_delay_s=1)


def test_scraper_walks_pages_and_listing_pages(tmp_path: Path) -> None:
    pages: dict[str, str | Exception] = {
        page_url(1): search_page_html(
            [("101", "iPhone 13, 128 ГБ", 40000), ("102", "iPhone 14 Pro, 256 ГБ", 70000)],
            has_next=True,
        ),
        page_url(2): search_page_html([("103", "iPhone 12, 64 ГБ", 20000)], has_next=False),
        item_url("101"): item_page_html("iPhone 13, 128 ГБ", 41000, "АКБ 90%. Коробка, чек."),
        item_url("102"): item_page_html(
            "iPhone 14 Pro, 256 ГБ", 70000, "Новый", {"Состояние": "Новое"}
        ),
        item_url("103"): item_page_html("iPhone 12, 64 ГБ", 20000, "акб 81"),
    }
    instance, fetcher, sleep = make_scraper(pages)
    writer = CsvListingWriter(tmp_path / "raw.csv")
    listings = asyncio.run(instance.scrape(writer))
    assert [listing.item_id for listing in listings] == ["101", "102", "103"]
    assert listings[0].price == 41000
    assert listings[0].battery_health == 90
    assert listings[1].condition == "new"
    assert listings[2].location == "Санкт-Петербург, Невский пр-т"
    assert [call[1] for call in fetcher.calls] == [
        SEARCH_READY_SELECTOR,
        ITEM_READY_SELECTOR,
        ITEM_READY_SELECTOR,
        SEARCH_READY_SELECTOR,
        ITEM_READY_SELECTOR,
    ]
    assert len(sleep.delays) == len(fetcher.calls) - 1
    assert all(1.0 <= delay <= 2.0 for delay in sleep.delays)
    assert writer.known_ids == frozenset({"101", "102", "103"})


def test_scraper_stops_when_avito_repeats_the_last_page() -> None:
    repeated = search_page_html([("201", "iPhone 13, 128 ГБ", 40000)], has_next=True)
    pages: dict[str, str | Exception] = {page_url(1): repeated, page_url(2): repeated}
    instance, fetcher, _ = make_scraper(pages, fetch_details=False, max_pages=5)
    listings = asyncio.run(instance.scrape())
    assert [listing.item_id for listing in listings] == ["201"]
    assert [call[0] for call in fetcher.calls] == [page_url(1), page_url(2)]


def test_scraper_resume_skips_known_ids_but_keeps_paginating(tmp_path: Path) -> None:
    writer = CsvListingWriter(tmp_path / "raw.csv")
    writer.write(build_listing(SearchCard("301", "u", "iPhone 13", 1, "", ""), None, SCRAPED_AT))
    pages: dict[str, str | Exception] = {
        page_url(1): search_page_html([("301", "iPhone 13, 128 ГБ", 40000)], has_next=True),
        page_url(2): search_page_html([("302", "iPhone 15, 128 ГБ", 60000)], has_next=False),
    }
    instance, _, _ = make_scraper(pages, fetch_details=False)
    listings = asyncio.run(instance.scrape(writer))
    assert [listing.item_id for listing in listings] == ["302"]
    assert writer.known_ids == frozenset({"301", "302"})


def test_scraper_respects_max_listings() -> None:
    cards = [(str(400 + i), "iPhone 13, 128 ГБ", 40000) for i in range(5)]
    pages: dict[str, str | Exception] = {page_url(1): search_page_html(cards, has_next=True)}
    instance, fetcher, _ = make_scraper(pages, fetch_details=False, max_listings=2)
    listings = asyncio.run(instance.scrape())
    assert len(listings) == 2
    assert len(fetcher.calls) == 1


def test_scraper_keeps_card_data_when_listing_page_fails() -> None:
    pages: dict[str, str | Exception] = {
        page_url(1): search_page_html([("501", "iPhone 13, 128 ГБ", 40000)], has_next=False),
        item_url("501"): FetchError("timeout"),
    }
    instance, _, _ = make_scraper(pages)
    listings = asyncio.run(instance.scrape())
    assert len(listings) == 1
    assert listings[0].price == 40000
    assert listings[0].model == "iPhone 13"
    assert listings[0].params == {}


def test_scraper_stops_gracefully_when_blocked() -> None:
    pages: dict[str, str | Exception] = {
        page_url(1): search_page_html(
            [("601", "iPhone 13, 128 ГБ", 40000), ("602", "iPhone 13, 256 ГБ", 45000)],
            has_next=True,
        ),
        item_url("601"): item_page_html("iPhone 13, 128 ГБ", 40000, "ok"),
        item_url("602"): BlockedError("captcha"),
    }
    instance, fetcher, _ = make_scraper(pages)
    listings = asyncio.run(instance.scrape())
    assert [listing.item_id for listing in listings] == ["601"]
    assert page_url(2) not in [call[0] for call in fetcher.calls]


@pytest.mark.parametrize("failure", [FetchError("down"), BlockedError("captcha")])
def test_scraper_returns_nothing_when_first_search_page_fails(failure: Exception) -> None:
    instance, _, _ = make_scraper({page_url(1): failure})
    assert asyncio.run(instance.scrape()) == []


def test_scraper_stops_on_empty_search_page() -> None:
    instance, fetcher, _ = make_scraper({page_url(1): "<html><body></body></html>"})
    assert asyncio.run(instance.scrape()) == []
    assert len(fetcher.calls) == 1


def test_long_pause_is_added_periodically() -> None:
    cards = [(str(700 + i), "iPhone 13, 128 ГБ", 40000) for i in range(4)]
    pages: dict[str, str | Exception] = {page_url(1): search_page_html(cards, has_next=False)}
    pages.update({item_url(card[0]): item_page_html(card[1], card[2], "") for card in cards})
    instance, _, sleep = make_scraper(pages, long_pause_every=2, long_pause_range_s=(100.0, 100.0))
    asyncio.run(instance.scrape())
    assert sum(delay >= 100 for delay in sleep.delays) == 2


def test_main_wires_cli_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def fake_run(config: ScrapeConfig, options: BrowserOptions, output: Path) -> int:
        captured.update(config=config, options=options, output=output)
        return 3

    proxy_file = tmp_path / "proxies.txt"
    proxy_file.write_text("# comment\nhttp://p2:8080\n\n", encoding="utf-8")
    monkeypatch.setattr(scraper, "run_scraper", fake_run)
    exit_code = scraper.main(
        [
            "--region",
            "sankt-peterburg",
            "--query",
            "iphone 15",
            "--max-pages",
            "2",
            "--no-details",
            "--headful",
            "--proxy",
            "http://user:pass@p1:3128",
            "--proxy-file",
            str(proxy_file),
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    assert exit_code == 0
    config = captured["config"]
    options = captured["options"]
    assert isinstance(config, ScrapeConfig)
    assert isinstance(options, BrowserOptions)
    assert config.search_url == build_search_url("sankt-peterburg", "iphone 15")
    assert config.max_pages == 2
    assert config.fetch_details is False
    assert options.headless is False
    assert options.proxies == ("http://user:pass@p1:3128", "http://p2:8080")
    assert captured["output"] == tmp_path / "out.csv"


def test_main_rejects_invalid_configuration() -> None:
    with pytest.raises(SystemExit) as excinfo:
        scraper.main(["--max-pages", "0"])
    assert excinfo.value.code == 2


def test_main_reports_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def interrupted(config: ScrapeConfig, options: BrowserOptions, output: Path) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(scraper, "run_scraper", interrupted)
    assert scraper.main(["--output", str(tmp_path / "out.csv")]) == 130


def test_clock_default_is_timezone_aware() -> None:
    instance = AvitoScraper(FakeFetcher({}), ScrapeConfig(search_url=SEARCH_URL))
    assert isinstance(instance._clock(), datetime)
    assert instance._clock().tzinfo is not None
