from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence

import pytest
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from iphone_valuator import browser
from iphone_valuator.browser import (
    BlockedError,
    BrowserOptions,
    FetchError,
    PlaywrightFetcher,
    build_user_agent,
    is_block_page,
    parse_proxy,
)


class ScriptedFetcher(PlaywrightFetcher):
    """Replaces the browser round-trip with scripted outcomes to test retry policy."""

    def __init__(self, outcomes: Sequence[str | Exception], options: BrowserOptions) -> None:
        super().__init__(options, rng=random.Random(0))
        self.outcomes = list(outcomes)
        self.recoveries: list[int] = []
        self.resets = 0

    async def _fetch_once(self, url: str, wait_for: str | None) -> str:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def _recover_from_block(self, attempt: int) -> None:
        self.recoveries.append(attempt)

    async def _reset_page(self) -> None:
        self.resets += 1


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(browser.asyncio, "sleep", fake_sleep)
    return delays


def test_is_block_page_detects_firewall(fixture_html: Callable[[str], str]) -> None:
    assert is_block_page(fixture_html("avito_block_page.html")) is True
    assert is_block_page("<html><head><title>Доступ ограничен</title></head></html>") is True
    assert is_block_page(fixture_html("avito_search_page.html")) is False
    assert is_block_page(fixture_html("avito_item_page.html")) is False


def test_parse_proxy_with_credentials() -> None:
    assert parse_proxy("http://us%40er:p%3Ass@10.0.0.1:3128") == {
        "server": "http://10.0.0.1:3128",
        "username": "us@er",
        "password": "p:ss",
    }


def test_parse_proxy_without_credentials_or_port() -> None:
    assert parse_proxy("socks5://proxy.local") == {"server": "socks5://proxy.local"}


@pytest.mark.parametrize("value", ["", "proxy.local:8080", "http://"])
def test_parse_proxy_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid proxy"):
        parse_proxy(value)


def test_build_user_agent_matches_engine_without_headless_marker() -> None:
    agent = build_user_agent("153.0.8010.12")
    assert "Chrome/153.0.0.0" in agent
    assert "Headless" not in agent


def test_fetch_returns_first_success(recorded_sleeps: list[float]) -> None:
    fetcher = ScriptedFetcher(["<html>ok</html>"], BrowserOptions(max_retries=3))
    assert asyncio.run(fetcher.fetch("https://example.test")) == "<html>ok</html>"
    assert fetcher.recoveries == []
    assert recorded_sleeps == []


def test_fetch_retries_navigation_errors_with_backoff(recorded_sleeps: list[float]) -> None:
    fetcher = ScriptedFetcher(
        [PlaywrightTimeoutError("slow"), PlaywrightError("crash"), "<html>ok</html>"],
        BrowserOptions(max_retries=3, backoff_base_s=2.0),
    )
    assert asyncio.run(fetcher.fetch("https://example.test")) == "<html>ok</html>"
    assert fetcher.resets == 2
    assert len(recorded_sleeps) == 2
    assert 2.0 <= recorded_sleeps[0] <= 3.0
    assert 4.0 <= recorded_sleeps[1] <= 5.0


def test_fetch_raises_fetch_error_after_exhausting_retries(recorded_sleeps: list[float]) -> None:
    fetcher = ScriptedFetcher(
        [PlaywrightTimeoutError("slow")] * 3, BrowserOptions(max_retries=3, backoff_base_s=1.0)
    )
    with pytest.raises(FetchError, match="after 3 attempts") as excinfo:
        asyncio.run(fetcher.fetch("https://example.test"))
    assert not isinstance(excinfo.value, BlockedError)
    assert len(recorded_sleeps) == 2


def test_fetch_recovers_from_blocks_then_gives_up(recorded_sleeps: list[float]) -> None:
    fetcher = ScriptedFetcher([BlockedError("captcha")] * 3, BrowserOptions(max_retries=3))
    with pytest.raises(BlockedError, match="Blocked on"):
        asyncio.run(fetcher.fetch("https://example.test"))
    assert fetcher.recoveries == [1, 2]


def test_fetch_succeeds_after_block_recovery(recorded_sleeps: list[float]) -> None:
    fetcher = ScriptedFetcher([BlockedError("captcha"), "<html>ok</html>"], BrowserOptions())
    assert asyncio.run(fetcher.fetch("https://example.test")) == "<html>ok</html>"
    assert fetcher.recoveries == [1]


def test_block_without_manual_mode_raises_immediately() -> None:
    fetcher = PlaywrightFetcher(BrowserOptions(headless=True, manual_captcha=True))
    with pytest.raises(BlockedError, match="challenge"):
        asyncio.run(fetcher._wait_for_manual_solution(page=None, url="https://example.test"))


def test_launch_requires_context_manager() -> None:
    with pytest.raises(RuntimeError, match="async context manager"):
        asyncio.run(PlaywrightFetcher()._launch())
