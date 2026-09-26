"""Stealth-hardened Playwright fetcher: pacing, retries, proxy rotation and block detection."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    ProxySettings,
    Route,
    ViewportSize,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from iphone_valuator.config import BROWSER_STATE_PATH

logger = logging.getLogger(__name__)

BLOCK_STATUS_CODES: Final = frozenset({403, 429})
BLOCK_TEXT_MARKERS: Final = (
    "доступ ограничен",
    "проблема с ip",
    "подозрительная активность",
    "вы не робот",
    "access denied",
    "too many requests",
)
BLOCK_SELECTOR: Final = ".firewall-container, .firewall-title, .js-firewall-form"
BLOCKED_RESOURCE_TYPES: Final = frozenset({"image", "media", "font"})
VIEWPORTS: Final[tuple[tuple[int, int], ...]] = (
    (1366, 768),
    (1440, 900),
    (1536, 864),
    (1920, 1080),
)
CAPTCHA_POLL_INTERVAL_S: Final = 5.0
ACCEPT_LANGUAGE: Final = "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"
STEALTH_INIT_SCRIPT: Final = """
Object.defineProperty(Navigator.prototype, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['ru-RU', 'ru', 'en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
window.chrome = window.chrome || { runtime: {} };
if (navigator.permissions && navigator.permissions.query) {
  const originalQuery = navigator.permissions.query.bind(navigator.permissions);
  navigator.permissions.query = (parameters) =>
    parameters && parameters.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : originalQuery(parameters);
}
"""


class FetchError(RuntimeError):
    """A page could not be fetched after exhausting all retries."""


class BlockedError(FetchError):
    """The site answered with an anti-bot, captcha or rate-limit page."""


class PageFetcher(Protocol):
    """Anything that can turn a URL into rendered HTML."""

    async def fetch(self, url: str, *, wait_for: str | None = None) -> str: ...


def is_block_page(html: str) -> bool:
    """Detect Avito's firewall/captcha page (e.g. "Доступ ограничен: проблема с IP")."""
    soup = BeautifulSoup(html, "lxml")
    if soup.select_one(BLOCK_SELECTOR) is not None:
        return True
    headline = " ".join(
        tag.get_text(" ", strip=True) for tag in soup.find_all(["title", "h1", "h2"])
    ).lower()
    return any(marker in headline for marker in BLOCK_TEXT_MARKERS)


def parse_proxy(url: str) -> ProxySettings:
    """Convert ``scheme://user:pass@host:port`` into Playwright proxy settings."""
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.hostname:
        raise ValueError(f"Invalid proxy URL: {url!r}")
    server = f"{parts.scheme}://{parts.hostname}"
    if parts.port:
        server += f":{parts.port}"
    proxy: ProxySettings = {"server": server}
    if parts.username:
        proxy["username"] = unquote(parts.username)
    if parts.password:
        proxy["password"] = unquote(parts.password)
    return proxy


def build_user_agent(browser_version: str) -> str:
    """Desktop Chrome UA matching the real engine version, without the ``Headless`` marker."""
    major = browser_version.split(".", 1)[0] or "140"
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
    )


@dataclass(frozen=True, slots=True)
class BrowserOptions:
    """Knobs for browser behaviour and anti-bot resilience."""

    headless: bool = True
    channel: str | None = None
    proxies: tuple[str, ...] = ()
    storage_state_path: Path | None = BROWSER_STATE_PATH
    navigation_timeout_s: float = 45.0
    selector_timeout_s: float = 15.0
    max_retries: int = 3
    backoff_base_s: float = 5.0
    block_cooldown_s: float = 90.0
    manual_captcha: bool = False
    captcha_timeout_s: float = 300.0
    block_resources: bool = True
    locale: str = "ru-RU"
    timezone_id: str = "Europe/Moscow"


class PlaywrightFetcher:
    """Async context manager that fetches pages through a single, human-paced Chromium tab.

    Anti-bot measures: automation flags disabled, navigator patches, realistic UA/locale/timezone,
    persisted cookies, random viewport, scrolling and mouse movement, heavy resources blocked,
    exponential backoff, cooldown or proxy rotation on block pages and optional manual captcha
    solving in headful mode.
    """

    def __init__(
        self, options: BrowserOptions | None = None, *, rng: random.Random | None = None
    ) -> None:
        self._options = options or BrowserOptions()
        self._rng = rng or random.Random()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._proxy_index = 0

    async def __aenter__(self) -> Self:
        self._playwright = await async_playwright().start()
        try:
            await self._launch()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._close_browser()
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def fetch(self, url: str, *, wait_for: str | None = None) -> str:
        retries = self._options.max_retries
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return await self._fetch_once(url, wait_for)
            except BlockedError as error:
                last_error = error
                logger.warning("Anti-bot page on %s (attempt %d/%d)", url, attempt, retries)
                if attempt < retries:
                    await self._recover_from_block(attempt)
            except PlaywrightError as error:
                last_error = error
                if attempt < retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning("Fetching %s failed: %s. Retrying in %.1fs", url, error, delay)
                    await asyncio.sleep(delay)
                    await self._reset_page()
        if isinstance(last_error, BlockedError):
            raise BlockedError(f"Blocked on {url} after {retries} attempts") from last_error
        raise FetchError(f"Failed to fetch {url} after {retries} attempts") from last_error

    async def _fetch_once(self, url: str, wait_for: str | None) -> str:
        page = await self._ensure_page()
        response = await page.goto(
            url, wait_until="domcontentloaded", timeout=self._options.navigation_timeout_s * 1000
        )
        status = response.status if response is not None else None
        if wait_for and status not in BLOCK_STATUS_CODES:
            try:
                await page.wait_for_selector(
                    wait_for, timeout=self._options.selector_timeout_s * 1000
                )
            except PlaywrightTimeoutError:
                logger.debug("Selector %r did not appear on %s", wait_for, url)
        if status in BLOCK_STATUS_CODES or is_block_page(await page.content()):
            await self._wait_for_manual_solution(page, url)
        await self._simulate_reading(page)
        return await page.content()

    async def _wait_for_manual_solution(self, page: Page, url: str) -> None:
        if not self._options.manual_captcha or self._options.headless:
            raise BlockedError(f"Anti-bot challenge on {url}")
        timeout = self._options.captcha_timeout_s
        logger.warning(
            "Anti-bot challenge on %s: solve it in the browser within %.0fs", url, timeout
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            await asyncio.sleep(CAPTCHA_POLL_INTERVAL_S)
            if not is_block_page(await page.content()):
                logger.info("Challenge solved, continuing")
                return
        raise BlockedError(f"Challenge on {url} was not solved within {timeout:.0f}s")

    async def _simulate_reading(self, page: Page) -> None:
        viewport = page.viewport_size or ViewportSize(width=1366, height=768)
        for _ in range(self._rng.randint(3, 6)):
            await page.mouse.move(
                self._rng.randint(50, viewport["width"] - 50),
                self._rng.randint(50, viewport["height"] - 50),
                steps=self._rng.randint(5, 15),
            )
            await page.mouse.wheel(0, self._rng.randint(300, 900))
            await asyncio.sleep(self._rng.uniform(0.3, 0.9))

    async def _recover_from_block(self, attempt: int) -> None:
        proxies = self._options.proxies
        if len(proxies) > 1:
            self._proxy_index = (self._proxy_index + 1) % len(proxies)
            logger.info("Rotating to proxy %d/%d", self._proxy_index + 1, len(proxies))
            await self._close_browser()
            await self._launch()
            return
        delay = self._options.block_cooldown_s * 2 ** (attempt - 1) * self._rng.uniform(0.8, 1.2)
        logger.warning("Cooling down for %.0fs before retrying", delay)
        await asyncio.sleep(delay)

    def _backoff_delay(self, attempt: int) -> float:
        return self._options.backoff_base_s * 2 ** (attempt - 1) + self._rng.uniform(0.0, 1.0)

    async def _launch(self) -> None:
        if self._playwright is None:
            raise RuntimeError("PlaywrightFetcher must be used as an async context manager")
        options = self._options
        proxy = parse_proxy(options.proxies[self._proxy_index]) if options.proxies else None
        self._browser = await self._playwright.chromium.launch(
            headless=options.headless,
            channel=options.channel,
            proxy=proxy,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        width, height = self._rng.choice(VIEWPORTS)
        state_path = options.storage_state_path
        self._context = await self._browser.new_context(
            user_agent=build_user_agent(self._browser.version),
            locale=options.locale,
            timezone_id=options.timezone_id,
            viewport=ViewportSize(width=width, height=height),
            storage_state=state_path if state_path is not None and state_path.exists() else None,
            extra_http_headers={"Accept-Language": ACCEPT_LANGUAGE},
        )
        await self._context.add_init_script(STEALTH_INIT_SCRIPT)
        if options.block_resources:
            await self._context.route("**/*", self._filter_route)
        self._page = await self._context.new_page()

    @staticmethod
    async def _filter_route(route: Route) -> None:
        if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
            await route.abort()
        else:
            await route.continue_()

    async def _ensure_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            if self._context is None:
                raise RuntimeError("Browser context is not running")
            self._page = await self._context.new_page()
        return self._page

    async def _reset_page(self) -> None:
        try:
            if self._page is not None and not self._page.is_closed():
                await self._page.close()
            self._page = None
            await self._ensure_page()
        except (PlaywrightError, RuntimeError) as error:
            logger.warning("Browser became unusable (%s); relaunching", error)
            await self._close_browser()
            await self._launch()

    async def _close_browser(self) -> None:
        state_path = self._options.storage_state_path
        if self._context is not None:
            try:
                if state_path is not None:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    await self._context.storage_state(path=state_path)
                await self._context.close()
            except PlaywrightError as error:
                logger.debug("Ignoring error while closing the browser context: %s", error)
        if self._browser is not None:
            try:
                await self._browser.close()
            except PlaywrightError as error:
                logger.debug("Ignoring error while closing the browser: %s", error)
        self._page = None
        self._context = None
        self._browser = None
