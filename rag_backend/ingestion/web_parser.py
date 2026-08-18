"""
Web page fetching and readable-text extraction for the RAG ingestion pipeline.

Two-tier approach:

1. Fast path:
   HTTP GET + BeautifulSoup.
   Best for normal server-rendered pages.

2. Browser fallback:
   Playwright.
   Used when the static request returns little/no useful content,
   which commonly happens with JavaScript-rendered websites.

Public interface intentionally remains:

    fetch_page(url) -> (title, clean_text)

This keeps compatibility with the existing ingestion pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup


class WebFetchError(Exception):
    """Raised when a web page cannot be fetched or parsed."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATIC_TIMEOUT = 15
BROWSER_TIMEOUT = 60_000

# If a page returns less than this amount of useful text,
# Playwright will be used as a fallback.
MIN_USEFUL_TEXT_LENGTH = 200

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _clean_html(html: str, url: str) -> tuple[str, str]:
    """
    Convert HTML into:

        (title, clean_text)

    This function is intentionally kept independent of whether
    the HTML came from requests or Playwright.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that generally do not contain useful RAG content.
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
            "svg",
            "form",
        ]
    ):
        tag.decompose()

    # Extract title safely.
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        title = url

    # Extract visible text.
    text = soup.get_text(separator="\n")

    # Normalize whitespace line by line.
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines.
    clean_text = "\n".join(
        line for line in lines if line
    )

    return title, clean_text


# ---------------------------------------------------------------------------
# Static HTTP fetch
# ---------------------------------------------------------------------------

def _fetch_static(url: str) -> tuple[str, str]:
    """
    Fast path.

    Uses requests + BeautifulSoup.

    This works well for normal server-rendered websites.
    It may return little content for JavaScript-heavy websites.
    """

    try:
        response = requests.get(
            url,
            timeout=STATIC_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise WebFetchError(
            f"Could not fetch {url} using HTTP: {exc}"
        ) from exc

    return _clean_html(response.text, url)


# ---------------------------------------------------------------------------
# Find an installed browser
# ---------------------------------------------------------------------------

def _find_installed_browser() -> str | None:
    """
    Look for an installed Chrome or Edge executable.

    This is used only as a fallback when Playwright's own Chromium
    executable is unavailable.

    Returns:
        Browser executable path, or None.
    """

    candidates: list[str] = []

    # Chrome
    candidates.extend(
        [
            os.path.expandvars(
                r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"
            ),
            os.path.expandvars(
                r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"
            ),
            os.path.expandvars(
                r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
            ),
        ]
    )

    # Edge
    candidates.extend(
        [
            os.path.expandvars(
                r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"
            ),
            os.path.expandvars(
                r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"
            ),
            os.path.expandvars(
                r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"
            ),
        ]
    )

    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate

    return None


# ---------------------------------------------------------------------------
# Playwright browser fetch
# ---------------------------------------------------------------------------

def _fetch_with_playwright(url: str) -> tuple[str, str]:
    """
    Browser fallback.

    Uses Playwright to execute JavaScript and extract the rendered HTML.

    Preference:
        1. Playwright bundled Chromium
        2. Installed Chrome/Edge if bundled Chromium is unavailable
    """

    try:
        from playwright.sync_api import (
            Error as PlaywrightError,
            sync_playwright,
        )

    except ImportError as exc:
        raise WebFetchError(
            "Playwright is not installed. Run:\n"
            "    pip install playwright\n"
            "Then install a browser:\n"
            "    python -m playwright install chromium"
        ) from exc

    browser = None
    context = None

    try:
        with sync_playwright() as p:

            # ---------------------------------------------------------------
            # Try Playwright's bundled Chromium first.
            # ---------------------------------------------------------------

            try:
                browser = p.chromium.launch(
                    headless=True
                )

            except PlaywrightError as bundled_error:

                # -----------------------------------------------------------
                # If bundled Chromium is unavailable, try installed Chrome
                # or Edge.
                # -----------------------------------------------------------

                executable_path = _find_installed_browser()

                if not executable_path:
                    raise WebFetchError(
                        "Playwright is installed, but no browser executable "
                        "is available.\n\n"
                        "Install Chromium with:\n"
                        "    python -m playwright install chromium\n\n"
                        f"Original Playwright error:\n"
                        f"{bundled_error}"
                    ) from bundled_error

                try:
                    browser = p.chromium.launch(
                        headless=True,
                        executable_path=executable_path,
                    )

                except Exception as installed_browser_error:
                    raise WebFetchError(
                        "Playwright could not launch the installed browser.\n"
                        f"Browser: {executable_path}\n"
                        f"Error: {installed_browser_error}"
                    ) from installed_browser_error

            # ---------------------------------------------------------------
            # Create an explicit browser context.
            # This is preferable to browser.new_page() for application code.
            # ---------------------------------------------------------------

            context = browser.new_context(
                user_agent=USER_AGENT,
                viewport={
                    "width": 1366,
                    "height": 768,
                },
                locale="en-US",
            )

            page = context.new_page()

            # Set navigation timeout.
            page.set_default_navigation_timeout(
                BROWSER_TIMEOUT
            )

            # ---------------------------------------------------------------
            # Navigate.
            #
            # We intentionally use DOMContentLoaded rather than networkidle.
            # Modern websites can continuously make background requests.
            # ---------------------------------------------------------------

            try:
                page.goto(
                    url,
                    timeout=BROWSER_TIMEOUT,
                    wait_until="domcontentloaded",
                )

            except PlaywrightError as navigation_error:

                # Some websites may continue loading after DOMContentLoaded.
                # If the DOM exists, we can still try extracting it.
                try:
                    current_url = page.url

                    if not current_url or current_url == "about:blank":
                        raise WebFetchError(
                            f"Browser navigation failed for {url}: "
                            f"{navigation_error}"
                        ) from navigation_error

                except Exception:
                    raise WebFetchError(
                        f"Browser navigation failed for {url}: "
                        f"{navigation_error}"
                    ) from navigation_error

            # ---------------------------------------------------------------
            # Give dynamically rendered content a short opportunity to appear.
            #
            # This is deliberately short; we don't use networkidle.
            # ---------------------------------------------------------------

            try:
                page.wait_for_load_state(
                    "domcontentloaded",
                    timeout=10_000,
                )
            except Exception:
                # DOMContentLoaded may already have happened or the page
                # may have unusual loading behavior. Continue to extraction.
                pass

            # ---------------------------------------------------------------
            # Extract rendered HTML.
            # ---------------------------------------------------------------

            html = page.content()

            if not html.strip():
                raise WebFetchError(
                    f"Browser returned empty HTML for {url}"
                )

            title, text = _clean_html(
                html,
                url,
            )

            if not text.strip():
                raise WebFetchError(
                    f"Browser rendered the page but no readable text "
                    f"was found at {url}"
                )

            return title, text

    except WebFetchError:
        raise

    except Exception as exc:
        raise WebFetchError(
            f"Could not render {url} with Playwright: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    finally:
        # ---------------------------------------------------------------
        # Always clean up browser resources.
        # ---------------------------------------------------------------

        try:
            if context is not None:
                context.close()
        except Exception:
            pass

        try:
            if browser is not None:
                browser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> tuple[str, str]:
    """
    Fetch a URL and return:

        (title, clean_text)

    Strategy:

        1. Try normal HTTP request.
        2. If it fails or returns insufficient content,
           use Playwright.
        3. If both approaches fail, raise WebFetchError.

    IMPORTANT:
        This function signature is kept unchanged so existing
        ingestion code does not need to change.
    """

    # Basic validation.
    if not isinstance(url, str) or not url.strip():
        raise WebFetchError(
            "URL must be a non-empty string."
        )

    url = url.strip()

    # ---------------------------------------------------------------
    # Fast path: requests + BeautifulSoup
    # ---------------------------------------------------------------

    try:
        title, text = _fetch_static(url)

        if len(text.strip()) >= MIN_USEFUL_TEXT_LENGTH:
            return title, text

    except WebFetchError:
        # Static fetching failed.
        # Playwright will be attempted below.
        pass

    # ---------------------------------------------------------------
    # Fallback: Playwright
    # ---------------------------------------------------------------

    title, text = _fetch_with_playwright(url)

    if not text.strip():
        raise WebFetchError(
            f"No readable text content found at {url}"
        )

    return title, text