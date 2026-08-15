"""
Fetches a web page and extracts its readable text content, so a URL
can be ingested through the exact same chunk -> embed -> store pipeline
as an uploaded file.
"""

import requests
from bs4 import BeautifulSoup


class WebFetchError(Exception):
    pass


def fetch_page(url: str) -> tuple[str, str]:
    """Returns (title, clean_text) for the given URL."""
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; UniversalRAG/1.0)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebFetchError(f"Could not fetch {url}: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)

    if not clean_text.strip():
        raise WebFetchError(f"No readable text content found at {url}")

    return title, clean_text