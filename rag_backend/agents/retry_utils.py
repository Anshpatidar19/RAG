"""
Short, bounded retry with exponential backoff for Gemini calls.
Only retries transient server-side errors (5xx, e.g. "503 UNAVAILABLE —
high demand") — never masks genuine programming errors, and never
retries forever.
"""

import time

from google.genai import errors as genai_errors


def generate_with_retry(client, max_attempts: int = 3, base_delay: float = 1.0, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ServerError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_exc


def generate_stream_with_retry(client, max_attempts: int = 3, base_delay: float = 1.0, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content_stream(**kwargs)
        except genai_errors.ServerError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_exc