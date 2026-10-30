"""Retry policy shared by the SAM and Muse Spark clients.

The OpenAI clients are built with max_retries=0, so this is the only retry layer.
"""
import time
from typing import Callable, TypeVar

T = TypeVar("T")

RETRY_STATUS = {429, 500, 502, 503, 504}
REJECTED_STATUS = {400, 413, 422}
# Timeouts are not retried: a call that already used its whole timeout would not fit in the stop's budget twice.
TRANSIENT_ERRORS = {"APIConnectionError"}


class BadRequest(Exception):
    """The API rejected the request (HTTP 400, 413 or 422). Retrying the same request will not help."""


def with_retry(fn: Callable[[], T], retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> T:
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status in REJECTED_STATUS:
                raise BadRequest(str(error)) from error
            transient = status in RETRY_STATUS or type(error).__name__ in TRANSIENT_ERRORS
            if not transient or attempt == retries:
                raise
            sleep(2 ** attempt * (4 if status == 429 else 1))  # rate limits clear slowly when 4 stops run at once
    raise AssertionError("unreachable")
