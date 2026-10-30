"""Retry policy shared by the SAM and Muse Spark clients."""
import time
from typing import Callable, TypeVar

T = TypeVar("T")

RETRY_STATUS = {429, 500, 502, 503, 504}
TRANSIENT_ERRORS = {"APIConnectionError", "APITimeoutError"}


class BadRequest(Exception):
    """The API rejected the request (HTTP 400). Retrying the same request will not help."""


def with_retry(fn: Callable[[], T], retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> T:
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status == 400:
                raise BadRequest(str(error)) from error
            transient = status in RETRY_STATUS or type(error).__name__ in TRANSIENT_ERRORS
            if not transient or attempt == retries:
                raise
            sleep(2 ** attempt)
    raise AssertionError("unreachable")
