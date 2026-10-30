import httpx
import openai
import pytest

from scout.retry import BadRequest, with_retry

_REQUEST = httpx.Request("POST", "https://api.meta.ai/v1/responses")


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _raiser(error):
    def fn():
        raise error
    return fn


def test_retries_transient_then_succeeds():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise StatusError(503)
        return "ok"

    assert with_retry(fn, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]


def test_rate_limit_waits_four_times_longer():
    sleeps = []

    def fn():
        raise StatusError(429)

    with pytest.raises(StatusError):
        with_retry(fn, retries=2, sleep=sleeps.append)
    assert sleeps == [4, 8]


@pytest.mark.parametrize("status", [400, 413, 422])
def test_rejected_request_is_not_retried(status):
    calls = []

    def fn():
        calls.append(1)
        raise StatusError(status)

    with pytest.raises(BadRequest):
        with_retry(fn, sleep=lambda s: None)
    assert len(calls) == 1


def test_non_transient_error_raises_immediately():
    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        with_retry(fn, sleep=lambda s: None)


def test_gives_up_after_retries():
    sleeps = []

    def fn():
        raise StatusError(503)

    with pytest.raises(StatusError):
        with_retry(fn, retries=2, sleep=sleeps.append)
    assert sleeps == [1, 2]


def test_sdk_connection_error_is_retried():
    sleeps = []
    with pytest.raises(openai.APIConnectionError):
        with_retry(_raiser(openai.APIConnectionError(request=_REQUEST)), retries=1, sleep=sleeps.append)
    assert sleeps == [1]


def test_sdk_timeout_is_not_retried():
    sleeps = []
    with pytest.raises(openai.APITimeoutError):
        with_retry(_raiser(openai.APITimeoutError(request=_REQUEST)), sleep=sleeps.append)
    assert sleeps == []


def test_sdk_bad_request_becomes_bad_request():
    error = openai.BadRequestError("rejected", response=httpx.Response(400, request=_REQUEST), body=None)
    with pytest.raises(BadRequest):
        with_retry(_raiser(error), sleep=lambda s: None)
