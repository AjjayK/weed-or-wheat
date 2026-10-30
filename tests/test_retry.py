import pytest

from scout.retry import BadRequest, with_retry


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_retries_transient_then_succeeds():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise StatusError(429)
        return "ok"

    assert with_retry(fn, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]


def test_bad_request_is_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise StatusError(400)

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
