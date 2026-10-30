"""Test-wide fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch):
    """Tests never call the network, so a real key from .env must not be visible to them."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
