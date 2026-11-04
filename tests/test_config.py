from scout.config import Settings, get_settings


def test_get_settings_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k-123")
    monkeypatch.setenv("AGENT_MODEL", "muse-spark-1.2")
    monkeypatch.delenv("SAM_MODEL", raising=False)
    settings = get_settings()
    assert settings.api_key == "k-123"
    assert settings.agent_model == "muse-spark-1.2"
    assert settings.sam_model == "sam-3.1"


def test_get_settings_reads_walk_limits_from_env(monkeypatch):
    monkeypatch.setenv("WALKS_PER_IP_PER_HOUR", "7")
    monkeypatch.setenv("WALKS_PER_HOUR", "50")
    settings = get_settings()
    assert settings.walks_per_ip_per_hour == 7 and settings.walks_per_hour == 50


def test_get_settings_walk_limits_default_when_unset(monkeypatch):
    monkeypatch.delenv("WALKS_PER_IP_PER_HOUR", raising=False)
    monkeypatch.delenv("WALKS_PER_HOUR", raising=False)
    settings = get_settings()
    assert settings.walks_per_ip_per_hour == 10 and settings.walks_per_hour == 30


def test_settings_defaults():
    settings = Settings(api_key="x")
    assert settings.base_url == "https://api.meta.ai/v1"
    assert settings.work_max_px == 1024 and settings.max_tool_calls == 30
    assert settings.stop_timeout_s == 360.0 and settings.max_concurrent_stops == 10


def test_public_site_defaults():
    settings = Settings(api_key="x")
    assert settings.walks_per_hour == 30 and settings.offset_gps is True
