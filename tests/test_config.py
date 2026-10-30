from scout.config import Settings, get_settings


def test_get_settings_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k-123")
    monkeypatch.setenv("AGENT_MODEL", "muse-spark-1.2")
    settings = get_settings()
    assert settings.api_key == "k-123"
    assert settings.agent_model == "muse-spark-1.2"
    assert settings.sam_model == "sam-3.1"


def test_settings_defaults():
    settings = Settings(api_key="x")
    assert settings.base_url == "https://api.meta.ai/v1"
    assert settings.work_max_px == 1024 and settings.max_tool_calls == 12
