from pathlib import Path
from agents.langgraph.settings import LangGraphSettings


def test_settings_follow_existing_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    settings = LangGraphSettings.from_environment(tmp_path, "ollama")
    assert settings.model == "ollama:test-model"
    assert Path(settings.checkpoint_path).is_absolute()


def test_settings_deepseek_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    settings = LangGraphSettings.from_environment(tmp_path, "deepseek")
    assert settings.model == "deepseek:deepseek-v4-pro"


def test_execution_approval_defaults_on(tmp_path, monkeypatch):
    monkeypatch.delenv("KITURB_REQUIRE_RUN_APPROVAL", raising=False)
    assert LangGraphSettings.from_environment(tmp_path).require_execution_approval is True


def test_no_legacy_engine_mode_exists():
    assert "workflow_engine" not in LangGraphSettings.model_fields
