from pathlib import Path


def test_obsolete_custom_runtime_files_are_removed():
    root = Path(__file__).resolve().parents[1]
    obsolete = [
        "agents/base_agent.py",
        "agents/orchestrator.py",
        "agents/runtime/workflow_runner.py",
        "agents/runtime/checkpoint.py",
        "agents/runtime/tracing.py",
        "agents/team_manager/prompts.py",
    ]
    assert not [item for item in obsolete if (root / item).exists()]


def test_unified_team_uses_only_langgraph():
    source = (Path(__file__).resolve().parents[1] / "agents/team_manager/__init__.py").read_text()
    assert "KITurbGraphEngine" in source
    assert "WorkflowRunner" not in source
    assert "LLMAgent" not in source
    assert "KITURB_WORKFLOW_ENGINE" not in source
