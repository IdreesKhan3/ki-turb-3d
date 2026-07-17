"""Blocked shell commands must fail before confirmation (no Accept loops)."""
from __future__ import annotations

from pathlib import Path

from agents.tools import execute_tool

ROOT = Path(__file__).resolve().parents[1]


def test_blocked_rm_does_not_request_confirmation():
    result = execute_tool(
        "run_shell_command",
        {"cmd": "rm -rf examples/delete"},
        ROOT,
        session_context={},
    )
    assert isinstance(result, str)
    assert result.startswith("Error: Blocked command")
    assert "delete_file" in result


def test_blocked_python_does_not_request_confirmation():
    result = execute_tool(
        "run_shell_command",
        {"cmd": 'python -c "import os; os.rmdir(\'examples/delete\')"'},
        ROOT,
        session_context={},
    )
    assert isinstance(result, str)
    assert "Blocked command" in result
    assert not (isinstance(result, dict) and result.get("status") == "pending_confirmation")


def test_delete_file_recursive_removes_directory(tmp_path):
    # Use a temp project-like root by writing under ROOT/examples via tool policy.
    target = ROOT / "examples" / "_tmp_delete_me_dir"
    target.mkdir(parents=True, exist_ok=True)
    (target / "note.txt").write_text("x", encoding="utf-8")
    try:
        result = execute_tool(
            "delete_file",
            {"filepath": "examples/_tmp_delete_me_dir", "recursive": True},
            ROOT,
            session_context={"tool_confirmation_approved": True},
        )
        assert "Directory deleted" in str(result)
        assert not target.exists()
    finally:
        if target.exists():
            import shutil

            shutil.rmtree(target)
