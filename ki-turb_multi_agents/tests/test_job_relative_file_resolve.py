"""Bare filenames in follow-ups resolve against the active simulation job."""
from __future__ import annotations

from pathlib import Path

from agents.langgraph.recovery import recovery_plan
from agents.langgraph.turn_memory import update_turn_memory
from agents.tools.core import execute_tool
from agents.tools._shared import resolve_existing_project_file


def test_resolve_bare_name_under_active_job(tmp_path: Path):
    job = "job_abc123"
    nested = tmp_path / "simulations" / job / "executable"
    nested.mkdir(parents=True)
    (nested / "whatever_output.txt").write_text("payload\n", encoding="utf-8")

    resolved = resolve_existing_project_file(
        "whatever_output.txt",
        tmp_path,
        {"simulation_job_id": job},
    )
    assert resolved is not None
    assert resolved.name == "whatever_output.txt"

    text = execute_tool(
        "read_file",
        {"filepath": "whatever_output.txt"},
        tmp_path,
        {"simulation_job_id": job},
    )
    assert "payload" in text


def test_find_file_defaults_to_active_job(tmp_path: Path):
    job = "job_abc123"
    nested = tmp_path / "simulations" / job / "raw"
    nested.mkdir(parents=True)
    (nested / "metrics.json").write_text("{}\n", encoding="utf-8")

    found = execute_tool(
        "find_file",
        {"pattern": "metrics.json", "directory": "."},
        tmp_path,
        {"simulation_job_id": job},
    )
    assert f"simulations/{job}/raw/metrics.json" in found


def test_turn_memory_records_job_dir():
    mem = update_turn_memory(
        None,
        user_request="build a case",
        plan={"steps": [{"role": "simulation", "tool": "build_simulation_case"}]},
        task_results=[{"role": "simulation", "text": "Prepared. job_id: job_x", "tool_outputs": []}],
        session_context={"simulation_job_id": "job_x"},
        final_text="Prepared. job_id: job_x",
        status="completed",
    )
    assert "simulations/job_x" in mem["last_paths"]


def test_file_not_found_recovery_does_not_retry_bare_path():
    plan = recovery_plan(
        user_request="open that log in the job folder",
        failure="steward.read_file: Error: File not found: mystery.log",
        task_results=[{"role": "steward", "text": "Error: File not found: mystery.log"}],
    )
    assert plan.steps[0].role == "steward"
    assert "Do NOT retry" in plan.steps[0].instruction
    assert "find_file" in plan.steps[0].instruction
