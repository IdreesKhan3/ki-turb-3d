"""World-state guards and verification for schema-first workflows."""

from __future__ import annotations

from pathlib import Path

from agents.langgraph.workflow_guards import guard_tool
from agents.langgraph.workflow_verify import verify_step
from agents.langgraph.workflow_world import WorkflowWorldState, snapshot_world


def test_guard_blocks_start_without_built_job():
    world = WorkflowWorldState(
        job_id="job_x",
        has_job_record=True,
        job_status="prepared",
        has_executable=False,
        capabilities={"start": False, "compile": True},
    )
    result = guard_tool("start_simulation", world)
    assert not result.allowed
    assert "start" in result.reason.lower() or "built" in result.reason.lower()


def test_guard_allows_start_when_built():
    world = WorkflowWorldState(
        job_id="job_x",
        has_job_record=True,
        job_status="built",
        has_executable=True,
        capabilities={"start": True},
    )
    assert guard_tool("start_simulation", world).allowed


def test_guard_blocks_fetch_while_running():
    world = WorkflowWorldState(
        job_id="job_x",
        has_job_record=True,
        job_status="running",
        capabilities={"fetch": False},
    )
    result = guard_tool("fetch_simulation_outputs", world)
    assert not result.allowed
    assert "running" in result.reason.lower()


def test_guard_blocks_analyze_without_data():
    world = WorkflowWorldState(capabilities={"analyze": False})
    result = guard_tool("compute_spectra", world)
    assert not result.allowed


def test_guard_allows_load_with_manifest():
    world = WorkflowWorldState(
        job_id="job_x",
        has_job_record=True,
        has_manifest=True,
        capabilities={"load": True},
    )
    assert guard_tool("load_dataset_manifest", world).allowed


def test_verify_fetch_requires_manifest():
    before = WorkflowWorldState(job_id="job_x", has_job_record=True, job_status="completed")
    after = WorkflowWorldState(job_id="job_x", has_job_record=True, job_status="fetched", has_manifest=False)
    result = verify_step("fetch_simulation_outputs", "Fetched outputs", before=before, after=after)
    assert not result.ok


def test_verify_start_requires_running():
    before = WorkflowWorldState(job_id="job_x", job_status="built", has_executable=True)
    after = WorkflowWorldState(job_id="job_x", job_status="prepared")
    result = verify_step("start_simulation", "Started simulation", before=before, after=after)
    assert not result.ok


def test_snapshot_world_from_real_job_with_manifest():
    root = Path(__file__).resolve().parents[1]
    job = "job_5fa8049d84b4"
    if not (root / "simulations" / job / "manifest.json").is_file():
        return
    world = snapshot_world(root, job_id=job, session_context={})
    assert world.has_job_record
    assert world.has_manifest
    assert world.capability("load")
    assert world.job_id == job
