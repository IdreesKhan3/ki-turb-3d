"""Tests for the agent-facing simulation tools and their safety wiring."""

import re
from pathlib import Path

from agents import tools
from agents.tools import simulation
from agents.tools.simulation import _store


def _job_id(result: str) -> str:
    match = re.search(r"job_id:\s*(\S+)", result)
    assert match, f"no job_id in result: {result}"
    return match.group(1)


def _build(project_root: Path, backend: str = "openlb") -> str:
    return simulation.execute_tool(
        "build_simulation_case",
        {"backend": backend, "name": "hit", "resolution": [64, 64, 64], "max_steps": 100},
        project_root,
    )


def test_build_case_creates_job_and_files(tmp_path):
    result = _build(tmp_path)
    job_id = _job_id(result)

    job_dir = _store.job_dir(tmp_path, job_id)
    assert (job_dir / "job.json").is_file()
    assert (job_dir / "case.json").is_file()
    assert (job_dir / "case.xml").is_file()

    job = _store.load_job(tmp_path, job_id)
    assert job is not None
    assert job.backend == "openlb"
    assert job.case_name == "hit"


def test_build_case_unknown_backend(tmp_path):
    result = simulation.execute_tool(
        "build_simulation_case", {"backend": "nope", "name": "x"}, tmp_path
    )
    assert result.startswith("Error:")


def test_start_without_executable_returns_error(tmp_path, monkeypatch):
    monkeypatch.delenv("KITURB_OPENLB_EXECUTABLE", raising=False)
    job_id = _job_id(_build(tmp_path))
    result = simulation.execute_tool("start_simulation", {"job_id": job_id}, tmp_path)
    assert "Error" in result
    assert "executable" in result.lower()


def test_status_and_cancel_unknown_job(tmp_path):
    assert "no job found" in simulation.execute_tool(
        "check_simulation_status", {"job_id": "missing"}, tmp_path
    )


def test_fetch_and_read_manifest(tmp_path):
    job_id = _job_id(_build(tmp_path))
    job = _store.load_job(tmp_path, job_id)
    output_dir = Path(job.paths.output_dir)
    (output_dir / "spectrum_1000.csv").write_text("k,E\n1,1.0\n", encoding="utf-8")
    (output_dir / "u_1000.vti").write_text("<VTKFile/>", encoding="utf-8")

    fetch_result = simulation.execute_tool(
        "fetch_simulation_outputs", {"job_id": job_id}, tmp_path
    )
    assert "files: 2" in fetch_result
    assert (_store.job_dir(tmp_path, job_id) / "manifest.json").is_file()

    read_result = simulation.execute_tool(
        "read_dataset_manifest", {"job_id": job_id}, tmp_path
    )
    assert "files: 2" in read_result
    assert "energy_spectrum" in read_result and "velocity_field" in read_result


def test_confirmation_gating_through_top_level(tmp_path):
    args = {"backend": "openlb", "name": "hit"}
    pending = tools.execute_tool("build_simulation_case", args, tmp_path, session_context={})
    assert isinstance(pending, dict)
    assert pending.get("status") == "pending_confirmation"

    approved = tools.execute_tool(
        "build_simulation_case", args, tmp_path,
        session_context={"tool_confirmation_approved": True},
    )
    assert isinstance(approved, str)
    assert "job_id" in approved


def test_scope_enforcement_blocks_unassigned_agent(tmp_path):
    result = tools.execute_tool(
        "build_simulation_case", {"backend": "openlb", "name": "hit"}, tmp_path,
        session_context={"tool_confirmation_approved": True},
        allowed_tool_names={"plot_spectrum"},
    )
    assert isinstance(result, str)
    assert "not available for this agent" in result
