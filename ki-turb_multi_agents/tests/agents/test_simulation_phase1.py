"""Agent-facing tests for physics validation and post-processing tools."""

import re
from pathlib import Path

import h5py
import numpy as np

from agents.tools import simulation
from agents.tools.simulation import _store
from schemas.simulation_job import JobStatus


def _job_id(result: str) -> str:
    match = re.search(r"job_id:\s*(\S+)", result)
    assert match, result
    return match.group(1)


def test_build_valid_case_writes_validation_report(tmp_path):
    result = simulation.execute_tool(
        "build_simulation_case",
        {"backend": "openlb", "flow": "hit", "name": "hit", "resolution": [32, 32, 32]},
        tmp_path,
    )
    job_id = _job_id(result)
    job = _store.load_job(tmp_path, job_id)
    assert job.status is JobStatus.PREPARED
    assert (Path(job.paths.case_dir) / "validation_report.json").is_file()


def test_build_invalid_hit_case_rejected(tmp_path):
    result = simulation.execute_tool(
        "build_simulation_case",
        {"backend": "openlb", "name": "bad",
         "case": {"name": "bad", "geometry": {"kind": "box", "size": [1.0, 2.0, 1.0]},
                  "flow": {"kind": "hit"}}},
        tmp_path,
    )
    assert result.startswith("Error: physics validation failed")
    assert "hit_cube_domain" in result


def test_postprocess_tool_generates_analysis_ready_data(tmp_path):
    build = simulation.execute_tool(
        "build_simulation_case",
        {"backend": "openlb", "flow": "hit", "name": "hit", "resolution": [16, 16, 16],
         "max_steps": 2000, "output_interval": 1000},
        tmp_path,
    )
    job_id = _job_id(build)
    job = _store.load_job(tmp_path, job_id)

    output_dir = Path(job.paths.output_dir)
    rng = np.random.default_rng(0)
    for step in (1000, 2000):
        with h5py.File(output_dir / f"velocity_{step}.h5", "w") as f:
            f.create_dataset("velocity", data=rng.standard_normal((16, 16, 16, 3)))

    simulation.execute_tool("fetch_simulation_outputs", {"job_id": job_id}, tmp_path)
    result = simulation.execute_tool(
        "postprocess_simulation_outputs", {"job_id": job_id}, tmp_path
    )
    assert "analysis_ready" in result

    job = _store.load_job(tmp_path, job_id)
    assert job.status is JobStatus.ANALYSIS_READY
    assert (output_dir / "processed" / "spectra" / "spectrum_data1_1000.dat").is_file()
