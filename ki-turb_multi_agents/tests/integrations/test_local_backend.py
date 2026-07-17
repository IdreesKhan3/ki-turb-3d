"""End-to-end tests for the local-execution backend flow.

A minimal backend runs a short Python script as its "solver" so the full
prepare/run/status/fetch/cancel cycle is exercised without a real CFD solver.
"""

import sys
import time
from pathlib import Path
from typing import List

import pytest

from integrations.base import BackendNotConfigured, LocalCommandBackend
from integrations.local_process import LocalProcessRunner
from schemas import CFDCase, JobStatus, SimulationJob

_SOLVER_SCRIPT = (
    "import os\n"
    "os.makedirs('output', exist_ok=True)\n"
    "open(os.path.join('output', 'spectrum_1000.csv'), 'w').write('k,E\\n1,1.0\\n')\n"
    "open(os.path.join('output', 'u_1000.vti'), 'w').write('<VTKFile/>')\n"
)


class _ScriptBackend(LocalCommandBackend):
    name = "scripttest"
    env_var = "SCRIPTTEST_EXECUTABLE"

    def _write_case_inputs(self, case: CFDCase, case_dir: Path) -> List[Path]:
        script = case_dir / "solver.py"
        script.write_text(_SOLVER_SCRIPT, encoding="utf-8")
        return [script]

    def _build_argv(self, job: SimulationJob, executable: str) -> List[str]:
        return [executable, str(Path(job.paths.case_dir) / "solver.py")]


def _wait_terminal(backend, job, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = backend.check_status(job)
        if job.status.is_terminal:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job did not finish; status={job.status}")


def test_full_cycle_prepare_run_status_fetch(tmp_path):
    backend = _ScriptBackend(executable=sys.executable, runner=LocalProcessRunner())
    case = CFDCase(name="hit")

    job = backend.prepare_case(case, tmp_path / "job1", job_id="job_1")
    assert job.status is JobStatus.PREPARED
    assert (tmp_path / "job1" / "case.json").is_file()
    assert (tmp_path / "job1" / "solver.py").is_file()

    job = backend.run_case(job)
    assert job.external_id is not None

    job = _wait_terminal(backend, job)
    assert job.status is JobStatus.COMPLETED
    assert job.return_code == 0

    manifest = backend.fetch_outputs(job)
    kinds = {f.kind for f in manifest.files}
    assert kinds == {"energy_spectrum", "velocity_field"}
    assert manifest.source_job_id == "job_1"
    assert manifest.backend == "scripttest"


def test_run_without_executable_raises():
    backend = _ScriptBackend(executable=None, runner=LocalProcessRunner())
    job = SimulationJob(job_id="job_x", backend="scripttest")
    job.paths.case_dir = "."
    with pytest.raises(BackendNotConfigured):
        backend.run_case(job)


def test_fetch_missing_output_dir_errors(tmp_path):
    from integrations.base import BackendError

    backend = _ScriptBackend(executable=sys.executable)
    job = SimulationJob(job_id="job_y", backend="scripttest")
    job.paths.output_dir = str(tmp_path / "does_not_exist")
    with pytest.raises(BackendError):
        backend.fetch_outputs(job)


def test_cancel_marks_cancelled(tmp_path):
    backend = _ScriptBackend(executable=sys.executable, runner=LocalProcessRunner())
    job = backend.prepare_case(CFDCase(name="hit"), tmp_path / "jobc", job_id="job_c")
    job = backend.run_case(job)
    job = backend.cancel_run(job)
    assert job.status is JobStatus.CANCELLED


def test_get_backend_registry():
    import integrations

    assert set(integrations.available_backends()) == {"openlb", "palabos", "ansys"}
    assert integrations.get_backend("openlb").name == "openlb"
    with pytest.raises(integrations.BackendError):
        integrations.get_backend("nonexistent")
