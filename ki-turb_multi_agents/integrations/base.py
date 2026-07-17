"""CFD backend interface and a shared local-execution base class.

``CFDBackend`` is the contract every solver adapter implements. Agents and tools
depend only on this interface, so adding a new solver means adding an adapter
here without touching the agent runtime.

``LocalCommandBackend`` implements the full prepare/run/status/fetch/cancel flow
for solvers invoked as a local command line. Concrete adapters supply only the
solver-specific pieces: the input files to write and the argument vector to run.
"""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from schemas import CFDCase, DatasetFile, DatasetManifest, SimulationJob
from schemas.simulation_job import JobPaths, JobStatus

from .local_process import LocalProcessError, LocalProcessRunner


class BackendError(RuntimeError):
    """Base error for CFD backend failures."""


class BackendNotConfigured(BackendError):
    """Raised when a backend cannot run because its executable is not configured."""


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


def new_manifest_id() -> str:
    return f"ds_{uuid.uuid4().hex[:12]}"


class CFDBackend(ABC):
    """Common interface for CFD solver backends."""

    name: str = "base"

    @abstractmethod
    def prepare_case(
        self, case: CFDCase, case_dir: os.PathLike, job_id: Optional[str] = None
    ) -> SimulationJob:
        """Write solver input files for ``case`` into ``case_dir`` and return a job."""

    def compile_case(self, job: SimulationJob) -> SimulationJob:
        """Compile the prepared case if the backend requires it.

        Default is a no-op so backends that do not build (ANSYS, OpenFOAM stock
        solvers) work unchanged. Compiling backends (OpenLB, Palabos) override this.
        """
        return job

    @abstractmethod
    def run_case(self, job: SimulationJob) -> SimulationJob:
        """Start the solver for a prepared job and return the updated job."""

    @abstractmethod
    def check_status(self, job: SimulationJob) -> SimulationJob:
        """Refresh and return the job status from the backend."""

    @abstractmethod
    def fetch_outputs(
        self, job: SimulationJob, dest_dir: Optional[os.PathLike] = None
    ) -> DatasetManifest:
        """Collect the job's output files into a dataset manifest."""

    @abstractmethod
    def cancel_run(self, job: SimulationJob) -> SimulationJob:
        """Stop a running job and return the updated job."""


# Filename substrings mapped to (kind, format) — checked before the suffix table
# so a spectrum_1000.dat is labelled "energy_spectrum", not a generic "table".
_OUTPUT_NAME_PATTERNS = [
    ("norm", "normalized_spectrum", "dat"),
    ("spectrum", "energy_spectrum", "dat"),
    ("isotropy_coeff", "spectral_isotropy", "dat"),
    ("flatness_data", "flatness", "txt"),
    ("structure_functions", "structure_functions", "txt"),
    ("eps_real_validation", "dissipation_validation", "csv"),
    ("reynolds_stress", "reynolds_stress", "csv"),
    ("turbulence_stats", "turbulence_stats", "csv"),
]

# Output file extensions mapped to (kind, format) for manifest classification.
_OUTPUT_KINDS = {
    ".vti": ("velocity_field", "vti"),
    ".vtu": ("velocity_field", "vtu"),
    ".vtk": ("velocity_field", "vtk"),
    ".pvd": ("field_series", "pvd"),
    ".h5": ("velocity_field", "hdf5"),
    ".hdf5": ("velocity_field", "hdf5"),
    ".csv": ("table", "csv"),
    ".dat": ("table", "dat"),
    ".npy": ("array", "npy"),
    ".npz": ("array", "npz"),
    ".log": ("log", "log"),
    ".txt": ("log", "txt"),
    ".json": ("metadata", "json"),
}


class LocalCommandBackend(CFDBackend):
    """Base class for solvers launched as a local command.

    Subclasses set :attr:`name` and :attr:`env_var` and implement
    :meth:`_write_case_inputs` and :meth:`_build_argv`.
    """

    name: str = "local"
    env_var: str = ""

    def __init__(
        self,
        executable: Optional[str] = None,
        *,
        runner: Optional[LocalProcessRunner] = None,
    ) -> None:
        self.executable = executable or (os.environ.get(self.env_var) if self.env_var else None)
        self.runner = runner or LocalProcessRunner()

    # -- solver-specific hooks ---------------------------------------------
    @abstractmethod
    def _write_case_inputs(self, case: CFDCase, case_dir: Path) -> List[Path]:
        """Write solver input files and return the paths written."""

    @abstractmethod
    def _build_argv(self, job: SimulationJob, executable: str) -> List[str]:
        """Return the argument vector used to launch the solver."""

    # -- interface implementation ------------------------------------------
    def prepare_case(
        self, case: CFDCase, case_dir: os.PathLike, job_id: Optional[str] = None
    ) -> SimulationJob:
        case_path = Path(case_dir)
        output_path = case_path / "output"
        case_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        (case_path / "case.json").write_text(case.to_json(), encoding="utf-8")
        self._write_case_inputs(case, case_path)

        job = SimulationJob(
            job_id=job_id or new_job_id(),
            backend=self.name,
            case_name=case.name,
            status=JobStatus.PENDING,
            paths=JobPaths(
                case_dir=str(case_path),
                output_dir=str(output_path),
                log_path=str(case_path / "run.log"),
            ),
        )
        job.mark(JobStatus.PREPARED, message="case prepared")
        return job

    def run_case(self, job: SimulationJob) -> SimulationJob:
        executable = self.executable or job.metadata.get("executable")
        if not executable:
            raise BackendNotConfigured(
                f"{self.name} backend has no executable configured. "
                f"Compile the case, set the {self.env_var} environment variable, "
                f"or pass executable=."
            )
        case_dir = job.paths.case_dir
        if not case_dir or not Path(case_dir).is_dir():
            raise BackendError(f"case directory missing for job {job.job_id}")

        argv = self._build_argv(job, executable)
        log_path = job.paths.log_path or str(Path(case_dir) / "run.log")
        try:
            handle = self.runner.spawn(argv, cwd=case_dir, log_path=log_path)
        except LocalProcessError as exc:
            job.mark(JobStatus.FAILED, message=str(exc))
            return job

        job.external_id = str(handle.pid)
        job.paths.log_path = handle.log_path
        job.mark(JobStatus.SUBMITTED, message=f"launched pid {handle.pid}")
        job.mark(JobStatus.RUNNING)
        return job

    def check_status(self, job: SimulationJob) -> SimulationJob:
        if job.status.is_terminal or not job.external_id:
            return job
        code = self.runner.poll(int(job.external_id))
        if code is None:
            job.mark(JobStatus.RUNNING)
        elif code == 0:
            job.mark(JobStatus.COMPLETED, message="process exited cleanly", return_code=0)
        else:
            job.mark(JobStatus.FAILED, message=f"process exited with code {code}", return_code=code)
        return job

    def fetch_outputs(
        self, job: SimulationJob, dest_dir: Optional[os.PathLike] = None
    ) -> DatasetManifest:
        output_dir = Path(dest_dir) if dest_dir else Path(job.paths.output_dir or "")
        if not output_dir.is_dir():
            raise BackendError(f"output directory not found for job {job.job_id}: {output_dir}")

        manifest = DatasetManifest(
            manifest_id=new_manifest_id(),
            base_dir=str(output_dir),
            source_job_id=job.job_id,
            source_simulation=job.case_name,
            backend=self.name,
        )
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            kind, fmt = self._classify_output(path)
            manifest.add_file(
                DatasetFile(
                    path=str(path.relative_to(output_dir)),
                    kind=kind,
                    format=fmt,
                    size_bytes=path.stat().st_size,
                )
            )
        return manifest

    def cancel_run(self, job: SimulationJob) -> SimulationJob:
        if job.external_id:
            self.runner.terminate(int(job.external_id))
        job.mark(JobStatus.CANCELLED, message="cancelled by request")
        return job

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _classify_output(path: Path) -> Tuple[str, Optional[str]]:
        name = path.name.lower()
        for token, kind, fmt in _OUTPUT_NAME_PATTERNS:
            if token in name:
                return kind, path.suffix.lstrip(".").lower() or fmt
        return _OUTPUT_KINDS.get(path.suffix.lower(), ("data", path.suffix.lstrip(".") or None))
