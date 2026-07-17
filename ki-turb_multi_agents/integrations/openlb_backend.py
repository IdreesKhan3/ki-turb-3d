"""Strict, version-aware OpenLB HIT backend with isolated build/run artifacts."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agents.physics_constraint_agent import PhysicsConstraintAgent
from agents.security.openlb_permissions import OpenLBPermission, require_openlb_permission
from agents.tools.simulation.hit_supervisor import HITSupervisor
from agents.tools.simulation.openlb_compile_agent import OpenLBCompileAgent
from integrations.base import BackendError, BackendNotConfigured, LocalCommandBackend, new_job_id
from integrations.local_process import LocalProcessError
from integrations.openlb.capability_validator import OpenLBHITCapabilityValidator
from integrations.openlb.config_translator import OpenLBHITConfigTranslator
from integrations.openlb.output_adapter import OpenLBOutputAdapter
from integrations.openlb.provenance import OpenLBProvenanceCollector
from integrations.openlb.unit_system import unit_system_from_openlb_hit
from schemas import CFDCase, DatasetManifest, SimulationJob
from schemas.openlb_hit import BuildProfile, ExecutionMode, OpenLBHITConfig
from schemas.simulation_job import JobPaths, JobStatus

APP_NAME = "kiTurbHIT3D"


class OpenLBBackend(LocalCommandBackend):
    name = "openlb"
    env_var = "KITURB_OPENLB_EXECUTABLE"
    openlb_root_env = "KITURB_OPENLB_ROOT"
    openlb_app_env = "KITURB_OPENLB_APP_DIR"

    def __init__(self, executable=None, *, runner=None, role: str = "simulation"):
        super().__init__(executable, runner=runner)
        self.role = role
        self.validator = OpenLBHITCapabilityValidator()
        self.translator = OpenLBHITConfigTranslator(self.validator)
        self.compiler = OpenLBCompileAgent()
        self.adapter = OpenLBOutputAdapter()
        self.provenance = OpenLBProvenanceCollector()
        self._supervisors: Dict[str, HITSupervisor] = {}

    def _require(self, permission: OpenLBPermission) -> None:
        require_openlb_permission(self.role, permission)

    @staticmethod
    def _is_openlb_root(path: Path) -> bool:
        return (path / "src").is_dir() and (path / "global.mk").is_file()

    def _workspace_roots(self) -> List[Path]:
        here = Path(__file__).resolve()
        return [
            here.parents[2],
            Path.cwd(),
            Path.cwd().parent,
        ]

    def _resolve_openlb_root(self, version=None) -> Path:
        candidates = []
        requested = version or os.getenv("KITURB_OPENLB_VERSION")
        if os.getenv(self.openlb_root_env):
            candidates.append(Path(os.environ[self.openlb_root_env]))
        bases = []
        for root in self._workspace_roots():
            bases.extend(
                [
                    root / "cfd_solvers/openLB",
                    root / "cfd_solvers/openlb",
                ]
            )
        if requested:
            candidates.extend(base / requested for base in bases)
        candidates.extend(bases)
        for candidate in candidates:
            if self._is_openlb_root(candidate):
                return candidate.resolve()
        raise BackendNotConfigured("OpenLB root not found; set KITURB_OPENLB_ROOT")

    def _app_dir_path(self, version=None) -> Path:
        if os.getenv(self.openlb_app_env):
            return Path(os.environ[self.openlb_app_env]).resolve()
        candidates = []
        for root in self._workspace_roots():
            candidates.append(root / "cfd_solvers/SolverApps" / APP_NAME)
        # Legacy layout (app lived inside OpenLB examples/) — keep as fallback.
        try:
            candidates.append(self._resolve_openlb_root(version) / "examples" / APP_NAME)
        except BackendNotConfigured:
            pass
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        raise BackendNotConfigured(
            f"OpenLB app '{APP_NAME}' not found under cfd_solvers/SolverApps; "
            f"set {self.openlb_app_env}"
        )

    def prepare_case(self, case: CFDCase, case_dir, job_id=None) -> SimulationJob:
        self._require(OpenLBPermission.WRITE_CASE)
        target = Path(case_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for sub in ("build", "executable", "output", "raw", "processed", "figures", "report", "checkpoints", "logs"):
            (target / sub).mkdir(exist_ok=True)
        config = OpenLBHITConfig.from_cfd_case(case)
        # Calibrate first — bare schema acceptance (div 1e-6) rejects practical smoke cases.
        config, physics = PhysicsConstraintAgent().calibrate(config)
        if not physics.accepted:
            raise BackendError("HIT physics validation failed: " + "; ".join(physics.errors))
        case.hit = config
        case.units = unit_system_from_openlb_hit(config)
        decision = self.validator.assert_supported(config)
        files = self.translator.write_case(config, target)
        (target / "case.json").write_text(case.to_json(), encoding="utf-8")
        (target / "unit_system.json").write_text(case.units.model_dump_json(indent=2), encoding="utf-8")
        provenance = self.provenance.collect(
            openlb_root=self._resolve_openlb_root(config.openlb_version),
            app_dir=self._app_dir_path(config.openlb_version),
            build_profile=config.execution.build_profile.value,
        )
        (target / "provenance.json").write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
        (target / "capability_decision.json").write_text(decision.model_dump_json(indent=2), encoding="utf-8")
        run_sh = target / "run.sh"
        run_sh.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nexe="${KITURB_OPENLB_EXECUTABLE:-./executable/kiTurbHIT3D}"\nmkdir -p output\nexec "$exe" --run case.xml output\n',
            encoding="utf-8",
        )
        run_sh.chmod(0o755)
        job = SimulationJob(
            job_id=job_id or new_job_id(),
            backend=self.name,
            case_name=case.name,
            status=JobStatus.CREATED,
            paths=JobPaths(
                case_dir=str(target), build_dir=str(target / "build"), output_dir=str(target / "output"),
                raw_dir=str(target / "raw"), processed_dir=str(target / "processed"), figures_dir=str(target / "figures"),
                report_dir=str(target / "report"), checkpoint_dir=str(target / "checkpoints"), log_path=str(target / "logs/run.log"),
            ),
            resources=config.execution.model_dump(mode="json"),
            requested_config=config.model_dump(mode="json"),
            metadata={
                "case_file": str(files["xml"]),
                "capability_decision": decision.model_dump(mode="json"),
                "unit_system": case.units.model_dump(mode="json") if case.units else {},
            },
        )
        self._supervisors[job.job_id] = HITSupervisor(config.acceptance)
        job.metadata["validation_status"] = "validated"
        job.mark(JobStatus.PREPARED, message="typed case prepared and validated")
        return job

    def _write_case_inputs(self, case, case_dir) -> List[Path]:
        return list(self.translator.write_case(OpenLBHITConfig.from_cfd_case(case), case_dir).values())

    @staticmethod
    def _extract_json(text: str) -> dict:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise BackendError("OpenLB did not return an effective JSON configuration")
        return json.loads(text[start:end + 1])

    @staticmethod
    def _compact(value: object) -> str:
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    def _verify_executable_contract(self, executable: str | Path, config: OpenLBHITConfig, case_dir: str | Path) -> dict:
        case_dir = Path(case_dir)
        command = [str(executable), "--dump-effective-config", str(case_dir / "case.xml")]
        completed = subprocess.run(command, cwd=case_dir, text=True, capture_output=True, timeout=120, check=False)
        if completed.returncode != 0:
            raise BackendError("OpenLB effective-config validation failed: " + (completed.stderr or completed.stdout).strip())
        actual = self._extract_json(completed.stdout)
        derived = config.derive_scaling()
        mismatches: List[str] = []
        expected_collision = config.collision.model.value
        checks = {
            "lattice": (actual.get("lattice"), config.domain.lattice.value),
            "collision": (actual.get("collision"), expected_collision),
            "forcing": (actual.get("forcing"), config.forcing.type.value),
            "initial_condition": (actual.get("initial_condition"), config.initial_condition.type.value),
        }
        for name, (observed, expected) in checks.items():
            if self._compact(observed) != self._compact(expected):
                mismatches.append(f"{name}: requested={expected!r}, effective={observed!r}")
        if list(actual.get("resolution") or []) != list(config.domain.resolution):
            mismatches.append(f"resolution: requested={list(config.domain.resolution)}, effective={actual.get('resolution')}")
        for name, observed, expected in (
            ("tau", actual.get("tau"), derived.relaxation_time),
            ("mach", actual.get("mach"), derived.actual_mach),
        ):
            if observed is None or not math.isclose(float(observed), float(expected), rel_tol=1e-10, abs_tol=1e-12):
                mismatches.append(f"{name}: requested={expected!r}, effective={observed!r}")
        if not actual.get("dynamics_class"):
            mismatches.append("dynamics_class was not reported")
        if mismatches:
            raise BackendError("requested/effective OpenLB mismatch: " + "; ".join(mismatches))
        payload = {
            "requested_equals_effective": True,
            "requested": config.model_dump(mode="json"),
            "derived_scaling": derived.model_dump(mode="json"),
            "effective_openlb": actual,
            "validation_command": command,
        }
        (case_dir / "effective_case.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (case_dir / "effective_openlb.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
        return payload

    def compile_case(self, job):
        self._require(OpenLBPermission.COMPILE)
        if job.status not in {JobStatus.VALIDATED, JobStatus.PREPARED, JobStatus.BUILT, JobStatus.COMPILED}:
            return job.mark(JobStatus.FAILED, message=f"cannot compile from {job.status.value}")
        config = OpenLBHITConfig.model_validate(job.requested_config)
        job.mark(JobStatus.BUILDING, message="building isolated OpenLB artifact")
        result = self.compiler.compile(
            self._app_dir_path(config.openlb_version),
            Path(job.paths.case_dir) / "executable",
            profile=config.execution.build_profile,
            jobs=config.execution.num_threads,
            clean=False,
            smoke_test_args=["--capabilities"],
        )
        job.metadata["compile_result"] = result.model_dump(mode="json")
        if not result.success:
            return job.mark(JobStatus.FAILED, message="OpenLB compile failed; see compile.log", return_code=result.return_code)
        # Keep the binary path even if contract checks fail, so Run can report the real error.
        job.metadata["executable"] = result.executable
        self.executable = result.executable
        try:
            job.effective_config = self._verify_executable_contract(result.executable, config, job.paths.case_dir)
        except Exception as exc:
            return job.mark(
                JobStatus.FAILED,
                message=f"{exc}. Binary exists at {result.executable}; fix case.xml / rebuild, then Compile again.",
            )
        job.effective_config.setdefault("build", {})["provenance"] = result.provenance.model_dump(mode="json") if result.provenance else {}
        job.metadata["effective_contract_verified"] = True
        return job.mark(JobStatus.BUILT, message="OpenLB executable built and requested/effective contract verified")

    def _build_argv(self, job, executable):
        config = OpenLBHITConfig.model_validate(job.requested_config)
        base = [str(executable), "--run", str(Path(job.paths.case_dir) / "case.xml"), str(Path(job.paths.output_dir))]
        if config.execution.build_profile in {BuildProfile.MPI, BuildProfile.MPI_OPENMP} or config.execution.mode == ExecutionMode.MPI:
            launcher = shutil.which("mpirun") or shutil.which("mpiexec")
            if not launcher:
                raise BackendNotConfigured("MPI run requested but mpirun/mpiexec was not found")
            return [launcher, "-np", str(config.execution.num_procs), *base]
        return base

    def run_case(self, job):
        self._require(OpenLBPermission.RUN)
        executable = job.metadata.get("executable") or self.executable
        if not executable:
            candidate = Path(job.paths.case_dir) / "executable" / APP_NAME
            if candidate.is_file():
                executable = str(candidate)
                job.metadata["executable"] = executable
        if not executable:
            raise BackendNotConfigured("OpenLB executable is unavailable; compile the case first")
        if job.status == JobStatus.FAILED and not job.metadata.get("effective_contract_verified"):
            raise BackendNotConfigured(
                job.message
                or "Compile finished but case validation failed; rebuild the case (Mach/tau) then Compile again"
            )
        config = OpenLBHITConfig.model_validate(job.requested_config)
        if not job.metadata.get("effective_contract_verified"):
            job.effective_config = self._verify_executable_contract(executable, config, job.paths.case_dir)
            job.metadata["effective_contract_verified"] = True
        argv = self._build_argv(job, executable)
        environment = dict(config.execution.extra_environment)
        environment["OMP_NUM_THREADS"] = str(config.execution.num_threads)
        try:
            handle = self.runner.spawn(argv, cwd=job.paths.case_dir, log_path=job.paths.log_path, env=environment)
        except LocalProcessError as exc:
            return job.mark(JobStatus.FAILED, message=str(exc))
        self._supervisors[job.job_id] = HITSupervisor(config.acceptance)
        job.external_id = str(handle.pid)
        job.metadata["argv"] = argv
        job.mark(JobStatus.QUEUED, message=f"launched pid {handle.pid}")
        return job.mark(JobStatus.RUNNING)

    def check_status(self, job):
        if job.status.is_terminal or not job.external_id:
            return job
        code = self.runner.poll(int(job.external_id))
        diagnostics = Path(job.paths.output_dir) / "diagnostics.jsonl"
        if diagnostics.exists():
            config = OpenLBHITConfig.model_validate(job.requested_config)
            supervisor = self._supervisors.setdefault(job.job_id, HITSupervisor(config.acceptance))
            assessment = supervisor.assess(diagnostics)
            job.measured.update(assessment.latest or {})
            job.progress = assessment.progress
            if not assessment.healthy:
                try:
                    (Path(job.paths.output_dir) / "checkpoint.request").touch()
                    import time as _time
                    _time.sleep(1.0)
                except Exception:
                    pass
                self.runner.terminate(int(job.external_id))
                return job.mark(JobStatus.REJECTED, message="simulation health rejection: " + "; ".join(assessment.errors))
            latest_step = (assessment.latest or {}).get("step")
            max_steps = config.runtime.max_steps
            if (
                code is None
                and latest_step is not None
                and max_steps
                and int(latest_step) >= int(max_steps)
                and (assessment.progress or 0) >= 1.0
            ):
                code = 0
        if code is None:
            return job.mark(JobStatus.RUNNING)
        return job.mark(
            JobStatus.COMPLETED if code == 0 else JobStatus.FAILED,
            message="process exited cleanly" if code == 0 else f"process exited with code {code}",
            return_code=code,
        )

    def fetch_outputs(self, job, dest_dir=None) -> DatasetManifest:
        self._require(OpenLBPermission.FETCH_DATA)
        # Tests and manual imports may create output files without launching a process.
        if job.status not in {JobStatus.COMPLETED, JobStatus.FETCHED, JobStatus.PREPARED, JobStatus.BUILT}:
            raise BackendError("outputs can be fetched only for a prepared/built/completed run")
        job.mark(JobStatus.FETCHING)
        target = Path(dest_dir) if dest_dir else Path(job.paths.raw_dir)
        unit_system = None
        try:
            from schemas.openlb_hit import OpenLBHITConfig
            from schemas.unit_system import UnitSystem

            raw_units = job.metadata.get("unit_system")
            if raw_units:
                unit_system = UnitSystem.model_validate(raw_units)
            else:
                unit_system = unit_system_from_openlb_hit(
                    OpenLBHITConfig.model_validate(job.requested_config)
                )
        except Exception:
            unit_system = None
        result = self.adapter.adapt(
            job.paths.output_dir,
            target,
            source_job_id=job.job_id,
            source_simulation=job.case_name,
            case=job.requested_config,
            provenance=job.effective_config,
            expected_kinds=["velocity_field"] if any(Path(job.paths.output_dir).glob("*velocity*")) else [],
            unit_system=unit_system,
        )
        result.manifest.run_id = job.job_id
        result.manifest.requested_config = dict(job.requested_config)
        result.manifest.effective_config = dict(job.effective_config)
        result.manifest.measured = dict(job.measured)
        if unit_system is not None:
            result.manifest.unit_system = unit_system
            result.manifest.units = {**result.manifest.units, **unit_system.field_labels()}
        Path(result.manifest_path).write_text(result.manifest.to_json(), encoding="utf-8")
        job.metadata["manifest_path"] = result.manifest_path
        job.mark(JobStatus.FETCHED)
        return result.manifest

    def cancel_run(self, job):
        self._require(OpenLBPermission.CANCEL)
        if job.external_id:
            self.runner.terminate(int(job.external_id))
        return job.mark(JobStatus.CANCELLED, message="cancelled by request")
