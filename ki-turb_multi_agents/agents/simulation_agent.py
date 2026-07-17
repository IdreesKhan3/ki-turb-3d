"""OpenLB HIT simulation lifecycle service used by the multi-agent workflow."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.physics_constraint_agent import PhysicsConstraintAgent, PhysicsConstraintDecision
from agents.security.openlb_permissions import OpenLBPermission, require_openlb_permission
from agents.tools.data.hit_data_collector import CollectionResult, HITDataCollector
from agents.tools.simulation.execution_runners import (
    ExecutionHandle,
    ExecutionRunner,
    ExecutionStatus,
    runner_from_config,
)
from agents.tools.simulation.hit_supervisor import HITSupervisor, SupervisorResult
from agents.tools.simulation.openlb_compile_agent import CompileResult, OpenLBCompileAgent
from integrations.openlb.config_translator import OpenLBHITConfigTranslator
from integrations.openlb.provenance import OpenLBProvenanceCollector
from schemas.openlb_hit import OpenLBHITConfig


class SimulationSession(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    run_id: str
    status: str
    run_dir: str
    case_dir: str
    build_dir: str
    output_dir: str
    diagnostics_path: str
    config: OpenLBHITConfig
    physics: Optional[PhysicsConstraintDecision] = None
    compile_result: Optional[CompileResult] = None
    execution_handle: Optional[ExecutionHandle] = None
    collection_manifest: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def save(self) -> Path:
        destination = Path(self.run_dir) / "simulation_session.json"
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SimulationSession":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class SimulationAgent:
    """Prepare, compile, execute, supervise and collect one OpenLB HIT run."""

    def __init__(
        self,
        *,
        physics_agent: Optional[PhysicsConstraintAgent] = None,
        translator: Optional[OpenLBHITConfigTranslator] = None,
        compile_agent: Optional[OpenLBCompileAgent] = None,
        collector: Optional[HITDataCollector] = None,
    ) -> None:
        self.role = "simulation"
        self.physics_agent = physics_agent or PhysicsConstraintAgent()
        self.translator = translator or OpenLBHITConfigTranslator(
            self.physics_agent.capability_validator
        )
        self.compile_agent = compile_agent or OpenLBCompileAgent()
        self.collector = collector or HITDataCollector()
        self._runners: Dict[str, ExecutionRunner] = {}

    def prepare(self, config: OpenLBHITConfig, run_root: str | Path) -> SimulationSession:
        require_openlb_permission(self.role, OpenLBPermission.WRITE_CASE)
        physics = self.physics_agent.validate(config)
        if not physics.accepted:
            messages = "; ".join(check.message for check in physics.report.errors())
            raise ValueError(f"HIT physics validation failed: {messages}")
        run_id = f"hit_{uuid.uuid4().hex[:12]}"
        run_dir = Path(run_root).expanduser().resolve() / run_id
        case_dir = run_dir / "case"
        build_dir = run_dir / "build"
        output_dir = run_dir / "raw"
        for directory in (case_dir, build_dir, output_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.translator.write_case(config, case_dir)
        (case_dir / "physics_validation.json").write_text(
            physics.report.model_dump_json(indent=2), encoding="utf-8"
        )
        session = SimulationSession(
            run_id=run_id,
            status="validated",
            run_dir=str(run_dir),
            case_dir=str(case_dir),
            build_dir=str(build_dir),
            output_dir=str(output_dir),
            diagnostics_path=str(output_dir / "diagnostics.jsonl"),
            config=config,
            physics=physics,
        )
        session.save()
        return session

    def compile(self, session: SimulationSession, app_dir: str | Path) -> SimulationSession:
        require_openlb_permission(self.role, OpenLBPermission.COMPILE)
        result = self.compile_agent.compile(
            app_dir,
            session.build_dir,
            profile=session.config.execution.build_profile,
        )
        session.compile_result = result
        session.status = "built" if result.success else "failed"
        session.save()
        if not result.success:
            raise RuntimeError("OpenLB compilation failed: " + "; ".join(result.diagnostics))
        from integrations.openlb_backend import OpenLBBackend
        contract = OpenLBBackend(role=self.role)._verify_executable_contract(
            result.executable, session.config, session.case_dir
        )
        session.metadata["effective_config"] = contract
        session.metadata["effective_contract_verified"] = True
        session.save()
        return session

    def start(self, session: SimulationSession, runner: Optional[ExecutionRunner] = None) -> SimulationSession:
        require_openlb_permission(self.role, OpenLBPermission.RUN)
        if not session.compile_result or not session.compile_result.success or not session.compile_result.executable:
            raise RuntimeError("session has no successful OpenLB build")
        selected_runner = runner or runner_from_config(session.config.execution)
        command = [
            session.compile_result.executable,
            "--run",
            str(Path(session.case_dir) / "case.xml"),
            session.output_dir,
        ]
        handle = selected_runner.submit(
            command,
            cwd=session.run_dir,
            log_path=Path(session.run_dir) / "run.log",
            environment=session.config.execution.extra_environment,
        )
        self._runners[session.run_id] = selected_runner
        session.execution_handle = handle
        session.status = "running"
        session.save()
        return session

    def status(self, session: SimulationSession) -> ExecutionStatus:
        if not session.execution_handle:
            return ExecutionStatus(state=session.status, message="simulation has not been launched")
        runner = self._runners.get(session.run_id)
        if runner is None:
            return ExecutionStatus(
                state="unknown",
                message="runner handle is not attached in this process; reconstruct the runner to poll it",
            )
        result = runner.status(session.execution_handle)
        session.status = result.state
        session.save()
        return result

    def supervise(
        self,
        session: SimulationSession,
        *,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> SupervisorResult:
        if not session.execution_handle:
            raise RuntimeError("simulation has not been launched")
        runner = self._runners.get(session.run_id)
        if runner is None:
            raise RuntimeError("runner is not attached to this session")
        supervisor = HITSupervisor(session.config.acceptance)
        result = supervisor.supervise_until_terminal(
            runner,
            session.execution_handle,
            session.diagnostics_path,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        session.status = result.state
        session.save()
        return result

    def cancel(self, session: SimulationSession) -> ExecutionStatus:
        require_openlb_permission(self.role, OpenLBPermission.CANCEL)
        if not session.execution_handle:
            return ExecutionStatus(state="cancelled", message="no active execution handle")
        runner = self._runners.get(session.run_id)
        if runner is None:
            raise RuntimeError("runner is not attached to this session")
        result = runner.cancel(session.execution_handle)
        session.status = result.state
        session.save()
        return result

    def collect(self, session: SimulationSession, *, allow_nonterminal: bool = False) -> CollectionResult:
        require_openlb_permission(self.role, OpenLBPermission.FETCH_DATA)
        if not allow_nonterminal and session.status not in {"completed", "failed", "rejected"}:
            raise RuntimeError(f"cannot collect a nonterminal run in state '{session.status}'")
        provenance = dict(session.metadata.get("effective_config") or {})
        if session.compile_result and session.compile_result.provenance:
            provenance.setdefault("build", {})["provenance"] = session.compile_result.provenance.model_dump(mode="json")
        result = self.collector.collect(
            session.output_dir,
            Path(session.run_dir) / "dataset",
            source_job_id=session.run_id,
            source_simulation=session.config.name,
            case=session.config.model_dump(mode="json"),
            provenance=provenance,
            expected_kinds=["velocity_field"] if session.config.outputs.write_velocity else [],
        )
        session.collection_manifest = result.manifest_path
        session.status = "fetched"
        session.save()
        return result


__all__ = ["SimulationSession", "SimulationAgent"]
