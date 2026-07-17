"""Deterministic OpenLB HIT services used by the LangGraph subgraph."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agents.hit_master_agent import HITMasterAgent
from agents.physics_constraint_agent import PhysicsConstraintAgent
from agents.simulation_agent import SimulationAgent, SimulationSession
from postprocessing.hit_products_adapter import load_products_from_manifest
from schemas import DatasetManifest
from schemas.openlb_hit import OpenLBHITConfig

from .role_agents import RoleAgentFactory
from .settings import LangGraphSettings
from .state import KITurbState


def _event(stage: str, status: str, message: str = "", **data) -> list[dict]:
    return [{"stage": stage, "status": status, "message": message, **data}]


def _load_session(state: KITurbState) -> SimulationSession:
    path = state.get("session_path")
    if not path:
        raise RuntimeError("workflow session path is missing")
    return SimulationSession.load(path)


def _load_manifest(path: str) -> DatasetManifest:
    return DatasetManifest.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class HITGraphServices:
    settings: LangGraphSettings
    project_root: Path
    physics: PhysicsConstraintAgent
    simulation: SimulationAgent
    master: HITMasterAgent
    role_factory: Optional[RoleAgentFactory] = None

    @classmethod
    def default(cls, settings: LangGraphSettings, project_root: str | Path, role_factory: Optional[RoleAgentFactory] = None):
        root = Path(project_root).resolve()
        physics = PhysicsConstraintAgent()
        simulation = SimulationAgent(physics_agent=physics)
        return cls(settings, root, physics, simulation, HITMasterAgent(simulation_agent=simulation), role_factory)

    def normalize_request(self, state: KITurbState) -> Dict[str, Any]:
        try:
            payload = state.get("requested_config") or {}
            if payload:
                config = OpenLBHITConfig.model_validate(payload)
            else:
                request = (state.get("user_request") or "").strip()
                if request.startswith("{"):
                    config = OpenLBHITConfig.model_validate(json.loads(request))
                elif self.role_factory is not None and self.settings.use_llm_hit_parser:
                    from langchain_core.output_parsers import PydanticOutputParser
                    from langchain_core.prompts import ChatPromptTemplate
                    parser = PydanticOutputParser(pydantic_object=OpenLBHITConfig)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Convert the OpenLB HIT request to the exact schema. Never invent unsupported capabilities. Return JSON only.\n{format_instructions}"),
                        ("human", "{request}"),
                    ]).partial(format_instructions=parser.get_format_instructions())
                    config = (prompt | self.role_factory.model | parser).invoke({"request": request})
                else:
                    raise ValueError("No typed HIT configuration was supplied")
            return {"requested_config": config.model_dump(mode="json"), "status": "parsed", "events": _event("parse", "ok", "typed HIT request created")}
        except Exception as exc:
            return {"status": "failed", "errors": [f"request parsing failed: {exc}"], "events": _event("parse", "failed", str(exc))}

    def validate_physics(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        try:
            config = OpenLBHITConfig.model_validate(state["requested_config"])
            decision = self.physics.validate(config)
            update: Dict[str, Any] = {
                "physics_report": decision.report.model_dump(mode="json"),
                "capability_report": decision.capability.model_dump(mode="json") if decision.capability else {},
                "derived_config": decision.derived.model_dump(mode="json") if decision.derived else {},
                "warnings": list(decision.warnings),
            }
            if not decision.accepted:
                update.update(status="rejected", errors=list(decision.errors), events=_event("physics", "rejected", "; ".join(decision.errors)))
            else:
                update.update(status="validated", events=_event("physics", "ok", "physics and capability checks passed"))
            return update
        except Exception as exc:
            return {"status": "failed", "errors": [f"physics validation failed: {exc}"], "events": _event("physics", "failed", str(exc))}

    def approval(self, state: KITurbState) -> Dict[str, Any]:
        if not state.get("require_approval", self.settings.require_execution_approval):
            return {"approved": True, "status": "approved", "events": _event("approval", "skipped", "approval disabled")}
        from langgraph.types import interrupt
        config = OpenLBHITConfig.model_validate(state["requested_config"])
        answer = interrupt({
            "kind": "openlb_execution_approval",
            "message": f"Approve OpenLB HIT run '{config.name}' at {config.domain.resolution} using {config.collision.model.value}/{config.forcing.type.value}?",
            "requested_config": state["requested_config"],
            "derived_config": state.get("derived_config") or {},
        })
        approved = bool(answer if not isinstance(answer, dict) else answer.get("approved"))
        return {
            "approved": approved,
            "status": "approved" if approved else "cancelled",
            "events": _event("approval", "approved" if approved else "rejected", "user decision"),
        }

    def prepare(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors") or state.get("status") in {"rejected", "cancelled"}:
            return {}
        try:
            session = self.simulation.prepare(OpenLBHITConfig.model_validate(state["requested_config"]), Path(state.get("run_root") or self.settings.run_root))
            return {"session_path": str(Path(session.run_dir) / "simulation_session.json"), "run_id": session.run_id, "status": "prepared", "events": _event("prepare", "ok", "isolated case prepared")}
        except Exception as exc:
            return {"status": "failed", "errors": [f"case preparation failed: {exc}"], "events": _event("prepare", "failed", str(exc))}

    def compile(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        try:
            session = self.simulation.compile(_load_session(state), Path(state["openlb_app_dir"]))
            return {"session_path": str(Path(session.run_dir) / "simulation_session.json"), "effective_config": dict(session.metadata.get("effective_config") or {}), "status": "built", "events": _event("compile", "ok", "OpenLB built and verified")}
        except Exception as exc:
            return {"status": "failed", "errors": [f"compilation failed: {exc}"], "events": _event("compile", "failed", str(exc))}

    def run_collect(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        try:
            session = _load_session(state)
            self.simulation.start(session)
            supervision = self.simulation.supervise(session, poll_interval=self.settings.max_poll_seconds, timeout=self.settings.run_timeout_seconds)
            if supervision.state != "completed":
                raise RuntimeError(supervision.message or f"simulation ended in {supervision.state}")
            result = self.simulation.collect(session)
            return {"session_path": str(Path(session.run_dir) / "simulation_session.json"), "manifest_path": result.manifest_path, "status": "fetched", "events": _event("run", "ok", "simulation completed and collected")}
        except Exception as exc:
            return {"status": "failed", "errors": [f"simulation failed: {exc}"], "events": _event("run", "failed", str(exc))}

    def analyse(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        try:
            session = _load_session(state)
            processed, products = self.master.analyse(session, _load_manifest(state["manifest_path"]))
            products_path = str(products.metadata.get("products_path") or Path(session.run_dir) / "analysis" / "hit_analysis_products.json")
            if not Path(products_path).exists():
                products.save(products_path)
            status = "insufficient_data" if str(products.validation_status).lower() == "insufficient_data" else "analysed"
            return {"manifest_path": str(Path(processed.base_dir) / "dataset_manifest.json"), "analysis_products_path": products_path, "status": status, "warnings": list(products.warnings), "events": _event("analysis", "ok", status)}
        except Exception as exc:
            return {"status": "failed", "errors": [f"analysis failed: {exc}"], "events": _event("analysis", "failed", str(exc))}

    def finalize(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        try:
            session = _load_session(state)
            manifest = _load_manifest(state["manifest_path"])
            result = self.master.finalize(session=session, products=load_products_from_manifest(manifest), manifest=manifest)
            artifacts = []
            for path, label in ((result.visualization_dashboard, "HIT visualization dashboard"), (result.report_path, "HIT scientific report")):
                if path:
                    artifacts.append({"artifact_type": "downloadable_file", "path": path, "filename": Path(path).name, "message": label})
            return {"validation_path": result.validation_path or "", "dashboard_path": result.visualization_dashboard or "", "report_path": result.report_path or "", "status": result.status, "artifacts": artifacts, "events": _event("review", "ok", result.status)}
        except Exception as exc:
            return {"status": "failed", "errors": [f"finalization failed: {exc}"], "events": _event("review", "failed", str(exc))}

    def summarize(self, state: KITurbState) -> Dict[str, Any]:
        status = state.get("status", "unknown")
        text = f"KI-TURB HIT workflow finished with status: {status}."
        if state.get("report_path"):
            text += f" Report: {state['report_path']}."
        if state.get("dashboard_path"):
            text += f" Dashboard: {state['dashboard_path']}."
        if state.get("errors"):
            text += " Errors: " + "; ".join(state["errors"])
        return {"final_text": text}


__all__ = ["HITGraphServices"]
