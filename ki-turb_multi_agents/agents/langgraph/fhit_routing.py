"""OpenLB HIT lifecycle routing — thin compatibility facade over schema-first intent.

Prefer:
  classify_request(text) → plan_from_intent(intent)

Legacy helpers remain for tests and call sites.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .intent_plans import plan_from_intent
from .models import WorkflowPlan, WorkflowStep
from .request_intent import (
    ACTIVE_SIMULATION_JOB_ID,
    LATEST_SIMULATION_JOB_ID,
    AnalysisSpec,
    classify_request,
    resolve_job_ref,
    _has_case_signal,
    _wants_reuse,
    _analysis_specs,
    _backend,
    _COMPILE_ONLY,
    _BUILD_VERB,
    _RUN_VERB,
)

# Backward-compatible alias used by analysis step builders / tests.
PostRunAnalysis = AnalysisSpec


def is_load_existing_openlb_request(text: str) -> bool:
    intent = classify_request(text)
    return intent is not None and intent.action == "load"


def is_hit_simulation_request(text: str) -> bool:
    intent = classify_request(text)
    return intent is not None and intent.action in {"run", "compile"}


def resolve_existing_job_id(
    text: str,
    session_summary: Optional[Dict[str, Any]] = None,
    project_root: Optional[Any] = None,
) -> str:
    job_ref, _ = resolve_job_ref(
        text, session_summary=session_summary, project_root=project_root, prefer="load"
    )
    return job_ref


def resolve_simulation_stage(text: str) -> str:
    text = text or ""
    if _COMPILE_ONLY.search(text) or (_BUILD_VERB.search(text) and not _RUN_VERB.search(text)):
        return "compile"
    return "run"


def requested_post_run_analyses(text: str) -> List[AnalysisSpec]:
    return _analysis_specs(text or "")


def wants_energy_spectra_pipeline(text: str) -> bool:
    return any(
        a.analysis_id == "energy_spectra" or a.plot_tool == "plot_spectrum"
        for a in requested_post_run_analyses(text)
    )


def _analysis_steps(analyses: Sequence[AnalysisSpec]) -> List[WorkflowStep]:
    from .intent_plans import _analysis_steps as _steps

    return _steps(analyses)


def fhit_simulation_pipeline_plan(text: str) -> Optional[WorkflowPlan]:
    intent = classify_request(text)
    if intent is None or intent.action not in {"run", "compile"}:
        return None
    return plan_from_intent(intent)


def existing_openlb_data_plan(
    text: str,
    session_summary: Optional[Dict[str, Any]] = None,
    project_root: Optional[Any] = None,
) -> Optional[WorkflowPlan]:
    intent = classify_request(text, session_summary=session_summary, project_root=project_root)
    if intent is None or intent.action != "load":
        return None
    return plan_from_intent(intent)


def schema_plan(
    text: str,
    session_summary: Optional[Dict[str, Any]] = None,
    project_root: Optional[Any] = None,
) -> Optional[WorkflowPlan]:
    """Primary entry: classify → plan for solver/data workflows."""
    return plan_from_intent(
        classify_request(text, session_summary=session_summary, project_root=project_root)
    )


__all__ = [
    "ACTIVE_SIMULATION_JOB_ID",
    "LATEST_SIMULATION_JOB_ID",
    "PostRunAnalysis",
    "is_hit_simulation_request",
    "is_load_existing_openlb_request",
    "resolve_existing_job_id",
    "resolve_simulation_stage",
    "requested_post_run_analyses",
    "wants_energy_spectra_pipeline",
    "fhit_simulation_pipeline_plan",
    "existing_openlb_data_plan",
    "schema_plan",
    # re-export helpers used in older tests / debugging
    "_has_case_signal",
    "_wants_reuse",
    "_backend",
]
