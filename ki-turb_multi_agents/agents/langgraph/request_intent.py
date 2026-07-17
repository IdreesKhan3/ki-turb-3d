"""Schema-first request classification for solver/data workflows.

Flow:
  text → RequestIntent {action, backend, job_ref?, case_params?, analyses[]}
       → plan_from_intent(intent) → WorkflowPlan

Action decides the tool graph. Keyword cues only fill the schema; they do not
directly choose tools.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.intent_detection import collect_analysis_intents
from agents.page_schema import (
    INTENT_OTHER_TURBULENCE_STATS,
    get_page_for_intent,
    get_routing_for_intent,
    get_workflow_for_page,
)

from .openlb_hit_build import has_explicit_openlb_case_params, parse_openlb_build_args

RequestAction = Literal[
    "load", "run", "compile", "status", "analyze", "inquire", "research", "research_then_run"
]
# Solver backends share the same lifecycle tools; archive is page/example data only.
BackendName = Literal["openlb", "palabos", "ansys", "openfoam", "archive"]
_SOLVER_BACKENDS = frozenset({"openlb", "palabos", "ansys", "openfoam"})

ACTIVE_SIMULATION_JOB_ID = "__active_simulation_job__"
LATEST_SIMULATION_JOB_ID = "__latest_simulation_job__"
# Resolved at execute time from session comparison_job_ids (multi-scheme runs).
COMPARISON_JOB_DIRS = "__comparison_job_dirs__"

_HIT = re.compile(r"\b(fhit|dhit|hit|homogeneous isotropic turbulence)\b", re.I)
_OPENLB = re.compile(r"\bopenlb\b", re.I)
_PALABOS = re.compile(r"\bpalabos\b", re.I)
_ANSYS = re.compile(r"\bansys\b", re.I)
_OPENFOAM = re.compile(r"\bopenfoam\b", re.I)
_JOB_ID = re.compile(r"\b(job_[a-f0-9]+)\b", re.I)
_ARCHIVE = re.compile(r"\bexamples/(?:dns|les)/\b", re.I)

# Structural cues (small closed sets — fill schema fields, not tool graphs).
_REUSE = re.compile(
    r"(?i)\b("
    r"load|reload|reuse|"
    r"already\s+(?:saved|run|fetched)|"
    r"existing\s+(?:job|simulation|data|results?|manifest|openlb|palabos|ansys|openfoam)|"
    r"previous\s+(?:job|simulation|run|data|results?|openlb|palabos|ansys|openfoam)|"
    r"saved\s+(?:agent|agents|job|simulation|data|results?|manifest|openlb|palabos|ansys|openfoam)|"
    r"from\s+(?:the\s+)?(?:saved|existing|previous|prior)\b|"
    r"agent(?:s)?\s+data"
    r")\b"
)
_CREATE = re.compile(
    r"(?i)\b(run|simulate|execute|start|launch|compile|build|create|prepare|configure|set)\b"
)
_RUN_VERB = re.compile(
    r"(?i)\b(run|simulate|execute|start|launch|monitor|fetch|smoke\s*test|supervise|postprocess)\b"
)
_BUILD_VERB = re.compile(r"(?i)\b(compile|build)\b")
_COMPILE_ONLY = re.compile(
    r"(?i)\b("
    r"compile\s+only|only\s+compile|just\s+compile|"
    r"compile\s+and\s+stop|stop\s+after\s+compil\w*|"
    r"and\s+stop\b|"
    r"do\s+not\s+(?:re)?(?:run|start)|don'?t\s+(?:re)?(?:run|start)|"
    r"without\s+(?:re)?(?:run|start)(?:ning|ing)?|"
    r"no\s+run|skip\s+(?:the\s+)?run|"
    r"do\s+not\s+start\s+the\s+solver|don'?t\s+start\s+the\s+solver"
    r")\b"
)
# Strip negated run/start phrases so "do not start the solver" does not count as run.
_NEGATED_RUN = re.compile(
    r"(?i)\b(?:do\s+not|don'?t|never|without)\s+(?:re)?(?:run|start|launch|execute)\b[^.——\n]*"
)


def _positive_run_verb(text: str) -> bool:
    """True if the user asked to run/start (ignoring 'do not start/run' negations)."""
    cleaned = _NEGATED_RUN.sub(" ", text or "")
    return bool(_RUN_VERB.search(cleaned))


def _is_compile_only_request(text: str) -> bool:
    """Compile/build without starting the solver (negated start/run does not count as run)."""
    raw = text or ""
    if _positive_run_verb(raw):
        return False
    if _COMPILE_ONLY.search(raw):
        return True
    return bool(_BUILD_VERB.search(raw))
_STATUS = re.compile(
    r"(?i)\b("
    r"simulation\s+status|check\s+(?:the\s+)?(?:simulation\s+)?status|"
    r"cancel\s+(?:the\s+)?simulation|is\s+(?:the\s+)?(?:job|simulation)\s+running|"
    r"monitor\s+(?:the\s+)?(?:job|simulation)|"
    r"did you (?:really )?(?:compile|build|run)|"
    r"(?:was|were).{0,20}(?:compiled|built|run successfully)"
    r")\b"
)
_INQUIRE = re.compile(
    r"(?i)\b("
    r"where|what path|which (?:dir|folder|directory)|location of|path to"
    r")\b"
)
_EXPLAIN = re.compile(r"(?i)\b(what is|how does|explain|describe|tell me about)\b")
_RESEARCH = re.compile(
    r"(?i)\b("
    r"search\s+(?:the\s+)?(?:web|online|internet)|"
    r"look\s+up|lookup|"
    r"google|"
    r"from\s+(?:the\s+)?(?:web|literature|papers?|arxiv)|"
    r"research\s+(?:online|papers?|literature)|"
    r"find\s+(?:papers?|references?|docs?|documentation)|"
    r"browse\s+(?:the\s+)?web|"
    r"learn\s+from\s+(?:the\s+)?web"
    r")\b"
)
_RESEARCH_THEN_ACT = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:first|then).{0,100}(?:search|look\s*up|learn|research)"
    r"|(?:search|look\s*up|learn|research).{0,160}(?:then|after\s+that|and\s+then|based\s+on)"
    r"|based\s+on\s+(?:those|these|the)\s+(?:parameters|findings|results|research)"
    r"|learn.{0,80}(?:parameters|settings|config).{0,80}(?:run|simulate)"
    r")"
)
_LBM = re.compile(r"(?i)\b(smagorinsky|wale|mrt|bgk|trt|rlb|lattice\s+boltzmann|lbm)\b")
_GRID = re.compile(r"(?i)\b\d+\s*(?:\^|×|x)\s*3\b|\b\d+\s*grid\b")
_COMPUTE_OVERRIDES = {"real_isotropy": "compute_isotropy"}
_PLOT_ARG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    INTENT_OTHER_TURBULENCE_STATS: {
        "traces": [
            {"data_source": "turbulence_stats", "x_col": "iter", "y_col": "eps_real", "label": "ε_real"},
            {"data_source": "turbulence_stats", "x_col": "iter", "y_col": "eps_spectral", "label": "ε_spectral"},
        ],
        "axis_labels": {"x": "Iteration", "y": "Dissipation rate"},
    },
}


class AnalysisSpec(BaseModel):
    """One analysis product chain derived from page_schema."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    compute_tool: Optional[str] = None
    compute_tool_args: Dict[str, Any] = Field(default_factory=dict)
    plot_tool: str
    plot_tool_args: Dict[str, Any] = Field(default_factory=dict)


class RequestIntent(BaseModel):
    """Normalized user request — the only input to solver/data plan builders."""

    model_config = ConfigDict(extra="forbid")

    action: RequestAction
    backend: Optional[BackendName] = None
    job_ref: Optional[str] = None
    job_ref_source: Optional[Literal["explicit", "session", "latest", "active", "none"]] = None
    case_params: Dict[str, Any] = Field(default_factory=dict)
    analyses: List[AnalysisSpec] = Field(default_factory=list)
    raw_text: str = ""
    rationale: str = ""


def _named_solver(text: str) -> Optional[BackendName]:
    """Explicit solver name in text (first match wins by specificity order)."""
    if _OPENLB.search(text):
        return "openlb"
    if _PALABOS.search(text):
        return "palabos"
    if _ANSYS.search(text):
        return "ansys"
    if _OPENFOAM.search(text):
        return "openfoam"
    return None


def _backend(text: str) -> Optional[BackendName]:
    named = _named_solver(text)
    if named:
        return named
    if _ARCHIVE.search(text):
        return "archive"
    # HIT / LBM case language currently maps to the OpenLB HIT app (only wired HIT solver).
    # Other solvers must be named explicitly until their case builders are registered.
    if _HIT.search(text):
        return "openlb"
    if _LBM.search(text) and _GRID.search(text) and _CREATE.search(text):
        return "openlb"
    return None


def _is_solver_backend(backend: Optional[str]) -> bool:
    return backend in _SOLVER_BACKENDS


def _has_case_signal(text: str) -> bool:
    """True only from explicit case/create cues — never from naming a solver alone."""
    return bool(
        _CREATE.search(text)
        or _GRID.search(text)
        or _LBM.search(text)
        or has_explicit_openlb_case_params(text)
    )


def _case_params_for_backend(backend: BackendName, text: str) -> Dict[str, Any]:
    """Backend-specific case parsing; shared lifecycle tools consume the result."""
    if backend == "openlb":
        params = parse_openlb_build_args(text)
        params.setdefault("backend", "openlb")
        return params
    # Stubs for future solvers — build_simulation_case already accepts backend name.
    return {"backend": backend, "name": backend.upper()}


def _wants_lifecycle(text: str) -> bool:
    """True when the user is asking to build/compile/run a case (not just research)."""
    return bool(_CREATE.search(text) or _BUILD_VERB.search(text) or _RUN_VERB.search(text))


def _research_then_run_hints(text: str, backend: BackendName) -> Dict[str, Any]:
    """Minimal case hints for research-guided runs (no calibrated defaults)."""
    lower = (text or "").lower()
    hints: Dict[str, Any] = {"backend": backend}
    if backend == "openlb" or _HIT.search(text):
        hints["flow"] = "hit"
        if re.search(r"\bdhit\b", lower) or re.search(r"\bdecaying\b", lower):
            hints["hit_mode"] = "decaying"
            hints["name"] = "DHIT"
        elif re.search(r"\bfhit\b", lower) or re.search(r"\bforced\b", lower):
            hints["hit_mode"] = "forced"
            hints["name"] = "FHIT"
        else:
            hints["hit_mode"] = "forced"
            hints["name"] = "HIT"
    else:
        hints["name"] = str(backend).upper()
    if re.search(r"\bvalidate\b", lower):
        hints["validate"] = True
    return hints


def _wants_research_then_run(text: str) -> bool:
    """True for Cursor-like: research/learn online, then run/build with findings."""
    if not (_RESEARCH.search(text) or _RESEARCH_THEN_ACT.search(text)):
        return False
    if not _wants_lifecycle(text):
        return bool(_RESEARCH_THEN_ACT.search(text))
    return True


_PLOT_OR_COMPUTE = re.compile(
    r"(?i)\b(plot|compute|calculate|show\s+me\s+the\s+figure|run\s+analysis)\b"
)


def _wants_reuse(text: str) -> bool:
    if not _REUSE.search(text):
        return False
    # "set ... load outputs" during a new case is still create, unless reuse is explicit.
    if _CREATE.search(text) and _has_case_signal(text) and not re.search(
        r"(?i)\b(already|existing|saved|previous|prior|agent(?:s)?\s+data)\b",
        text,
    ):
        return False
    return True


def _analysis_specs(text: str) -> List[AnalysisSpec]:
    found: List[AnalysisSpec] = []
    seen_plot: set[str] = set()
    for intent in collect_analysis_intents(text or ""):
        page_id = get_page_for_intent(intent)
        routing = get_routing_for_intent(intent, text)
        plot_tool = routing.get("tool")
        if not plot_tool or str(plot_tool).startswith("get_"):
            continue
        if plot_tool in seen_plot:
            continue
        cfg = get_workflow_for_page(page_id) if page_id else {}
        compute_tool = cfg.get("compute_tool") or _COMPUTE_OVERRIDES.get(page_id or "")
        plot_args = dict(_PLOT_ARG_OVERRIDES.get(intent) or {})
        if intent == INTENT_OTHER_TURBULENCE_STATS and not re.search(
            r"(?i)\b(dissipation|eps_spectral|eps_real|\beps\b|energy\s+balance|tke)\b",
            text or "",
        ):
            plot_args = {}
        found.append(
            AnalysisSpec(
                analysis_id=str(intent),
                compute_tool=compute_tool,
                plot_tool=str(plot_tool),
                plot_tool_args=plot_args,
            )
        )
        seen_plot.add(str(plot_tool))
    return found


def resolve_job_ref(
    text: str = "",
    *,
    session_summary: Optional[Dict[str, Any]] = None,
    project_root: Optional[str | Path] = None,
    prefer: Literal["load", "active"] = "load",
) -> tuple[str, Literal["explicit", "session", "latest", "active", "none"]]:
    """Resolve job_ref: explicit id → session → latest manifest job."""
    match = _JOB_ID.search(text or "")
    if match:
        return match.group(1), "explicit"

    sess = session_summary or {}
    for key in ("simulation_job_id", "sim_workflow_job"):
        value = str(sess.get(key) or "").strip()
        if value and value not in {ACTIVE_SIMULATION_JOB_ID, LATEST_SIMULATION_JOB_ID}:
            return value, "session"

    if prefer == "active":
        return ACTIVE_SIMULATION_JOB_ID, "active"

    if project_root is not None:
        from agents.tools.simulation import _store as job_store

        root = Path(project_root)
        latest = job_store.latest_job_id_with_manifest(root) or job_store.latest_job_id(root)
        if latest:
            return latest, "latest"

    return LATEST_SIMULATION_JOB_ID, "latest"


def classify_request(
    text: str,
    *,
    session_summary: Optional[Dict[str, Any]] = None,
    project_root: Optional[str | Path] = None,
) -> Optional[RequestIntent]:
    """Classify explicit solver/data lifecycle intents — not keyword analyze pipelines."""
    text = (text or "").strip()
    if not text:
        return None

    backend = _backend(text)
    analyses = _analysis_specs(text)
    session_summary = session_summary or {}

    # 1) Load existing data (never create).
    if _wants_reuse(text) and (
        _is_solver_backend(backend) or _JOB_ID.search(text) or session_summary.get("simulation_job_id")
    ):
        job_ref, source = resolve_job_ref(
            text, session_summary=session_summary, project_root=project_root, prefer="load"
        )
        return RequestIntent(
            action="load",
            backend=backend if _is_solver_backend(backend) else None,
            job_ref=job_ref,
            job_ref_source=source,
            analyses=analyses,
            raw_text=text,
            rationale="Reuse saved simulation job data",
        )

    # 2) Status / cancel of an existing job.
    if _STATUS.search(text) and (
        _is_solver_backend(backend) or _JOB_ID.search(text) or session_summary
    ):
        job_ref, source = resolve_job_ref(
            text, session_summary=session_summary, project_root=project_root, prefer="load"
        )
        return RequestIntent(
            action="status",
            backend=backend if _is_solver_backend(backend) else None,
            job_ref=job_ref,
            job_ref_source=source,
            raw_text=text,
            rationale="Query existing simulation job status",
        )

    # 3) Storage location inquiry (not a job lifecycle).
    if (
        (_is_solver_backend(backend) or session_summary.get("simulation_job_id"))
        and _INQUIRE.search(text)
        and not _has_case_signal(text)
    ):
        return RequestIntent(
            action="inquire",
            backend=backend if _is_solver_backend(backend) else None,
            raw_text=text,
            rationale="Simulation storage location inquiry",
        )

    # 3a) Research online THEN run/build (Cursor-like compound tasks).
    if _is_solver_backend(backend) and _wants_research_then_run(text):
        assert backend is not None
        stage: RequestAction = "compile" if (
            _is_compile_only_request(text) and "then run" not in text.lower()
        ) else "run"
        action: RequestAction = "research_then_run"
        hints = _research_then_run_hints(text, backend)
        if stage == "compile":
            hints["stop_after"] = "compile"
        return RequestIntent(
            action=action,
            backend=backend,
            job_ref=ACTIVE_SIMULATION_JOB_ID,
            job_ref_source="active",
            case_params=hints,
            analyses=analyses if stage == "run" else [],
            raw_text=text,
            rationale=f"Web research then {backend} lifecycle from learned parameters",
        )

    # 3b) Explicit web / literature research (planner may use this; not an analyze hijack).
    if _RESEARCH.search(text) and not _wants_lifecycle(text):
        return RequestIntent(
            action="research",
            backend=backend,
            raw_text=text,
            rationale="Web / literature research request",
        )

    if (
        _EXPLAIN.search(text)
        and not _wants_lifecycle(text)
        and not _PLOT_OR_COMPUTE.search(text)
        and not _has_case_signal(text)
    ):
        return RequestIntent(
            action="research",
            backend=backend,
            raw_text=text,
            rationale="Explain/theory request via web-backed research",
        )

    # Archive example paths are page data unless a solver backend is explicit.
    if backend == "archive" and not _named_solver(text):
        return None

    # 4) Create / compile / run — any wired solver; requires lifecycle intent, not mere keywords.
    if _is_solver_backend(backend) and _has_case_signal(text) and _wants_lifecycle(text):
        assert backend is not None
        stage = "compile" if _is_compile_only_request(text) else "run"
        case_params = _case_params_for_backend(backend, text)
        if backend == "openlb" and case_params.get("hit_mode") is None:
            case_params["hit_mode"] = "forced"
        job_ref, source = (
            (ACTIVE_SIMULATION_JOB_ID, "active")
            if stage in {"run", "compile"}
            else resolve_job_ref(text, session_summary=session_summary, project_root=project_root)
        )
        return RequestIntent(
            action=stage,
            backend=backend,
            job_ref=job_ref,
            job_ref_source=source,
            case_params=case_params,
            analyses=analyses if stage == "run" else [],
            raw_text=text,
            rationale=f"{backend} lifecycle ({stage})",
        )

    # Analysis/plot-from-keywords is intentionally NOT classified here — LLM planner decides.
    return None


__all__ = [
    "ACTIVE_SIMULATION_JOB_ID",
    "LATEST_SIMULATION_JOB_ID",
    "COMPARISON_JOB_DIRS",
    "AnalysisSpec",
    "RequestIntent",
    "classify_request",
    "resolve_job_ref",
]
