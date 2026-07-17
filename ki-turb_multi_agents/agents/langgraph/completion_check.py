"""Pre-finalize acceptance checklist — keep working until the user request is met."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .models import RoleName, WorkflowPlan, WorkflowStep
from .multi_case import is_multi_case_request

MAX_COMPLETION_ATTEMPTS = 2

_SPECTRA = re.compile(
    r"(?i)\b(spectra|spectrum|compute_spectra|plot_spectrum|e\s*\(\s*k\s*\)|kolmogorov)\b"
)
_PLOT = re.compile(r"(?i)\b(plot|figure|visuali[sz]e|show\s+(?:it|the\s+figure)\s+here)\b")
_JOB_IDS_ASKED = re.compile(r"(?i)\bjob[_\s-]?ids?\b")
_SKIPPED_CASE = re.compile(
    r"(?i)\b("
    r"unsupported\s+collision|not\s+supported|skipped\s+case|"
    r"could\s+not\s+run\s+case|invalid\s+collision"
    r")\b"
)
_JOB_ID = re.compile(r"\b(job_[a-f0-9]+)\b", re.I)
_COMPUTE_DONE = re.compile(r"(?i)\bcompute_spectra\b")
_PLOT_DONE = re.compile(r"(?i)\bplot_spectrum\b")


def _evidence_text(
    task_results: Sequence[Dict[str, Any]] | None,
    final_text: str = "",
) -> str:
    parts: List[str] = []
    for item in task_results or []:
        if isinstance(item, dict):
            parts.append(str(item.get("text") or ""))
            for out in item.get("tool_outputs") or []:
                parts.append(str(out))
    parts.append(final_text or "")
    return "\n".join(parts)


def _job_ids_from_evidence(
    evidence: str,
    session_context: Optional[Dict[str, Any]],
) -> List[str]:
    found: List[str] = []
    seen: set[str] = set()
    for match in _JOB_ID.finditer(evidence or ""):
        jid = match.group(1)
        if jid not in seen:
            found.append(jid)
            seen.add(jid)
    for jid in list((session_context or {}).get("comparison_job_ids") or []):
        jid = str(jid or "").strip()
        if jid and jid not in seen and not jid.startswith("__"):
            found.append(jid)
            seen.add(jid)
    active = str((session_context or {}).get("simulation_job_id") or "").strip()
    if active and active not in seen and not active.startswith("__"):
        found.append(active)
    return found


def evaluate_completion(
    *,
    user_request: str,
    task_results: Optional[Sequence[Dict[str, Any]]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Sequence[Any]] = None,
    final_text: str = "",
) -> Dict[str, Any]:
    """
    Deterministic checklist against the user request.

    Returns {complete: bool, missing: [str], gaps: [str]}.
    """
    request = (user_request or "").strip()
    evidence = _evidence_text(task_results, final_text)
    session_context = session_context or {}
    missing: List[str] = []
    gaps: List[str] = []

    job_ids = _job_ids_from_evidence(evidence, session_context)
    multi = is_multi_case_request(request)
    skipped_ok = bool(_SKIPPED_CASE.search(evidence))

    if multi:
        if len(job_ids) < 2 and not skipped_ok:
            missing.append("multiple_cases")
            gaps.append(
                "Need at least two completed job_ids (or an explicit unsupported-case skip "
                "plus the remaining case(s))."
            )
        elif len(job_ids) < 1:
            missing.append("multiple_cases")
            gaps.append("No completed simulation job_id found for the multi-case request.")

    wants_spectra = bool(_SPECTRA.search(request))
    wants_plot = bool(_PLOT.search(request)) or wants_spectra
    has_figure = bool(artifacts) or bool(
        session_context.get("last_figure") or session_context.get("last_figure_image")
    )
    if wants_spectra and not (_COMPUTE_DONE.search(evidence) or "spectra" in evidence.lower()):
        # Allow product evidence from successful overlay without tool name echo.
        if "spectrum" not in evidence.lower() and not has_figure:
            missing.append("spectra")
            gaps.append("Energy spectra were requested but compute/plot evidence is missing.")
    if wants_plot and not (has_figure or _PLOT_DONE.search(evidence) or "figure" in evidence.lower()):
        missing.append("figure")
        gaps.append("A figure/plot was requested but no figure artifact or plot evidence was found.")

    if _JOB_IDS_ASKED.search(request) and not job_ids:
        missing.append("job_ids")
        gaps.append("User asked for job_ids but none appear in the evidence.")

    return {
        "complete": not missing,
        "missing": missing,
        "gaps": gaps,
        "job_ids": job_ids,
        "multi_case": multi,
    }


def finish_work_plan(
    *,
    user_request: str,
    evaluation: Dict[str, Any],
    attempt: int,
) -> WorkflowPlan:
    """Inject a free-form finish step for remaining gaps."""
    gaps = "; ".join(evaluation.get("gaps") or ["unfinished deliverables"])
    missing = ", ".join(evaluation.get("missing") or ["unknown"])
    role: RoleName = "simulation"
    missing_set = set(evaluation.get("missing") or [])
    if missing_set <= {"spectra", "figure"} and evaluation.get("job_ids"):
        role = "analyst"
    if missing_set == {"figure"}:
        role = "visualizer"

    if role == "visualizer":
        instruction = (
            f"COMPLETION SELF-CHECK FAILED (attempt {attempt + 1}/{MAX_COMPLETION_ATTEMPTS}).\n"
            f"User request:\n{user_request}\n\n"
            f"Gaps: {gaps}\n"
            "Produce the missing figure with the registered plot_* tool from session "
            "analysis products / data_directories. Overlay all completed simulations when "
            "comparison was requested. Report job_ids in your reply."
        )
    else:
        instruction = (
            f"COMPLETION SELF-CHECK FAILED (attempt {attempt + 1}/{MAX_COMPLETION_ATTEMPTS}).\n"
            f"User request:\n{user_request}\n\n"
            f"Missing: {missing}\nGaps: {gaps}\n"
            f"Known job_ids so far: {', '.join(evaluation.get('job_ids') or []) or '(none)'}\n\n"
            "Finish the REMAINING work only. Do not claim success until the gaps are closed.\n"
            "- Multi-case: run any unfinished supported cases (full lifecycle each).\n"
            "- Unsupported collision: report + list catalog; do not silently substitute; "
            "continue other cases.\n"
            "- Spectra/compare: compute_spectra + plot_spectrum overlay on completed jobs.\n"
            "- End by reporting every job_id and what was skipped."
        )
        if role == "analyst":
            instruction += (
                "\nYou are analyst lead for remaining compute — use load_dataset_manifest / "
                "load_analysis_products as needed, then compute_* tools."
            )
    return WorkflowPlan(
        steps=[WorkflowStep(role=role, instruction=instruction)],
        rationale=f"completion_check:finish missing={missing}",
    )


def exhaustion_message(user_request: str, evaluation: Dict[str, Any]) -> str:
    gaps = evaluation.get("gaps") or ["Could not verify full completion."]
    jobs = evaluation.get("job_ids") or []
    return (
        "Stopped after completion self-check budget without fully satisfying the request.\n"
        f"Still missing: {'; '.join(gaps)}\n"
        f"Job ids seen: {', '.join(jobs) if jobs else '(none)'}\n"
        "Re-run or clarify unsupported parameters (e.g. collision names) if needed.\n"
        f"Original request: {user_request[:500]}"
    )


__all__ = [
    "MAX_COMPLETION_ATTEMPTS",
    "evaluate_completion",
    "finish_work_plan",
    "exhaustion_message",
]
