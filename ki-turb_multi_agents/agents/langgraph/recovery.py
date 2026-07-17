"""Recover from step failures by handing off to a better role instead of dying."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import WorkflowPlan, WorkflowStep

MAX_RECOVER_ATTEMPTS = 2

_UNAUTHORIZED = re.compile(
    r"(?i)\b("
    r"unknown tool|not (?:authorized|allowed|permitted)|"
    r"tool (?:not|isn't|is not) (?:available|allowed)|"
    r"forbidden|no permission|cannot use tool"
    r")\b"
)
_PHYSICS = re.compile(
    r"(?i)\b("
    r"physics validation failed|step0_divergence|divergence_budget|"
    r"acceptance limit|constraint check"
    r")\b"
)
_FILE_MISSING = re.compile(
    r"(?i)\b("
    r"file not found|no files matching|directory not found"
    r")\b"
)
_MISSING_DATA = re.compile(
    r"(?i)\b("
    r"no (?:active )?job_id|requires an active simulation job_id|"
    r"no saved simulation|eps_real_validation|turbulence_validation|manifest|"
    r"use data_dir|csv_path"
    r")\b"
)
_FILE_OPS = re.compile(
    r"(?i)\b("
    r"filepath|read_file|write_file|delete_file|find_file|"
    r"list_directory|search_codebase|codebase|repository"
    r")\b"
)


def recovery_plan(
    *,
    user_request: str,
    failure: str,
    task_results: Optional[List[Dict[str, Any]]] = None,
    planner_agent: Any = None,
) -> WorkflowPlan:
    """Build a short recovery plan: hand off, explain, or reassess — do not repeat the same dead step blindly."""
    fail = (failure or "").strip()
    user = (user_request or "").strip()
    evidence = _prior_evidence(task_results)

    if _UNAUTHORIZED.search(fail) or (_FILE_OPS.search(fail) and "simulation" in fail.lower()):
        return WorkflowPlan(
            steps=[
                WorkflowStep(
                    role="steward",
                    instruction=(
                        "HANDOFF: the previous specialist could not use the required tool.\n"
                        f"Failure: {fail}\nUser request: {user}\n{evidence}\n"
                        "Complete the user request with your authorized tools "
                        "(locate/read/search/write/verify as needed). "
                        "Do not restart an OpenLB run unless explicitly asked."
                    ),
                )
            ],
            rationale="Recover: unauthorized/wrong-role → steward handoff",
        )

    if _PHYSICS.search(fail):
        return WorkflowPlan(
            steps=[
                WorkflowStep(
                    role="orchestrator",
                    instruction=(
                        "A simulation build/validation step failed. Do NOT blindly rebuild "
                        "with the same parameters.\n"
                        f"Failure: {fail}\nUser request: {user}\n{evidence}\n"
                        "Explain the failure clearly and propose a corrected next request "
                        "(e.g. compile-only, retuned grid/Mach/IC band). "
                        "If the user only asked to compile, say whether compile was reached."
                    ),
                )
            ],
            rationale="Recover: physics failure → explain + propose fix",
        )

    if _FILE_MISSING.search(fail):
        return WorkflowPlan(
            steps=[
                WorkflowStep(
                    role="steward",
                    instruction=(
                        "Previous file open/list failed.\n"
                        f"Failure: {fail}\nUser request: {user}\n{evidence}\n"
                        "Do NOT retry the same bare filepath. "
                        "find_file(pattern=<basename>) and/or list_directory under "
                        "simulations/<job_id> from turn_memory/session, then "
                        "read_file with the full relative path. "
                        "locate/list tools ARE available to steward."
                    ),
                )
            ],
            rationale="Recover: missing file → locate then read",
        )

    if _MISSING_DATA.search(fail):
        return WorkflowPlan(
            steps=[
                WorkflowStep(
                    role="steward",
                    instruction=(
                        "Previous step lacked job/data context.\n"
                        f"Failure: {fail}\nUser request: {user}\n{evidence}\n"
                        "If the user asked how many/which saved jobs exist: call "
                        "list_simulation_jobs (not load_dataset_manifest). "
                        "Otherwise locate the active job/manifest with "
                        "list_simulation_jobs / list_directory / find_file. "
                        "Report counts and paths; do not invent plots without products."
                    ),
                )
            ],
            rationale="Recover: missing data → steward locate/report",
        )

    if planner_agent is not None:
        try:
            prompt = (
                "A workflow step FAILED. Produce a short recovery plan (1–2 steps).\n"
                "Hand off to the role that can finish the user's request. "
                "Do not repeat the exact failed tool call with identical args.\n"
                f"User request:\n{user}\n\nFailure:\n{fail}\n\n{evidence}\n\n"
                "If the failure is wrong-role/unauthorized → steward or the correct specialist. "
                "If physics/build failed → orchestrator explains; do not silent-rebuild. "
                "If data missing → steward locates products. "
                "If the user asked a question/compare → answer/read, do not start simulations."
            )
            result = planner_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            plan = WorkflowPlan.model_validate(result["structured_response"])
            if plan.steps:
                plan.rationale = plan.rationale or "Recover: LLM handoff plan"
                plan.steps = plan.steps[:2]
                return plan
        except Exception:
            pass

    return WorkflowPlan(
        steps=[
            WorkflowStep(
                role="orchestrator",
                instruction=(
                    "A specialist step failed. Reassess and finish helpfully.\n"
                    f"Failure: {fail}\nUser request: {user}\n{evidence}\n"
                    "Either answer from evidence, or describe which specialist/tools "
                    "should run next. Do not claim success."
                ),
            )
        ],
        rationale="Recover: orchestrator reassess",
    )


def _prior_evidence(task_results: Optional[List[Dict[str, Any]]]) -> str:
    lines = ["Prior step evidence:"]
    for item in (task_results or [])[-3:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or "?"
        text = str(item.get("text") or "").strip()
        if text:
            lines.append(f"- {role}: {text[:400]}")
    if len(lines) == 1:
        lines.append("- (none)")
    return "\n".join(lines)


__all__ = ["MAX_RECOVER_ATTEMPTS", "recovery_plan"]
