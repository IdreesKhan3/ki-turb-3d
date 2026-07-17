"""LLM-first request planning with a thin hard gate for simulation lifecycle safety.

The hard gate is backend-agnostic (openlb today; palabos/ansys/openfoam when wired).
It only covers explicit load/status/run/compile/inquire — rather than domain-keyword pipelines.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from agents.remote_document import (
    is_remote_document_request,
    remote_document_plan_instruction,
)
from .engineering_intent import (
    extract_named_paths,
    is_document_or_latex_edit_request,
    is_simple_file_edit_request,
)
from .fhit_routing import schema_plan
from .models import WorkflowPlan, WorkflowStep
from .multi_case import is_multi_case_request
from .request_intent import classify_request
from .turn_memory import format_turn_memory

# Hard-gate actions: explicit solver/data lifecycle only (never keyword "analyze").
_HARD_GATE_ACTIONS = frozenset({
    "load", "status", "inquire", "run", "compile", "research_then_run",
})

_DATA_TOOLS = frozenset({"load_data", "load_dataset_manifest", "load_analysis_products"})
_COMPUTE_TOOLS = frozenset({
    "compute_spectra", "compute_flatness", "compute_structure_functions",
    "compute_spectral_isotropy", "compute_isotropy", "compute_pdfs",
    "compute_overview_validation", "compute_volume_field",
})
_SIM_LIFECYCLE_TOOLS = frozenset({
    "build_simulation_case", "compile_simulation", "start_simulation",
    "supervise_simulation", "fetch_simulation_outputs", "postprocess_simulation_outputs",
    "cancel_simulation",
})
_FILE_EDIT_TOOLS = frozenset({
    "read_file", "write_file", "modify_file", "find_file", "list_directory",
    "compile_latex", "run_verify_command", "run_import_check",
})

_TEXT_SOURCE_EXT = re.compile(r"\.(?:md|txt|py|toml|yml|yaml|json|cfg|ini|rst|tex|bib|cls|sty|sh)$", re.I)
_PATH_IN_TEXT = re.compile(
    r"""(?ix)
    (?P<path>
        (?:[a-z]:[\\/]|[/\\]|\.{1,2}[\\/])?
        (?:[\w.-]+[\\/])*[\w.-]+\.(?:md|txt|py|toml|yml|yaml|json|cfg|ini|rst|tex|bib|pdf|docx?|png|jpe?g)
    )
    (?=$|[\s"'.,;:)\]])
    """
)


def _prepend_active_manifest_load(
    plan: WorkflowPlan,
    session_summary: Dict[str, Any],
) -> WorkflowPlan:
    """Ensure compute steps can see postprocessed job products when a manifest is active."""
    if not plan.steps:
        return plan
    if any(step.tool in _DATA_TOOLS for step in plan.steps):
        return plan
    if not any(step.tool in _COMPUTE_TOOLS for step in plan.steps):
        return plan

    manifest_path = str(
        session_summary.get("manifest_path")
        or session_summary.get("dataset_manifest_path")
        or ""
    ).strip()
    job_id = str(
        session_summary.get("simulation_job_id")
        or session_summary.get("sim_workflow_job")
        or ""
    ).strip()
    if not manifest_path and job_id:
        manifest_path = f"simulations/{job_id}/manifest.json"
    if not manifest_path:
        return plan

    load_step = WorkflowStep(
        role="steward",
        instruction=(
            "Load the active simulation dataset manifest so analysis tools can access "
            "postprocessed products (spectra, PDFs, etc.) under processed/."
        ),
        tool="load_dataset_manifest",
        tool_args={"manifest_path": manifest_path},
    )
    return WorkflowPlan(
        steps=[load_step, *plan.steps],
        rationale=plan.rationale + " (reload active simulation manifest)",
        kind=plan.kind,
    )


def _remote_document_plan(text: str) -> Optional[WorkflowPlan]:
    if not is_remote_document_request(text):
        return None
    return WorkflowPlan(
        steps=[
            WorkflowStep(
                role="analyst",
                instruction=remote_document_plan_instruction(text),
            )
        ],
        rationale="schema:remote document",
    )


def _steward_file_edit_plan(request: str) -> WorkflowPlan:
    """Free-form steward plan for concrete create/modify of user-named files."""
    paths = extract_named_paths(request)
    path_hint = (
        ", ".join(f"`{p}`" for p in paths[:5])
        if paths
        else "the file(s) named in the request (or last_paths / find_file if only pronouns)"
    )
    documentish = is_document_or_latex_edit_request(request)
    verify = (
        "If compiling a document was requested, use an appropriate compile command "
        "on the same existing source path; otherwise self-verify code edits with "
        "run_import_check / `python -m compileall` via run_verify_command."
        if documentish
        else (
            "Self-verify with run_import_check(module=<path>) or "
            "`run_verify_command` using `python -m compileall <path>` (no second Accept)."
        )
    )
    instruction = (
        f"{request}\n\n"
        f"Fulfill this as a direct repository edit of {path_hint}. "
        "1) Locate with find_file/list_directory if needed; read_file existing targets "
        "(never assume empty; use turn_memory.last_paths for pronouns). "
        "2) write_file or modify_file with the intended content/patch. "
        "If the user said to keep/modify the current file, do not create a parallel copy. "
        f"3) {verify} "
        "Do NOT start an engineering_workflow, page_schema, or registry change. "
        "Do NOT call list_directory_files (not a tool) — use list_directory / find_file. "
        "Do NOT start analysis/simulation pipelines (load_dataset_manifest, compute_*, "
        "plot_*, build/start_simulation) unless the user explicitly asked to analyze "
        "data or run a case in this same request."
    )
    return WorkflowPlan(
        kind="agent_workflow",
        steps=[WorkflowStep(role="steward", instruction=instruction)],
        rationale="Named/document file edit → steward free-form",
    )


def _plan_looks_like_misrouted_file_edit(plan: WorkflowPlan, request: str) -> bool:
    """True if a file-edit request was planned with simulation/analysis lifecycle tools."""
    if not plan.steps:
        return False
    if not (is_document_or_latex_edit_request(request) or is_simple_file_edit_request(request)):
        return False
    tools = {str(step.tool or "") for step in plan.steps}
    tools.discard("")
    return bool(tools & (_DATA_TOOLS | _COMPUTE_TOOLS | _SIM_LIFECYCLE_TOOLS))


def _sanitize_plan(plan: WorkflowPlan, request: str) -> WorkflowPlan:
    """Normalize planner output (paths, reader tools, file-edit kind)."""
    if plan.kind == "engineering_workflow" and is_simple_file_edit_request(request):
        return _steward_file_edit_plan(request)
    if _plan_looks_like_misrouted_file_edit(plan, request):
        return _steward_file_edit_plan(request)

    steps: list[WorkflowStep] = []
    for step in plan.steps:
        tool = step.tool
        args = dict(step.tool_args or {})
        filepath = str(args.get("filepath") or "").strip()
        role = "steward" if step.role == "simulation" and tool in {
            "read_file", "read_document", "write_file", "modify_file", "find_file", "list_directory",
        } else step.role

        if tool in {"read_file", "read_document"} and not filepath:
            match = _PATH_IN_TEXT.search(step.instruction or "") or _PATH_IN_TEXT.search(request or "")
            if match:
                filepath = match.group("path").lstrip("./")
                args["filepath"] = filepath
            else:
                steps.append(
                    WorkflowStep(
                        role="steward" if role == "simulation" else role,
                        instruction=(
                            f"{step.instruction}\n"
                            "Locate the target with find_file and/or list_directory first. "
                            "Then call read_file(filepath=...) for text/source files "
                            "(.md/.py/.txt/.json/…) or read_document(filepath=...) for "
                            "PDF/Office/images. Never call a reader tool without filepath."
                        ),
                    )
                )
                continue

        if tool == "read_document" and filepath and _TEXT_SOURCE_EXT.search(filepath):
            tool = "read_file"
        elif tool == "read_file" and filepath and not _TEXT_SOURCE_EXT.search(filepath):
            if re.search(r"\.(?:pdf|docx?|png|jpe?g|gif|webp|bmp|tiff?)$", filepath, re.I):
                tool = "read_document"

        steps.append(
            WorkflowStep(
                role=role,
                instruction=step.instruction,
                tool=tool,
                tool_args=args,
            )
        )
    return WorkflowPlan(kind=plan.kind, steps=steps, rationale=plan.rationale)


def _freeform_fallback(request: str, session_summary: Dict[str, Any]) -> WorkflowPlan:
    """No LLM planner: one free-form step instead of keyword pipelines."""
    memory = format_turn_memory(
        session_summary.get("turn_memory") if isinstance(session_summary.get("turn_memory"), dict) else None
    )
    if is_multi_case_request(request) and re.search(
        r"(?i)\b(run|compile|build|simulate|start|launch)\b",
        request or "",
    ):
        instruction = (
            f"{request}\n\n{memory}\n"
            "MULTI-CASE SIMULATION LEAD: complete the FULL user request before stopping.\n"
            "- Run a full lifecycle per distinct case (build→compile→start→supervise→fetch→postprocess→load).\n"
            "- If a collision/scheme is unsupported by the catalog, report it and list supported "
            "names; do NOT silently substitute; continue remaining supported cases.\n"
            "- After cases finish, run requested analyses (e.g. compute_spectra + plot_spectrum) "
            "with ALL completed jobs on one figure when comparison was asked.\n"
            "- Self-check: report every job_id, note skipped unsupported cases, confirm the figure.\n"
            "- Do not stop after the first load_dataset_manifest."
        )
        return WorkflowPlan(
            steps=[WorkflowStep(role="simulation", instruction=instruction)],
            rationale="Free-form simulation multi-case (LLM planner unavailable)",
        )
    instruction = (
        f"{request}\n\n{memory}\n"
        "Solve this with your authorized tools. Prefer locate/read/search/compare when the "
        "user asks about code, formulae, capability, or prior findings. "
        "Do NOT call build_simulation_case, compile_simulation, start_simulation, or "
        "compute_*/plot_* unless the user explicitly asked to run a case or produce a figure now. "
        "After write_file/modify_file for code, self-validate with run_import_check and/or run_pytest. "
        "Use last_paths / prior tools for pronouns like 'that file'."
    )
    return WorkflowPlan(
        steps=[WorkflowStep(role="steward", instruction=instruction)],
        rationale="Free-form steward (LLM planner unavailable)",
    )


class RequestRouter:
    def __init__(self, planner_agent=None, max_steps: int = 8, project_root=None):
        self.planner_agent = planner_agent
        self.max_steps = max_steps
        self.project_root = project_root

    def deterministic_plan(
        self,
        request: str,
        session_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowPlan]:
        """Hard gate for solver lifecycle and file edits; otherwise defer to the LLM planner."""
        text = (request or "").strip()
        if not text:
            return None
        session_summary = session_summary or {}

        # Document edits before solver compile/run classification.
        if is_document_or_latex_edit_request(text):
            return _steward_file_edit_plan(text)

        # Explicit solver lifecycle (load/status/run/compile/inquire).
        # Multi-case runs stay with the planner / free-form simulation lead.
        intent = classify_request(
            text,
            session_summary=session_summary,
            project_root=self.project_root,
        )
        multi_lifecycle = bool(
            intent is not None
            and is_multi_case_request(text)
            and intent.action in {"run", "compile", "research_then_run"}
        )
        if intent is not None and intent.action in _HARD_GATE_ACTIONS:
            if multi_lifecycle:
                return None
            return schema_plan(text, session_summary, self.project_root)

        if not multi_lifecycle:
            remote_plan = _remote_document_plan(text)
            if remote_plan is not None:
                return remote_plan

        if is_simple_file_edit_request(text):
            return _steward_file_edit_plan(text)

        return None

    def plan(self, request: str, session_summary: Dict[str, Any]) -> WorkflowPlan:
        session_summary = dict(session_summary or {})
        deterministic = self.deterministic_plan(request, session_summary)
        if deterministic is not None:
            return _sanitize_plan(
                _prepend_active_manifest_load(deterministic, session_summary),
                request,
            )

        if self.planner_agent is None:
            return _sanitize_plan(_freeform_fallback(request, session_summary), request)

        memory_block = format_turn_memory(
            session_summary.get("turn_memory") if isinstance(session_summary.get("turn_memory"), dict) else None
        )
        prompt = (
            f"User request:\n{request}\n\n"
            f"Session context:\n{session_summary}\n\n"
            f"{memory_block}\n\n"
            f"Return at most {self.max_steps} ordered steps.\n"
            "Reason about intent from the full request and turn_memory. "
            "Domain nouns alone are not execute triggers — distinguish answer/compare/search "
            "from requests to run a case or produce a figure/analysis now.\n"
            "Repo/file work: steward free-form (locate then read); never empty filepath.\n"
            "Code writes: instruct self-validation. Resolve pronouns via turn_memory."
        )
        result = self.planner_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        plan = WorkflowPlan.model_validate(result["structured_response"])
        plan.steps = plan.steps[: self.max_steps]
        return _sanitize_plan(
            _prepend_active_manifest_load(plan, session_summary),
            request,
        )


__all__ = ["RequestRouter", "_sanitize_plan"]
