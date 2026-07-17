"""Map RequestIntent → WorkflowPlan tool graphs.

The action field alone selects the graph. No keyword soup here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .models import WorkflowPlan, WorkflowStep
from .request_intent import (
    ACTIVE_SIMULATION_JOB_ID,
    AnalysisSpec,
    RequestIntent,
)


def _backend_label(intent: RequestIntent) -> str:
    return str(intent.backend or "simulation")


def _analysis_steps(analyses: Sequence[AnalysisSpec]) -> List[WorkflowStep]:
    steps: List[WorkflowStep] = []
    compute_done: set[str] = set()
    for analysis in analyses:
        if analysis.compute_tool and analysis.compute_tool not in compute_done:
            steps.append(
                WorkflowStep(
                    role="analyst",
                    instruction=(
                        f"Run the registered {analysis.compute_tool} tool on the "
                        "simulation products now loaded in the session."
                    ),
                    tool=analysis.compute_tool,
                    tool_args=dict(analysis.compute_tool_args or {}),
                )
            )
            compute_done.add(analysis.compute_tool)
        steps.append(
            WorkflowStep(
                role="visualizer",
                instruction=(
                    f"Create the requested figure with {analysis.plot_tool} from the "
                    f"session analysis products (intent={analysis.analysis_id})."
                ),
                tool=analysis.plot_tool,
                tool_args=dict(analysis.plot_tool_args or {}),
            )
        )
    return steps


def _lifecycle_steps(build_args: Dict[str, Any], label: str, job_id: str) -> List[WorkflowStep]:
    job_ref = {"job_id": job_id}
    name = build_args.get("name") or label
    return [
        WorkflowStep(
            role="simulation",
            instruction=f"Build the {label} case '{name}' from the parsed user parameters.",
            tool="build_simulation_case",
            tool_args=dict(build_args),
        ),
        WorkflowStep(
            role="simulation",
            instruction=f"Compile the prepared {label} solver case.",
            tool="compile_simulation",
            tool_args=dict(job_ref),
        ),
        WorkflowStep(
            role="simulation",
            instruction=f"Launch the {label} simulation job.",
            tool="start_simulation",
            tool_args=dict(job_ref),
        ),
        WorkflowStep(
            role="simulation",
            instruction="Monitor the simulation until it completes or fails.",
            tool="supervise_simulation",
            tool_args=dict(job_ref),
        ),
        WorkflowStep(
            role="simulation",
            instruction="Collect raw solver outputs into the dataset manifest.",
            tool="fetch_simulation_outputs",
            tool_args=dict(job_ref),
        ),
        WorkflowStep(
            role="simulation",
            instruction="Generate canonical HIT analysis products from the fetched manifest.",
            tool="postprocess_simulation_outputs",
            tool_args=dict(job_ref),
        ),
        WorkflowStep(
            role="steward",
            instruction=(
                "Load the post-processed simulation manifest into the app session "
                "so manual analysis pages can use the new data."
            ),
            tool="load_dataset_manifest",
            tool_args=dict(job_ref),
        ),
    ]


def _load_plan(intent: RequestIntent) -> WorkflowPlan:
    job_id = intent.job_ref or ACTIVE_SIMULATION_JOB_ID
    steps = [
        WorkflowStep(
            role="steward",
            instruction=(
                f"Load the already-saved {_backend_label(intent)} dataset for job "
                f"'{job_id}' into the session. "
                "Do NOT build, compile, start, or supervise a new simulation."
            ),
            tool="load_dataset_manifest",
            tool_args={"job_id": job_id},
        )
    ]
    steps.extend(_analysis_steps(intent.analyses))
    rationale = f"schema:load job={job_id}"
    if intent.analyses:
        rationale += " + " + ", ".join(a.analysis_id for a in intent.analyses)
    return WorkflowPlan(steps=steps, rationale=rationale)


def _compile_plan(intent: RequestIntent) -> WorkflowPlan:
    build_args = dict(intent.case_params or {})
    build_args.setdefault("backend", intent.backend)
    label = _backend_label(intent)
    job_ref = {"job_id": intent.job_ref or ACTIVE_SIMULATION_JOB_ID}
    return WorkflowPlan(
        steps=[
            WorkflowStep(
                role="simulation",
                instruction=(
                    f"Build the {label} case '{build_args.get('name')}' from the parsed user parameters."
                ),
                tool="build_simulation_case",
                tool_args=build_args,
            ),
            WorkflowStep(
                role="simulation",
                instruction=f"Compile the prepared {label} solver case.",
                tool="compile_simulation",
                tool_args=dict(job_ref),
            ),
        ],
        rationale="schema:compile (stop; no run)",
    )


def _run_plan(intent: RequestIntent) -> WorkflowPlan:
    build_args = dict(intent.case_params or {})
    build_args.setdefault("backend", intent.backend)
    label = _backend_label(intent)
    job_id = intent.job_ref or ACTIVE_SIMULATION_JOB_ID
    steps = _lifecycle_steps(build_args, label, job_id)
    steps.extend(_analysis_steps(intent.analyses))
    rationale = "schema:run build→compile→start→supervise→fetch→postprocess→load"
    if intent.analyses:
        rationale += " + " + ", ".join(a.analysis_id for a in intent.analyses)
    return WorkflowPlan(steps=steps, rationale=rationale)


def _status_plan(intent: RequestIntent) -> WorkflowPlan:
    job_id = intent.job_ref or ACTIVE_SIMULATION_JOB_ID
    text = (intent.raw_text or "").lower()
    tool = "cancel_simulation" if "cancel" in text else "check_simulation_status"
    return WorkflowPlan(
        steps=[
            WorkflowStep(
                role="simulation",
                instruction=f"Report status for simulation job '{job_id}'.",
                tool=tool,
                tool_args={"job_id": job_id},
            )
        ],
        rationale=f"schema:status job={job_id}",
    )


def _analyze_plan(intent: RequestIntent) -> WorkflowPlan:
    job_id = intent.job_ref or ACTIVE_SIMULATION_JOB_ID
    steps = [
        WorkflowStep(
            role="steward",
            instruction=(
                f"Ensure {_backend_label(intent)} job '{job_id}' products are loaded before analysis."
            ),
            tool="load_dataset_manifest",
            tool_args={"job_id": job_id},
        )
    ]
    steps.extend(_analysis_steps(intent.analyses))
    rationale = f"schema:analyze job={job_id}"
    if intent.analyses:
        rationale += " + " + ", ".join(a.analysis_id for a in intent.analyses)
    return WorkflowPlan(steps=steps, rationale=rationale)


def _inquire_plan(intent: RequestIntent) -> WorkflowPlan:
    return WorkflowPlan(
        steps=[
            WorkflowStep(
                role="steward",
                instruction=(
                    "Answer where simulation outputs are stored. Do NOT build, compile, "
                    "run, fetch, or postprocess a simulation. Use session context "
                    "(simulation_job_id, manifest_path, data_directory, dataset_manifest) and "
                    "list_directory/read_file on simulations/<job_id>/ (raw/, processed/, "
                    "manifest.json) as needed.\n"
                    f"Question: {intent.raw_text}"
                ),
            )
        ],
        rationale="schema:inquire storage",
    )


def _research_plan(intent: RequestIntent) -> WorkflowPlan:
    """Web-backed research: search → browse → answer with citations."""
    question = (intent.raw_text or "").strip()
    return WorkflowPlan(
        steps=[
            WorkflowStep(
                role="analyst",
                instruction=(
                    "Solve this with live web learning. Do NOT start simulations.\n"
                    "1) Call web_search with a focused query derived from the question.\n"
                    "2) For scientific/theory topics also call search_research_papers (arXiv).\n"
                    "3) Call browse_web on the 1–3 most relevant URLs (docs, forum, papers).\n"
                    "4) Answer using what you learned; cite URLs. If session analysis products "
                    "are loaded, relate findings to them; otherwise stay literature/docs-based.\n"
                    f"Question: {question}"
                ),
            )
        ],
        rationale="schema:research web",
    )


def _research_then_run_plan(intent: RequestIntent) -> WorkflowPlan:
    """Cursor-like: web research → build from learned params → compile → run → collect."""
    import json

    hints = {k: v for k, v in dict(intent.case_params or {}).items() if k != "stop_after"}
    stop_after = str((intent.case_params or {}).get("stop_after") or "").strip().lower()
    question = (intent.raw_text or "").strip()
    job_ref = {"job_id": intent.job_ref or ACTIVE_SIMULATION_JOB_ID}

    label = _backend_label(intent)
    research = WorkflowStep(
        role="analyst",
        instruction=(
            "WEB RESEARCH FIRST — do not build or start a simulation yet.\n"
            f"Goal: find recommended {label} parameters for this request "
            "and how to validate the run.\n"
            f"User request: {question}\n"
            f"Soft hints from the user text (incomplete; do not treat as final): "
            f"{json.dumps(hints, default=str)}\n"
            "Procedure:\n"
            f"1) web_search for {label} recommended case parameters and validation checks.\n"
            "2) search_research_papers on arXiv for relevant CFD/turbulence benchmarks.\n"
            "3) browse_web the best 1–3 docs/papers/forum posts.\n"
            "4) Recommend a concrete parameter set the case builder can consume "
            "(resolution, scheme, viscosity/Re, max_steps, outputs, validation diagnostics).\n"
            "End with a JSON object under the heading CASE_PARAMS that build_simulation_case "
            "can consume. Cite URLs. Do not call simulation lifecycle tools."
        ),
    )
    build = WorkflowStep(
        role="simulation",
        instruction=(
            "Apply the researched CASE_PARAMS from Prior step evidence.\n"
            "Call build_simulation_case with the full recommended case (prefer researched "
            f"values over calibrated defaults). Keep backend={label}. "
            "Respect explicit user overrides if any appear in the user request.\n"
            f"User request: {question}\n"
            f"Hints: {json.dumps(hints, default=str)}\n"
            "Do not compile or start yet — later workflow steps will."
        ),
    )
    steps: List[WorkflowStep] = [research, build]
    steps.append(
        WorkflowStep(
            role="simulation",
            instruction="Compile the prepared OpenLB solver case.",
            tool="compile_simulation",
            tool_args=dict(job_ref),
        )
    )
    if stop_after != "compile":
        steps.extend(
            [
                WorkflowStep(
                    role="simulation",
                    instruction="Launch the OpenLB simulation job.",
                    tool="start_simulation",
                    tool_args=dict(job_ref),
                ),
                WorkflowStep(
                    role="simulation",
                    instruction="Monitor the simulation until it completes or fails.",
                    tool="supervise_simulation",
                    tool_args=dict(job_ref),
                ),
                WorkflowStep(
                    role="simulation",
                    instruction="Collect raw solver outputs into the dataset manifest.",
                    tool="fetch_simulation_outputs",
                    tool_args=dict(job_ref),
                ),
                WorkflowStep(
                    role="simulation",
                    instruction="Generate canonical HIT analysis products from the fetched manifest.",
                    tool="postprocess_simulation_outputs",
                    tool_args=dict(job_ref),
                ),
                WorkflowStep(
                    role="steward",
                    instruction=(
                        "Load the post-processed simulation manifest into the app session "
                        "so manual analysis pages can use the new data."
                    ),
                    tool="load_dataset_manifest",
                    tool_args=dict(job_ref),
                ),
            ]
        )
        if hints.get("validate") and not intent.analyses:
            steps.append(
                WorkflowStep(
                    role="analyst",
                    instruction=(
                        "Validate the DHIT/HIT run using loaded products: call "
                        "compute_overview_validation (and compute_spectra if useful). "
                        "Summarize whether the run looks physically reasonable."
                    ),
                )
            )
        steps.extend(_analysis_steps(intent.analyses))

    rationale = "schema:research_then_run web→build→compile"
    if stop_after != "compile":
        rationale += "→start→supervise→fetch→postprocess→load"
    return WorkflowPlan(steps=steps, rationale=rationale)


def plan_from_intent(intent: Optional[RequestIntent]) -> Optional[WorkflowPlan]:
    """Pure action→graph mapping."""
    if intent is None:
        return None
    if intent.action == "load":
        return _load_plan(intent)
    if intent.action == "compile":
        return _compile_plan(intent)
    if intent.action == "run":
        return _run_plan(intent)
    if intent.action == "status":
        return _status_plan(intent)
    if intent.action == "analyze":
        return _analyze_plan(intent)
    if intent.action == "inquire":
        return _inquire_plan(intent)
    if intent.action == "research":
        return _research_plan(intent)
    if intent.action == "research_then_run":
        return _research_then_run_plan(intent)
    return None


__all__ = ["plan_from_intent"]
