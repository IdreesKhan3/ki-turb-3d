"""Services for the engineering_workflow subgraph."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.knowledge.capability_loader import load_capability_context
from agents.knowledge.lesson_store import format_lessons, record_lesson, retrieve_lessons
from agents.tools import execute_tool

from .engineering_intent import extract_named_paths, is_simple_file_edit_request
from .models import EngineeringDiscovery, EngineeringPlan, EngineeringStep, WorkflowPlan, WorkflowStep
from .role_agents import RoleAgentFactory
from .settings import LangGraphSettings
from .state import KITurbState

MAX_REPAIR_ATTEMPTS = 2
_PLATFORM_CAPS = frozenset({"app_pages", "plotting", "solvers", "viz_external", "hpc"})


def _reraise_graph_control(exc: BaseException) -> None:
    """Let LangGraph HITL / interrupt control-flow escape nested agent.invoke()."""
    from langgraph.errors import GraphBubbleUp

    if isinstance(exc, GraphBubbleUp):
        raise exc


def _event(stage: str, status: str, message: str = "", **data) -> list[dict]:
    return [{"stage": stage, "status": status, "message": message, **data}]


def _format_plan(plan: EngineeringPlan) -> str:
    lines = [
        f"## Engineering plan: {plan.goal}",
        f"Capability: {plan.capability or ', '.join(plan.capabilities) or 'general'}",
        "",
        "### Discoveries",
    ]
    for item in plan.discoveries:
        lines.append(f"- `{item.file}` — {item.role}")
    lines.append("")
    lines.append("### Create")
    lines.extend([f"- `{p}`" for p in plan.create] or ["- (none)"])
    lines.append("")
    lines.append("### Modify")
    lines.extend([f"- `{p}`" for p in plan.modify] or ["- (none)"])
    lines.append("")
    lines.append("### Do not touch")
    lines.extend([f"- `{p}`" for p in plan.do_not_touch] or ["- (none)"])
    lines.append("")
    lines.append("### Verify")
    lines.extend([f"- `{c}`" for c in plan.verify] or ["- (none)"])
    lines.append("")
    lines.append("### Steps")
    for i, step in enumerate(plan.steps, 1):
        lines.append(f"{i}. **{step.title}** (`{step.id}`)")
        lines.append(f"   {step.instruction}")
        if step.create:
            lines.append("   create: " + ", ".join(f"`{p}`" for p in step.create))
        if step.modify:
            lines.append("   modify: " + ", ".join(f"`{p}`" for p in step.modify))
        if step.verify:
            lines.append("   verify: " + ", ".join(f"`{c}`" for c in step.verify))
    if plan.rationale:
        lines.extend(["", f"_Rationale:_ {plan.rationale}"])
    if plan.plan_only:
        lines.extend(["", "Mode: **plan only** (no edits until you approve and ask to continue)."])
    return "\n".join(lines)


def _repo_discoveries(project_root: Path, capabilities: List[str]) -> List[EngineeringDiscovery]:
    seeds = [
        ("agents/page_schema.py", "page workflow schema"),
        ("agents/intent_detection.py", "NL intent routing"),
        ("agents/runtime/tool_registry.py", "tool permissions"),
        ("integrations/base.py", "CFDBackend contract"),
        ("agents/langgraph/router.py", "request planner"),
        ("agents/langgraph/app_graph.py", "root LangGraph"),
    ]
    cap_extra = {
        "app_pages": [("pages/00_Autonomous_Lab.py", "Autonomous Lab entry")],
        "plotting": [("agents/tools/physics/__init__.py", "physics/plot tool package")],
        "solvers": [("integrations/palabos_backend.py", "Palabos adapter"), ("integrations/ansys_backend.py", "Ansys adapter")],
        "viz_external": [("postprocessing/writers.py", "output writers")],
        "hpc": [("integrations/remote_runner.py", "remote/HPC runner hook")],
    }
    found: List[EngineeringDiscovery] = []
    for rel, role in seeds:
        if (project_root / rel).exists():
            found.append(EngineeringDiscovery(file=rel, role=role))
    for cap in capabilities:
        for rel, role in cap_extra.get(cap, []):
            if (project_root / rel).exists():
                found.append(EngineeringDiscovery(file=rel, role=role))
    # Deduplicate
    seen = set()
    out = []
    for item in found:
        if item.file in seen:
            continue
        seen.add(item.file)
        out.append(item)
    return out


def _file_edit_engineering_plan(
    request: str,
    paths: List[str],
    discoveries: List[EngineeringDiscovery],
    plan_only: bool,
) -> EngineeringPlan:
    """Targeted plan for concrete paths — never expands into page_schema/registry."""
    modify = list(paths)
    verify = [f"python -m compileall {paths[0]} -q"] if paths[0].endswith(".py") else [
        f'python -c "from pathlib import Path; p=Path({paths[0]!r}); '
        f'print(p.exists(), p.stat().st_size if p.exists() else 0)"'
    ]
    path_list = ", ".join(f"`{p}`" for p in paths)
    return EngineeringPlan(
        goal=request.strip() or f"Edit {path_list}",
        capability="file_edit",
        capabilities=["file_edit"],
        discoveries=discoveries,
        create=[],
        modify=modify,
        do_not_touch=[
            "agents/page_schema.py",
            "agents/intent_detection.py",
            "agents/runtime/tool_registry.py",
            "agents/langgraph/hit_graph.py",
            "cfd_solvers/",
            "simulations/",
        ],
        verify=verify,
        steps=[
            EngineeringStep(
                id="edit_named_files",
                title="Edit the named file(s)",
                instruction=(
                    f"{request}\n\n"
                    f"Read then edit ONLY: {path_list}. "
                    "Use read_file before write_file/modify_file. "
                    "Do not touch page_schema, registries, or create new Streamlit pages."
                ),
                modify=modify,
                verify=verify,
            )
        ],
        plan_only=plan_only,
        rationale="Named-path edit plan (no platform capability pack).",
    )


def build_deterministic_plan(
    request: str,
    *,
    capabilities: List[str],
    discoveries: List[EngineeringDiscovery],
    plan_only: bool,
) -> EngineeringPlan:
    lower = (request or "").lower()
    named_paths = extract_named_paths(request)
    platform_caps = [c for c in capabilities if c in _PLATFORM_CAPS]

    # Named user/script paths win over keyword "plot"/"page" capability noise.
    if named_paths and (
        is_simple_file_edit_request(request)
        or not platform_caps
        or all(p.lower().startswith("examples/") for p in named_paths)
    ):
        return _file_edit_engineering_plan(request, named_paths, discoveries, plan_only)

    primary = platform_caps[0] if platform_caps else ""
    create: List[str] = []
    modify: List[str] = []
    verify: List[str] = [
        "pytest tests/agents/test_tool_registry_permissions.py -q",
        "pytest tests/test_engineering_routing.py -q",
    ]
    do_not_touch = [
        "agents/langgraph/hit_graph.py",
        "cfd_solvers/",
        "simulations/",
    ]
    steps: List[EngineeringStep] = []

    # Only use packed templates when the capability (or clear platform phrasing) matches.
    wants_page = primary == "app_pages" or bool(
        re.search(r"\b(?:streamlit\s+)?pages?\b|page_schema", lower)
    )
    wants_plot_tool = primary == "plotting" and bool(
        re.search(r"\b(?:plot(?:ting)?\s+tool|register(?:ed)?\s+plot|visualizer\s+tool)\b", lower)
    )
    wants_solver = primary == "solvers" or any(
        w in lower for w in ("palabos", "openfoam", "ansys", "cfd backend", "solver adapter")
    )
    wants_vtk = primary == "viz_external" or "vtk" in lower or "paraview" in lower
    wants_hpc = primary == "hpc" or bool(re.search(r"\b(?:hpc|slurm|remote runner)\b", lower))

    if wants_page and not named_paths:
        modify = [
            "agents/page_schema.py",
            "agents/intent_detection.py",
            "agents/runtime/tool_registry.py",
        ]
        create = ["pages/NN_New_Page.py", "tests/test_new_page_schema.py"]
        steps = [
            EngineeringStep(
                id="schema",
                title="Extend page schema and intent routing",
                instruction="Add PAGE_SCHEMA entry and intent patterns for the new/changed page.",
                modify=["agents/page_schema.py", "agents/intent_detection.py"],
                verify=["pytest tests/test_engineering_routing.py -q"],
            ),
            EngineeringStep(
                id="page_module",
                title="Add Streamlit page module",
                instruction="Create the Streamlit page module under pages/ wired to existing analysis patterns.",
                create=["pages/NN_New_Page.py"],
                verify=["python -m compileall pages -q"],
            ),
            EngineeringStep(
                id="tests",
                title="Add focused tests",
                instruction="Add schema/routing tests for the page change.",
                create=["tests/test_new_page_schema.py"],
                modify=["agents/runtime/tool_registry.py"],
                verify=["pytest tests/agents/test_tool_registry_permissions.py -q"],
            ),
        ]
        primary = primary or "app_pages"
    elif wants_plot_tool and not named_paths:
        modify = [
            "agents/tools/physics/__init__.py",
            "agents/runtime/tool_registry.py",
            "agents/page_schema.py",
            "agents/intent_detection.py",
        ]
        create = ["agents/tools/physics/new_plot.py", "tests/test_new_plot_tool.py"]
        steps = [
            EngineeringStep(
                id="plot_tool",
                title="Implement plot tool",
                instruction="Add a registered plot_* tool and wire visualizer permissions.",
                create=["agents/tools/physics/new_plot.py"],
                modify=["agents/tools/physics/__init__.py", "agents/runtime/tool_registry.py"],
                verify=["pytest tests/agents/test_tool_registry_permissions.py -q"],
            ),
            EngineeringStep(
                id="wire_page",
                title="Wire page schema and intents",
                instruction="Attach the new plot tool to PAGE_SCHEMA and intent detection.",
                modify=["agents/page_schema.py", "agents/intent_detection.py"],
                verify=["pytest tests/test_engineering_routing.py -q"],
            ),
        ]
        primary = primary or "plotting"
    elif wants_solver and not (named_paths and all(p.startswith("examples/") for p in named_paths)):
        modify = ["integrations/base.py", "integrations/palabos_backend.py"]
        create = ["tests/integrations/test_backend_contract_stub.py"]
        verify = ["pytest tests/integrations -q"]
        steps = [
            EngineeringStep(
                id="adapter",
                title="Extend solver adapter toward CFDBackend contract",
                instruction="Inspect CFDBackend and extend the target adapter methods with tests.",
                modify=["integrations/palabos_backend.py", "integrations/base.py"],
                create=["tests/integrations/test_backend_contract_stub.py"],
                verify=["pytest tests/integrations -q"],
            ),
        ]
        primary = primary or "solvers"
    elif wants_vtk:
        modify = ["postprocessing/writers.py"]
        create = ["agents/tools/export_vtk.py", "tests/test_vtk_export_stub.py"]
        steps = [
            EngineeringStep(
                id="vtk_export",
                title="Add VTK export stub from analysis products",
                instruction="Add an export helper and a focused test; do not embed ParaView server control yet.",
                create=["agents/tools/export_vtk.py", "tests/test_vtk_export_stub.py"],
                modify=["postprocessing/writers.py"],
                verify=["pytest tests/test_vtk_export_stub.py -q"],
            ),
        ]
        primary = primary or "viz_external"
    elif wants_hpc:
        modify = ["integrations/remote_runner.py", "schemas/simulation_job.py"]
        create = ["tests/integrations/test_remote_runner_stub.py"]
        steps = [
            EngineeringStep(
                id="remote_runner",
                title="Extend remote/HPC runner hooks",
                instruction="Extend remote_runner abstractions for GPU/HPC job submission metadata.",
                modify=["integrations/remote_runner.py", "schemas/simulation_job.py"],
                create=["tests/integrations/test_remote_runner_stub.py"],
                verify=["pytest tests/integrations/test_remote_runner_stub.py -q"],
            ),
        ]
        primary = primary or "hpc"
    elif named_paths:
        return _file_edit_engineering_plan(request, named_paths, discoveries, plan_only)
    else:
        # No matched platform pack and no paths — explore first; do NOT invent a page.
        primary = primary or "general"
        verify = ["python -m compileall agents -q"]
        steps = [
            EngineeringStep(
                id="explore_implement",
                title="Explore then implement the requested change",
                instruction=(
                    f"{request}\n\n"
                    "Search the repo for the real target files first. "
                    "Implement the smallest change. Do not create a new Streamlit page "
                    "or touch page_schema unless the user explicitly asked for a page."
                ),
                modify=[],
                create=[],
                verify=verify,
            ),
        ]

    return EngineeringPlan(
        goal=request.strip() or "Platform engineering task",
        capability=primary,
        capabilities=list(capabilities) or ([primary] if primary else []),
        discoveries=discoveries,
        create=create,
        modify=modify,
        do_not_touch=do_not_touch,
        verify=verify,
        steps=steps,
        plan_only=plan_only,
        rationale="Deterministic capability-pack plan (refine with search evidence before editing).",
    )


@dataclass
class EngineeringGraphServices:
    settings: LangGraphSettings
    project_root: Path
    session_context: Dict[str, Any]
    role_factory: Optional[RoleAgentFactory] = None

    @classmethod
    def default(
        cls,
        settings: LangGraphSettings,
        project_root: str | Path,
        session_context: Dict[str, Any],
        role_factory: Optional[RoleAgentFactory] = None,
    ):
        return cls(settings, Path(project_root).resolve(), session_context, role_factory)

    def _eng_meta(self, state: KITurbState) -> Dict[str, Any]:
        meta = dict(state.get("metadata") or {})
        return dict(meta.get("engineering") or {})

    def discover(self, state: KITurbState) -> Dict[str, Any]:
        request = state.get("user_request") or ""
        eng = self._eng_meta(state)
        if eng.get("continue_execution") and self.session_context.get("engineering_plan"):
            plan = EngineeringPlan.model_validate(self.session_context["engineering_plan"])
            return {
                "engineering_plan": plan.model_dump(mode="json"),
                "engineering_capability": plan.capability,
                "engineering_context": self.session_context.get("engineering_context") or "",
                "status": "planned",
                "events": _event("discover", "ok", "reusing approved engineering plan"),
            }

        lessons = retrieve_lessons(self.project_root, request, k=5)
        cap = load_capability_context(
            request,
            self.project_root,
            lessons_text=format_lessons(lessons),
        )
        discoveries = _repo_discoveries(self.project_root, cap["capabilities"])
        context = cap["context"]
        self.session_context["engineering_capability"] = cap["primary_capability"]
        self.session_context["engineering_context"] = context
        self.session_context["engineering_discoveries"] = [d.model_dump(mode="json") for d in discoveries]
        return {
            "engineering_capability": cap["primary_capability"],
            "engineering_context": context,
            "engineering_discoveries": [d.model_dump(mode="json") for d in discoveries],
            "status": "running",
            "events": _event(
                "discover",
                "ok",
                f"capabilities={','.join(cap['capabilities']) or 'none'}; discoveries={len(discoveries)}",
            ),
            "metadata": {
                **dict(state.get("metadata") or {}),
                "engineering": {
                    **eng,
                    "capabilities": cap["capabilities"],
                    "lessons": [lesson.to_dict() for lesson in lessons],
                },
            },
        }

    def draft_plan(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        eng = self._eng_meta(state)
        if eng.get("continue_execution") and state.get("engineering_plan"):
            plan = EngineeringPlan.model_validate(state["engineering_plan"])
            return {
                "engineering_plan": plan.model_dump(mode="json"),
                "status": "planned",
                "final_text": _format_plan(plan),
                "events": _event("draft_plan", "ok", "using existing plan"),
            }

        request = state.get("user_request") or ""
        capabilities = list((eng.get("capabilities") or []))
        if not capabilities and state.get("engineering_capability"):
            capabilities = [str(state.get("engineering_capability"))]
        discoveries = [
            EngineeringDiscovery.model_validate(item)
            for item in (state.get("engineering_discoveries") or [])
            if isinstance(item, dict)
        ]
        plan_only = bool(eng.get("plan_only"))
        plan = build_deterministic_plan(
            request,
            capabilities=capabilities,
            discoveries=discoveries,
            plan_only=plan_only,
        )

        # Optional LLM refinement when a role factory is available.
        if self.role_factory is not None and self.settings.use_llm_planner:
            try:
                from .structured_output import invoke_structured

                payload = (
                    f"User request:\n{request}\n\n"
                    f"Capability context:\n{state.get('engineering_context') or ''}\n\n"
                    f"Seed plan JSON:\n{json.dumps(plan.model_dump(mode='json'), indent=2)}\n\n"
                    "Refine into a concrete EngineeringPlan. Prefer real repo paths from discoveries. "
                    f"plan_only={plan_only}."
                )
                refined = invoke_structured(
                    self.role_factory.model,
                    self.role_factory.model_name,
                    EngineeringPlan,
                    (
                        "You produce KI-TURB EngineeringPlan JSON only. "
                        "Keep steps small with verify commands. Do not invent files outside the repo layout."
                    ),
                    payload,
                    agent_name="kiturb_engineering_planner",
                )
                if isinstance(refined, EngineeringPlan) and refined.steps:
                    plan = refined
                    plan.plan_only = plan_only or plan.plan_only
            except Exception:
                pass

        self.session_context["engineering_plan"] = plan.model_dump(mode="json")
        self.session_context["engineering_step_index"] = 0
        text = _format_plan(plan)
        return {
            "engineering_plan": plan.model_dump(mode="json"),
            "engineering_step_index": 0,
            "status": "planned",
            "final_text": text,
            "events": _event("draft_plan", "ok", f"{len(plan.steps)} steps drafted"),
        }

    def approve_plan(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        plan = EngineeringPlan.model_validate(state.get("engineering_plan") or {})
        eng = self._eng_meta(state)

        # Plan-only: present plan without interrupt when approval disabled or plan_only.
        if plan.plan_only and not eng.get("continue_execution"):
            self.session_context["engineering_plan"] = plan.model_dump(mode="json")
            self.session_context["engineering_plan_approved"] = False
            return {
                "approved": False,
                "status": "completed",
                "final_text": (state.get("final_text") or _format_plan(plan))
                + "\n\nPlan only — say **approve and execute** or **do step 1** to proceed.",
                "events": _event("approval", "plan_only", "plan presented without execution"),
            }

        if eng.get("continue_execution") and self.session_context.get("engineering_plan_approved"):
            return {
                "approved": True,
                "status": "approved",
                "events": _event("approval", "ok", "previously approved plan"),
            }

        if not state.get("require_approval", self.settings.require_execution_approval):
            self.session_context["engineering_plan_approved"] = True
            return {
                "approved": True,
                "status": "approved",
                "events": _event("approval", "skipped", "approval disabled"),
            }

        from langgraph.types import interrupt

        answer = interrupt({
            "kind": "engineering_plan_approval",
            "message": f"Approve engineering plan for: {plan.goal}?",
            "engineering_plan": plan.model_dump(mode="json"),
            "plan_text": state.get("final_text") or _format_plan(plan),
        })
        approved = bool(answer if not isinstance(answer, dict) else answer.get("approved"))
        self.session_context["engineering_plan_approved"] = approved
        if not approved:
            # Fall back to free-form steward instead of ending on "rejected".
            request = state.get("user_request") or plan.goal
            fallback = WorkflowPlan(
                kind="agent_workflow",
                steps=[
                    WorkflowStep(
                        role="steward",
                        instruction=(
                            f"{request}\n\n"
                            "The previous engineering plan was rejected. "
                            "Fulfill the user request directly: read the target file(s), "
                            "apply write_file/modify_file, then self-verify. "
                            "Do not reopen engineering_workflow or invent page/registry changes."
                        ),
                    )
                ],
                rationale="Fallback to steward after engineering plan rejection",
            )
            return {
                "approved": False,
                "status": "cancelled",
                "plan": fallback.model_dump(mode="json"),
                "task_index": 0,
                "final_text": "",
                "metadata": {
                    **dict(state.get("metadata") or {}),
                    "engineering_fallback": True,
                    "engineering": {**eng, "fallback_steward": True},
                },
                "events": _event(
                    "approval",
                    "rejected_fallback",
                    "plan rejected — falling back to steward",
                ),
            }
        return {
            "approved": True,
            "status": "approved",
            "events": _event("approval", "approved", "user decision"),
        }

    def execute_step(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors") or state.get("status") in {"cancelled", "rejected", "completed"}:
            return {}
        if not state.get("approved") and not self.session_context.get("engineering_plan_approved"):
            return {"status": "cancelled", "errors": ["engineering plan not approved"]}

        plan = EngineeringPlan.model_validate(state.get("engineering_plan") or {})
        eng = self._eng_meta(state)
        index = int(
            eng.get("step_index")
            if eng.get("step_index") is not None
            else state.get("engineering_step_index")
            or self.session_context.get("engineering_step_index")
            or 0
        )
        # When continuing a single explicit step, only run that step.
        single_step = eng.get("step_index") is not None
        if index >= len(plan.steps):
            return {
                "status": "completed",
                "final_text": (state.get("final_text") or "") + "\n\nAll engineering steps complete.",
                "events": _event("execute", "ok", "no remaining steps"),
            }

        step = plan.steps[index]
        instruction = (
            f"Execute engineering step {index + 1}/{len(plan.steps)}: {step.title}\n"
            f"{step.instruction}\n"
            f"Create: {step.create}\nModify: {step.modify}\n"
            f"Do not touch: {plan.do_not_touch}\n"
            "Use authorized engineer tools. Make the smallest change that satisfies this step."
        )

        result_text = ""
        if self.role_factory is not None:
            try:
                agent = self.role_factory.create_role_agent("engineer")
                result = agent.invoke({"messages": [{"role": "user", "content": instruction}]})
                messages = result.get("messages") if isinstance(result, dict) else None
                if messages:
                    last = messages[-1]
                    result_text = str(getattr(last, "content", last) or "")
                else:
                    result_text = str(result)
            except Exception as exc:
                _reraise_graph_control(exc)
                result_text = f"Engineer agent step error: {exc}"
        else:
            result_text = (
                f"Dry-run execute step {step.id}: would create {step.create}, "
                f"modify {step.modify}."
            )

        self.session_context["engineering_step_index"] = index
        return {
            "engineering_step_index": index,
            "engineering_last_step_result": result_text,
            "status": "running",
            "final_text": f"### Step {index + 1}: {step.title}\n{result_text}",
            "task_results": [{"role": "engineer", "text": result_text, "step_id": step.id}],
            "events": _event("execute", "ok", f"step {index + 1}: {step.title}"),
            "metadata": {
                **dict(state.get("metadata") or {}),
                "engineering": {**eng, "single_step": single_step, "step_index": index},
            },
        }

    def verify_step(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors") or state.get("status") in {"cancelled", "rejected", "completed"}:
            return {}
        plan = EngineeringPlan.model_validate(state.get("engineering_plan") or {})
        index = int(state.get("engineering_step_index") or 0)
        if index >= len(plan.steps):
            return {"status": "completed", "events": _event("verify", "ok", "nothing to verify")}
        step = plan.steps[index]
        commands = list(step.verify or plan.verify or [])
        if not commands:
            return {
                "engineering_verify_ok": True,
                "events": _event("verify", "skipped", "no verify commands"),
            }

        outputs: List[str] = []
        ok = True
        for command in commands:
            # cat/head/… are normalized inside run_verify_command to a file read.
            raw = execute_tool(
                "run_verify_command",
                {"command": str(command or "").strip()},
                self.project_root,
                session_context={
                    **self.session_context,
                    "tool_confirmation_approved": True,
                },
                allowed_tool_names={"run_verify_command"},
            )
            text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            outputs.append(f"$ {command}\n{text}")
            if "status: failed" in text.lower() or text.lower().startswith("error:"):
                ok = False

        evidence = "\n\n".join(outputs)
        if ok:
            return {
                "engineering_verify_ok": True,
                "engineering_repair_attempts": 0,
                "final_text": (state.get("final_text") or "") + "\n\n### Verify\n" + evidence,
                "events": _event("verify", "ok", f"step {index + 1} verified"),
            }
        attempts = int(state.get("engineering_repair_attempts") or 0)
        return {
            "engineering_verify_ok": False,
            "engineering_repair_attempts": attempts,
            "final_text": (state.get("final_text") or "") + "\n\n### Verify failed\n" + evidence,
            "events": _event("verify", "failed", f"step {index + 1} verify failed"),
        }

    def repair_step(self, state: KITurbState) -> Dict[str, Any]:
        attempts = int(state.get("engineering_repair_attempts") or 0) + 1
        if attempts > MAX_REPAIR_ATTEMPTS:
            plan = EngineeringPlan.model_validate(state.get("engineering_plan") or {})
            index = int(state.get("engineering_step_index") or 0)
            step = plan.steps[index] if index < len(plan.steps) else None
            record_lesson(
                self.project_root,
                task=plan.goal,
                capability=plan.capability,
                symptoms=state.get("final_text") or "verify failed",
                fix="repair budget exhausted — needs human intervention",
                files=list((step.create if step else []) + (step.modify if step else [])),
                verify=list(step.verify if step else plan.verify),
                reuse_when=f"verify failure on {step.id if step else 'engineering step'}",
                outcome="failed",
            )
            return {
                "status": "failed",
                "errors": ["engineering repair budget exhausted"],
                "engineering_repair_attempts": attempts,
                "events": _event("repair", "failed", "max repairs exceeded"),
            }

        plan = EngineeringPlan.model_validate(state.get("engineering_plan") or {})
        index = int(state.get("engineering_step_index") or 0)
        step = plan.steps[index]
        instruction = (
            f"Repair engineering step {index + 1}: {step.title}\n"
            f"Previous verify failure evidence:\n{state.get('final_text') or ''}\n"
            "Fix the smallest issue and stop."
        )
        result_text = ""
        if self.role_factory is not None:
            try:
                agent = self.role_factory.create_role_agent("engineer")
                result = agent.invoke({"messages": [{"role": "user", "content": instruction}]})
                messages = result.get("messages") if isinstance(result, dict) else None
                if messages:
                    result_text = str(getattr(messages[-1], "content", messages[-1]) or "")
                else:
                    result_text = str(result)
            except Exception as exc:
                _reraise_graph_control(exc)
                result_text = f"Repair error: {exc}"
        else:
            result_text = "Dry-run repair (no role factory)."

        return {
            "engineering_repair_attempts": attempts,
            "engineering_last_step_result": result_text,
            "status": "running",
            "final_text": (state.get("final_text") or "") + f"\n\n### Repair attempt {attempts}\n{result_text}",
            "events": _event("repair", "ok", f"repair attempt {attempts}"),
        }

    def advance_or_finish(self, state: KITurbState) -> Dict[str, Any]:
        if state.get("errors") or state.get("status") in {"failed", "cancelled", "rejected"}:
            return {}
        eng = self._eng_meta(state)
        plan = EngineeringPlan.model_validate(state.get("engineering_plan") or {})
        index = int(state.get("engineering_step_index") or 0)
        single_step = bool(eng.get("single_step") or eng.get("step_index") is not None)

        next_index = index + 1
        self.session_context["engineering_step_index"] = next_index

        if single_step or next_index >= len(plan.steps):
            record_lesson(
                self.project_root,
                task=plan.goal,
                capability=plan.capability,
                symptoms="",
                fix="step verified" if state.get("engineering_verify_ok") else "completed",
                files=list(plan.create) + list(plan.modify),
                verify=list(plan.verify),
                reuse_when=plan.goal,
                outcome="success",
            )
            remaining = max(0, len(plan.steps) - next_index)
            suffix = (
                f"\n\nStep {index + 1} complete."
                + (f" {remaining} step(s) remaining — say **continue**." if remaining else " All steps done.")
            )
            return {
                "engineering_step_index": next_index,
                "status": "completed",
                "final_text": (state.get("final_text") or "") + suffix,
                "events": _event("advance", "ok", "engineering wave finished"),
            }

        return {
            "engineering_step_index": next_index,
            "engineering_repair_attempts": 0,
            "status": "running",
            "events": _event("advance", "ok", f"advance to step {next_index + 1}"),
            "metadata": {
                **dict(state.get("metadata") or {}),
                "engineering": {**eng, "step_index": next_index, "single_step": False},
            },
        }

    def finalize(self, state: KITurbState) -> Dict[str, Any]:
        meta = dict(state.get("metadata") or {})
        if meta.get("engineering_fallback") and state.get("plan"):
            # Preserve steward fallback plan for the parent graph.
            return {
                "status": "planned",
                "plan": state["plan"],
                "task_index": 0,
                "final_text": state.get("final_text") or "",
                "metadata": meta,
                "events": _event("finalize", "fallback", "handing off to steward"),
            }

        if state.get("errors"):
            text = state.get("final_text") or ("Errors: " + "; ".join(state["errors"]))
            plan_data = state.get("engineering_plan")
            if plan_data:
                try:
                    plan = EngineeringPlan.model_validate(plan_data)
                    record_lesson(
                        self.project_root,
                        task=plan.goal,
                        capability=plan.capability,
                        symptoms=text,
                        fix="",
                        files=list(plan.create) + list(plan.modify),
                        verify=list(plan.verify),
                        reuse_when=plan.goal,
                        outcome="failed",
                    )
                except Exception:
                    pass
            return {"status": "failed", "final_text": text, "events": _event("finalize", "failed", text)}

        text = state.get("final_text") or "Engineering workflow completed."
        if state.get("engineering_plan"):
            self.session_context["engineering_plan"] = state["engineering_plan"]
        return {
            "status": state.get("status") if state.get("status") in {"completed", "cancelled"} else "completed",
            "final_text": text,
            "events": _event("finalize", "ok", "engineering workflow finished"),
        }


__all__ = [
    "EngineeringGraphServices",
    "build_deterministic_plan",
    "MAX_REPAIR_ATTEMPTS",
]
