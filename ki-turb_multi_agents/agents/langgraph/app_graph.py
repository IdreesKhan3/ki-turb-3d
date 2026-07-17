"""Single root LangGraph for every KI-TURB user request."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .engineering_graph import build_engineering_subgraph
from .engineering_intent import parse_engineering_intent
from .engineering_services import EngineeringGraphServices
from .hit_graph import build_hit_subgraph
from .hit_services import HITGraphServices
from .models import WorkflowPlan
from .recovery import MAX_RECOVER_ATTEMPTS, recovery_plan
from .request_intent import (
    ACTIVE_SIMULATION_JOB_ID,
    COMPARISON_JOB_DIRS,
    LATEST_SIMULATION_JOB_ID,
    classify_request,
)
from .router import RequestRouter
from .state import KITurbState
from .workflow_guards import guard_tool
from .workflow_verify import verify_step
from .workflow_world import snapshot_world
from agents.runtime import tool_registry
from agents.tools import execute_tool

ROLES = ("orchestrator", "steward", "simulation", "analyst", "visualizer", "reviewer", "engineer")
_JOB_ID_TOOLS = frozenset({
    "compile_simulation",
    "start_simulation",
    "check_simulation_status",
    "cancel_simulation",
    "supervise_simulation",
    "fetch_simulation_outputs",
    "postprocess_simulation_outputs",
    "read_dataset_manifest",
    "load_dataset_manifest",
})


def _extract_job_id(raw: Any) -> str | None:
    for line in str(raw or "").splitlines():
        if line.lower().startswith("job_id:"):
            return line.split(":", 1)[1].strip()
    return None


def _remember_simulation_job(session_context: Dict[str, Any], job_id: str) -> None:
    session_context["simulation_job_id"] = job_id
    session_context["sim_workflow_job"] = job_id


def _track_comparison_job(session_context: Dict[str, Any], job_id: str) -> None:
    job_id = str(job_id or "").strip()
    if not job_id or job_id.startswith("__"):
        return
    tracked = session_context.setdefault("comparison_job_ids", [])
    if job_id not in tracked:
        tracked.append(job_id)


def _resolve_comparison_data_directories(
    project_root: Path,
    session_context: Dict[str, Any],
) -> list[str]:
    from agents.tools.simulation import _store as job_store

    dirs: list[str] = []
    for job_id in list(session_context.get("comparison_job_ids") or []):
        manifest = job_store.load_manifest(project_root, str(job_id))
        if manifest is not None and getattr(manifest, "base_dir", None):
            base = Path(str(manifest.base_dir))
            if base.is_dir() and str(base) not in dirs:
                dirs.append(str(base))
                continue
        processed = job_store.job_dir(project_root, str(job_id)) / "processed"
        if processed.is_dir() and str(processed) not in dirs:
            dirs.append(str(processed))
    return dirs


def _expand_comparison_dirs_in_args(
    tool_args: Dict[str, Any],
    project_root: Path,
    session_context: Dict[str, Any],
) -> None:
    raw = tool_args.get("data_directories")
    if not isinstance(raw, list) or COMPARISON_JOB_DIRS not in raw:
        return
    resolved = _resolve_comparison_data_directories(project_root, session_context)
    if not resolved:
        return
    tool_args["data_directories"] = resolved
    session_context["data_directories"] = list(resolved)
    session_context["data_directory"] = resolved[0]


def _store_world(session_context: Dict[str, Any], world) -> None:
    session_context["workflow_world"] = world.model_dump(mode="json")


def _intent_action(state: KITurbState, session_context: Dict[str, Any]) -> str | None:
    meta = state.get("metadata") or {}
    intent = meta.get("request_intent") or session_context.get("request_intent") or {}
    if isinstance(intent, dict):
        return intent.get("action")
    return None


def _event(stage: str, status: str, message: str = "") -> list[dict]:
    return [{"stage": stage, "status": status, "message": message}]


def _content(message: Any) -> str:
    value = getattr(message, "content", "")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value or "")


def _parse_tool_payload(content: str):
    try:
        value = json.loads(content)
    except Exception:
        return None
    return value


# KI_TURB_TOOLMESSAGE_ARTIFACTS_V1
def _tool_message_artifacts(message: Any) -> list[dict[str, Any]]:
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, dict) and artifact.get("artifact_type"):
        return [artifact]
    if isinstance(artifact, dict) and isinstance(artifact.get("artifacts"), list):
        return [item for item in artifact["artifacts"] if isinstance(item, dict)]
    if isinstance(artifact, list):
        return [item for item in artifact if isinstance(item, dict) and item.get("artifact_type")]
    return []


def _artifact_metadata(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep task-result evidence concise and avoid duplicating binary payloads."""
    return [
        {
            "artifact_type": item.get("artifact_type"),
            "artifact_title": item.get("artifact_title"),
            "message": item.get("message"),
        }
        for item in items
    ]


def _agent_context_snapshot(context: Dict[str, Any], *, limit: int = 12000) -> str:
    """Serialize live session context for agent prompts without huge binary payloads."""
    skip_keys = {"kiturb_workflow_state", "last_figure", "last_figure_image", "analysis_products"}
    payload: Dict[str, Any] = {}
    for key, value in (context or {}).items():
        if key in skip_keys:
            continue
        if key == "artifact_history":
            payload[key] = [
                {
                    "type": item.get("type"),
                    "caption": item.get("caption"),
                    "artifact_type": item.get("artifact_type") or item.get("type"),
                }
                for item in (value or [])[-12:]
                if isinstance(item, dict)
            ]
            continue
        if key == "turn_memory" and isinstance(value, dict):
            payload[key] = value
            continue
        payload[key] = value
    text = json.dumps(payload, default=str)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


class AppGraphNodes:
    # KI_TURB_DIRECT_TOOL_EXECUTION_V1
    def __init__(self, router: RequestRouter, project_root: str | Path, session_context: Dict[str, Any]):
        self.router = router
        self.project_root = Path(project_root).resolve()
        self.session_context = session_context

    @staticmethod
    def _step(state: KITurbState):
        plan = WorkflowPlan.model_validate(state["plan"])
        return plan.steps[int(state.get("task_index", 0))]

    @staticmethod
    def _is_supervise_step(state: KITurbState) -> bool:
        plan = WorkflowPlan.model_validate(state["plan"])
        index = int(state.get("task_index", 0))
        return index < len(plan.steps) and plan.steps[index].tool == "supervise_simulation"

    def _flush_activity_ui(self) -> None:
        from agents.tools.simulation._activity import (
            ACTIVITY_RENDER_CALLBACK_KEY,
            flush_simulation_progress,
        )

        flush_simulation_progress(self.session_context)
        render = self.session_context.get(ACTIVITY_RENDER_CALLBACK_KEY)
        if callable(render):
            try:
                render(force=True)
            except Exception:
                pass

    @staticmethod
    def _tool_failure(raw: Any) -> tuple[bool, str]:
        payload = raw
        if isinstance(raw, str):
            stripped = raw.strip()
            from agents.tools.simulation._status import tool_text_indicates_failure
            if tool_text_indicates_failure(stripped):
                return True, stripped
            try:
                payload = json.loads(stripped)
            except Exception:
                payload = None
            if stripped.lower().startswith(("error:", "tool error:")):
                return True, stripped
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").lower()
            message = str(payload.get("message") or payload.get("error") or "")
            if status in {"error", "failed", "failure"} or payload.get("success") is False:
                return True, message or json.dumps(payload, default=str)
        return False, ""

    @staticmethod
    def _tool_summary(raw: Any) -> str:
        if isinstance(raw, dict):
            return str(raw.get("message") or raw.get("error") or raw.get("status") or "Tool completed.")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                return raw
            if isinstance(payload, dict):
                return str(payload.get("message") or payload.get("error") or payload.get("status") or raw)
        return str(raw)

    def execute_step(self, state: KITurbState) -> Dict[str, Any]:
        step = self._step(state)
        role = step.role
        tool_name = step.tool
        tool_args = dict(step.tool_args or {})
        if not tool_name:
            return {"errors": [f"Direct step for {role} has no tool."], "status": "failed"}

        if tool_name in _JOB_ID_TOOLS:
            job_id = str(tool_args.get("job_id") or "").strip()
            if job_id == LATEST_SIMULATION_JOB_ID:
                from agents.tools.simulation import _store as job_store

                latest = job_store.latest_job_id_with_manifest(Path(self.project_root))
                if not latest:
                    latest = job_store.latest_job_id(Path(self.project_root))
                if not latest:
                    return {
                        "errors": [f"{tool_name} failed: no saved simulation jobs found."],
                        "status": "failed",
                        "final_text": f"{tool_name} failed: no saved simulation jobs under simulations/.",
                    }
                tool_args["job_id"] = latest
                _remember_simulation_job(self.session_context, latest)
            elif not job_id or job_id == ACTIVE_SIMULATION_JOB_ID:
                active = str(self.session_context.get("simulation_job_id") or "").strip()
                if not active:
                    return {
                        "errors": [f"{tool_name} requires an active simulation job_id."],
                        "status": "failed",
                        "final_text": f"Simulation step '{tool_name}' failed: no active job_id.",
                    }
                tool_args["job_id"] = active
            else:
                _remember_simulation_job(self.session_context, job_id)

        _expand_comparison_dirs_in_args(tool_args, Path(self.project_root), self.session_context)

        # Guarded transition against explicit world state.
        world_before = snapshot_world(
            self.project_root,
            job_id=str(tool_args.get("job_id") or self.session_context.get("simulation_job_id") or "") or None,
            session_context=self.session_context,
        )
        _store_world(self.session_context, world_before)
        guard = guard_tool(tool_name, world_before)
        if not guard.allowed:
            message = f"Guard blocked {tool_name}: {guard.reason}"
            return {
                "errors": [message],
                "status": "failed",
                "final_text": message,
                "events": _event("guard", "failed", message),
                "metadata": {
                    **dict(state.get("metadata") or {}),
                    "workflow_world": world_before.model_dump(mode="json"),
                    "last_guard": guard.model_dump(mode="json"),
                },
            }

        allowed = set(tool_registry.tools_for_agent(role))
        call_id = f"direct_{uuid.uuid4().hex}"
        call_message = AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": tool_args, "id": call_id, "type": "tool_call"}],
        )
        old_approved = self.session_context.get("tool_confirmation_approved")
        self.session_context["tool_confirmation_approved"] = True
        try:
            raw = execute_tool(
                tool_name,
                tool_args,
                self.project_root,
                session_context=self.session_context,
                allowed_tool_names=allowed,
            )
        finally:
            if old_approved is None:
                self.session_context.pop("tool_confirmation_approved", None)
            else:
                self.session_context["tool_confirmation_approved"] = old_approved
        captured_job_id = _extract_job_id(raw)
        if captured_job_id:
            _remember_simulation_job(self.session_context, captured_job_id)
        if tool_name == "fetch_simulation_outputs" and isinstance(raw, str):
            for line in raw.splitlines():
                if line.lower().startswith("manifest:"):
                    self.session_context["manifest_path"] = line.split(":", 1)[1].strip()
                    break
        failed, failure_message = self._tool_failure(raw)
        summary = failure_message if failed else self._tool_summary(raw)
        if not failed and tool_name in {
            "postprocess_simulation_outputs",
            "load_dataset_manifest",
        }:
            tracked = str(
                tool_args.get("job_id")
                or self.session_context.get("simulation_job_id")
                or captured_job_id
                or ""
            ).strip()
            _track_comparison_job(self.session_context, tracked)


        world_after = snapshot_world(
            self.project_root,
            job_id=str(
                tool_args.get("job_id")
                or self.session_context.get("simulation_job_id")
                or captured_job_id
                or ""
            ) or None,
            session_context=self.session_context,
        )
        _store_world(self.session_context, world_after)
        verify = verify_step(
            tool_name,
            raw if not failed else f"Error: {summary}",
            before=world_before,
            after=world_after,
            intent_action=_intent_action(state, self.session_context),
        )
        if not failed and not verify.ok:
            failed = True
            failure_message = f"Verify failed for {tool_name}: {verify.reason}"
            summary = failure_message

        artifact = raw if isinstance(raw, dict) and raw.get("artifact_type") else None
        tool_message = ToolMessage(
            content=summary,
            name=tool_name,
            tool_call_id=call_id,
            artifact=artifact,
        )
        # Put the tool outcome in AI text so collect/finalize never fall back to a
        # previous turn's final_text when this step has no free-form LLM prose.
        final_message = AIMessage(content=summary)
        update: Dict[str, Any] = {
            "messages": [call_message, tool_message, final_message],
            "status": "failed" if failed else "running",
            "final_text": summary,
            "events": _event(
                "tool",
                "failed" if failed else "ok",
                f"{role}: {tool_name} — {summary}",
            ),
            "metadata": {
                **dict(state.get("metadata") or {}),
                "workflow_world": world_after.model_dump(mode="json"),
                "last_guard": guard.model_dump(mode="json"),
                "last_verify": verify.model_dump(mode="json"),
            },
        }
        if failed:
            update["errors"] = [f"{role}.{tool_name}: {summary}"]
        return update

    def poll_simulation(self, state: KITurbState) -> Dict[str, Any]:
        """Deterministic single poll for supervise_simulation — no LLM, UI updates each tick."""
        import time

        from agents.tools.simulation import _store as job_store

        step = self._step(state)
        job_id = str(self.session_context.get("simulation_job_id") or "").strip()
        if not job_id:
            message = "Simulation monitor failed: no active job_id."
            return {
                "errors": [message],
                "status": "failed",
                "final_text": message,
                "events": _event("monitor", "failed", message),
            }

        allowed = set(tool_registry.tools_for_agent(step.role))
        raw = execute_tool(
            "check_simulation_status",
            {"job_id": job_id},
            self.project_root,
            session_context=self.session_context,
            allowed_tool_names=allowed,
        )
        self._flush_activity_ui()

        failed, failure_message = self._tool_failure(raw)
        summary = failure_message if failed else self._tool_summary(raw)
        index = int(state.get("task_index", 0))

        if failed:
            from .health_retry import (
                MAX_HEALTH_RETRIES,
                build_params_from_job,
                is_recoverable_health_rejection,
                load_job_measured,
                retune_build_params,
                splice_health_retry_plan,
            )

            meta = dict(state.get("metadata") or {})
            attempt = int(meta.get("health_retry_attempt") or self.session_context.get("health_retry_attempt") or 0)
            if is_recoverable_health_rejection(summary) and attempt < MAX_HEALTH_RETRIES:
                try:
                    base_params = build_params_from_job(self.project_root, job_id)
                    measured = load_job_measured(self.project_root, job_id)
                    retuned = retune_build_params(
                        base_params,
                        summary,
                        attempt=attempt,
                        measured=measured,
                    )
                    plan = WorkflowPlan.model_validate(state["plan"])
                    new_plan, new_index = splice_health_retry_plan(plan, index, retuned)
                    meta["health_retry_attempt"] = attempt + 1
                    meta["health_retry_reason"] = summary
                    meta["health_retry_params"] = {
                        k: retuned.get(k)
                        for k in (
                            "name", "char_velocity", "target_urms", "mach_number",
                            "scheme", "reynolds_number", "turbulence_regime", "hit_mode",
                        )
                    }
                    self.session_context["health_retry_attempt"] = attempt + 1
                    self.session_context["health_retry_params"] = meta["health_retry_params"]
                    retry_msg = (
                        f"Health rejection — retuning and retrying "
                        f"({attempt + 1}/{MAX_HEALTH_RETRIES}): "
                        f"u={retuned.get('char_velocity')}, scheme={retuned.get('scheme')}, "
                        f"Ma={retuned.get('mach_number')}"
                    )
                    return {
                        "plan": new_plan.model_dump(mode="json"),
                        "task_index": new_index,
                        "status": "running",
                        "final_text": retry_msg,
                        "task_results": [{
                            "role": step.role,
                            "text": retry_msg + f"\nPrior: {summary}",
                            "tool_outputs": [summary],
                        }],
                        "metadata": meta,
                        "events": _event("monitor", "retry", retry_msg),
                    }
                except Exception as exc:
                    summary = f"{summary} (health retry failed: {exc})"

            return {
                "errors": [f"{step.role}.supervise_simulation: {summary}"],
                "status": "failed",
                "final_text": summary,
                "task_index": index + 1,
                "task_results": [{"role": step.role, "text": summary, "tool_outputs": [summary]}],
                "events": _event("monitor", "failed", summary),
            }

        job = job_store.load_job(self.project_root, job_id)
        if job is not None and job.status.is_terminal:
            return {
                "task_results": [{"role": step.role, "text": summary, "tool_outputs": [summary]}],
                "task_index": index + 1,
                "final_text": summary,
                "status": "running",
                "events": _event("monitor", "ok", summary),
            }

        # Still running — graph loops back to this node after a short pause.
        time.sleep(2.0)
        return {
            "status": "running",
            "events": _event("monitor", "running", summary),
        }

    def plan(self, state: KITurbState) -> Dict[str, Any]:
        try:
            session_summary = state.get("session_summary") or {}
            request = state.get("user_request", "")
            intent = classify_request(
                request,
                session_summary=session_summary,
                project_root=self.project_root,
            )
            if intent is not None:
                self.session_context["request_intent"] = intent.model_dump(mode="json")
            plan = self.router.plan(request, session_summary)
            self.session_context["comparison_job_ids"] = []
            self.session_context.pop("health_retry_attempt", None)
            self.session_context.pop("health_retry_params", None)
            job_hint = None
            if intent and intent.job_ref and not str(intent.job_ref).startswith("__"):
                job_hint = intent.job_ref
            world = snapshot_world(
                self.project_root,
                job_id=job_hint,
                session_context=self.session_context,
            )
            _store_world(self.session_context, world)
            meta = {
                **dict(state.get("metadata") or {}),
                "request_intent": intent.model_dump(mode="json") if intent else None,
                "workflow_world": world.model_dump(mode="json"),
                "health_retry_attempt": 0,
                "recover_attempts": 0,
            }
            if plan.kind == "engineering_workflow":
                eng_intent = parse_engineering_intent(request, session_summary)
                eng_meta = eng_intent.to_metadata() if eng_intent else {"plan_only": False}
                # Merge any annotations the router stashed on the summary dict.
                if isinstance(session_summary.get("engineering_intent"), dict):
                    eng_meta = {**eng_meta, **session_summary["engineering_intent"]}
                self.session_context["engineering_intent"] = eng_meta
                meta["engineering"] = eng_meta
            return {
                "plan": plan.model_dump(mode="json"),
                "task_index": 0,
                "status": "planned",
                "final_text": "",
                "events": _event("plan", "ok", plan.rationale),
                "metadata": meta,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "errors": [f"planning failed: {exc}"],
                "final_text": f"Planning failed: {exc}",
                "events": _event("plan", "failed", str(exc)),
            }

    def prepare_step(self, state: KITurbState) -> Dict[str, Any]:
        plan = WorkflowPlan.model_validate(state["plan"])
        index = int(state.get("task_index", 0))
        step = plan.steps[index]
        parts = []
        intent_override = str(state.get("intent_override_text") or "").strip()
        if intent_override:
            parts.append(intent_override)
        if step.tool:
            # Direct tool execution — keep the delegate prompt minimal (no session JSON dump).
            parts.append(
                f"KI-TURB workflow step {index + 1}/{len(plan.steps)}.\n"
                f"Task: {step.instruction}\n"
                f"Required tool: {step.tool}"
            )
            if step.tool_args:
                parts.append("Tool args: " + json.dumps(step.tool_args, default=str))
        else:
            parts.append(
                f"KI-TURB workflow step {index + 1}/{len(plan.steps)}.\n"
                f"User request: {state.get('user_request', '')}\n"
                f"Assigned task: {step.instruction}"
            )
            prior = state.get("task_results") or []
            if prior:
                last = prior[-1] if isinstance(prior[-1], dict) else {}
                evidence = str(last.get("text") or "").strip()
                if evidence:
                    parts.append(
                        "Prior step evidence (use this — especially CASE_PARAMS JSON):\n"
                        + evidence[:12000]
                    )
            prevent_tools = [str(name) for name in (state.get("prevent_tools") or []) if name]
            if prevent_tools:
                parts.append("Do not call these tools: " + ", ".join(prevent_tools))
            from .turn_memory import format_turn_memory

            memory = self.session_context.get("turn_memory") or (state.get("session_summary") or {}).get("turn_memory")
            memory_text = format_turn_memory(memory if isinstance(memory, dict) else None)
            if memory_text:
                parts.append(memory_text)
            history = state.get("chat_history") or []
            if history:
                recent = []
                for item in history[-6:]:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    content = str(item.get("content") or "").strip()
                    if role in {"user", "assistant"} and content:
                        recent.append(f"{role}: {content[:500]}")
                if recent:
                    parts.append("Recent chat turns:\n" + "\n".join(recent))
            parts.append(
                "Session context (manual UI pages read these values after sync):\n"
                + _agent_context_snapshot(self.session_context)
            )
            parts.append("Use only your authorized tools. Return a concise evidence-based result.")
        instruction = "\n\n".join(parts)

        # Attach turn images only when the active provider supports vision.
        from agents.shared.image_processor import (
            collect_turn_images,
            figure_text_context,
            langchain_human_content,
            provider_supports_vision,
        )

        provider = (
            str(self.session_context.get("llm_provider_name") or "").strip()
            or str((state.get("metadata") or {}).get("provider") or "").strip()
            or "deepseek"
        )
        supports_vision = provider_supports_vision(provider)
        text_fallback = figure_text_context(self.session_context)

        # Attach binary images at most once per user turn.
        turn_images: list = []
        already_attached = bool(self.session_context.get("_kiturb_vision_attached"))
        if supports_vision and not already_attached:
            turn_images = collect_turn_images(self.session_context, include_figures=True)
            if turn_images:
                self.session_context["_kiturb_vision_attached"] = True
                instruction = (
                    instruction
                    + "\n\nThe user attached image(s) and/or page plot figure(s) to this chat. "
                    "Inspect the image content visually and ground your answer in what you see."
                )
        elif text_fallback and "Latest plot summary" not in instruction:
            instruction = (
                instruction
                + "\n\nPlot/image context for this turn (text summary — "
                + ("provider has no vision" if not supports_vision else "images already attached earlier")
                + "):\n"
                + text_fallback
            )

        human_content = langchain_human_content(
            instruction,
            turn_images,
            supports_vision=supports_vision,
            text_fallback="" if (supports_vision and turn_images) else text_fallback,
        )
        image_note = ""
        if turn_images:
            image_note = f" (+{len(turn_images)} vision image)"
        elif text_fallback:
            image_note = " (+plot text context)"
        return {
            "active_role": step.role,
            "active_tool": step.tool,
            "active_tool_args": dict(step.tool_args or {}),
            "message_cursor": len(state.get("messages") or []),
            "messages": [HumanMessage(content=human_content)],
            "status": "running",
            "events": _event(
                "delegate",
                "ok",
                f"{step.role}: {step.instruction}" + image_note,
            ),
        }

    def collect_step(self, state: KITurbState) -> Dict[str, Any]:
        messages = state.get("messages") or []
        cursor = int(state.get("message_cursor", 0))
        new_messages = messages[cursor:]
        artifacts = []
        tool_outputs = []
        final_text = ""
        tool_text_bits: list[str] = []
        for message in new_messages:
            if isinstance(message, ToolMessage):
                job_id = _extract_job_id(_content(message))
                if job_id:
                    _remember_simulation_job(self.session_context, job_id)
                if message.name == "fetch_simulation_outputs":
                    for line in _content(message).splitlines():
                        if line.lower().startswith("manifest:"):
                            self.session_context["manifest_path"] = line.split(":", 1)[1].strip()
                            break

                message_artifacts = _tool_message_artifacts(message)
                if message_artifacts:
                    artifacts.extend(message_artifacts)
                    tool_outputs.extend(_artifact_metadata(message_artifacts))
                    continue

                payload = _parse_tool_payload(_content(message))
                if isinstance(payload, dict) and payload.get("artifact_type"):
                    artifacts.append(payload)
                elif isinstance(payload, dict) and isinstance(payload.get("artifacts"), list):
                    artifacts.extend(item for item in payload["artifacts"] if isinstance(item, dict))
                content = _content(message).strip()
                tool_outputs.append(payload if payload is not None else content)
                if content:
                    name = str(getattr(message, "name", "") or "tool")
                    tool_text_bits.append(f"{name}: {content}")
            elif isinstance(message, AIMessage) and _content(message).strip():
                final_text = _content(message).strip()
        # Prefer this step's LLM prose; else tool evidence. Never reuse a prior turn.
        if not final_text and tool_text_bits:
            final_text = "\n".join(tool_text_bits)
        index = int(state.get("task_index", 0))
        result = {
            "role": state.get("active_role", "unknown"),
            "text": final_text,
            "tool_outputs": tool_outputs,
        }
        return {
            "task_results": [result],
            "artifacts": artifacts,
            "task_index": index + 1,
            "final_text": final_text,
            "events": _event(
                "step",
                "failed" if state.get("errors") else "ok",
                f"{'failed' if state.get('errors') else 'completed'} {state.get('active_role', 'role')} step",
            ),
        }

    def recover_step(self, state: KITurbState) -> Dict[str, Any]:
        """On step failure: hand off / replan instead of dying immediately."""
        from langgraph.types import Overwrite

        meta = dict(state.get("metadata") or {})
        attempts = int(meta.get("recover_attempts") or 0)
        raw_errors = state.get("errors")
        if type(raw_errors).__name__ == "Overwrite":
            raw_errors = getattr(raw_errors, "value", None) or []
        errors = [str(e) for e in (raw_errors or []) if e]
        failure = "; ".join(errors) or str(state.get("final_text") or "step failed")

        if attempts >= MAX_RECOVER_ATTEMPTS:
            return {
                "status": "failed",
                "final_text": (
                    f"Could not recover after {attempts} handoff attempt(s).\n{failure}"
                ),
                "events": _event("recover", "exhausted", failure[:300]),
                "metadata": meta,
            }

        plan = recovery_plan(
            user_request=str(state.get("user_request") or ""),
            failure=failure,
            task_results=list(state.get("task_results") or []),
            planner_agent=getattr(self.router, "planner_agent", None),
        )
        meta["recover_attempts"] = attempts + 1
        meta["last_recovery"] = {
            "from_error": failure[:500],
            "rationale": plan.rationale,
            "attempt": attempts + 1,
        }
        return {
            "plan": plan.model_dump(mode="json"),
            "task_index": 0,
            "errors": Overwrite([]),
            "status": "running",
            "final_text": "",
            "metadata": meta,
            "task_results": [{
                "role": "orchestrator",
                "text": f"Recovering (attempt {attempts + 1}): {plan.rationale}\nPrior failure: {failure[:400]}",
                "tool_outputs": [],
            }],
            "events": _event("recover", "ok", plan.rationale),
        }

    def completion_check(self, state: KITurbState) -> Dict[str, Any]:
        """Before finalize: verify deliverables; inject finish steps if gaps remain."""
        from .completion_check import (
            MAX_COMPLETION_ATTEMPTS,
            evaluate_completion,
            exhaustion_message,
            finish_work_plan,
        )

        meta = dict(state.get("metadata") or {})
        # Failures already handled — do not reopen the request.
        if state.get("errors") or state.get("status") == "failed":
            meta["completion_check"] = {"skipped": True, "reason": "failed"}
            return {"metadata": meta, "status": state.get("status") or "failed"}

        evaluation = evaluate_completion(
            user_request=str(state.get("user_request") or ""),
            task_results=list(state.get("task_results") or []),
            session_context=self.session_context,
            artifacts=list(state.get("artifacts") or []),
            final_text=str(state.get("final_text") or ""),
        )
        attempts = int(meta.get("completion_attempts") or 0)
        meta["completion_check"] = {
            "complete": evaluation["complete"],
            "missing": evaluation.get("missing") or [],
            "gaps": evaluation.get("gaps") or [],
            "job_ids": evaluation.get("job_ids") or [],
            "attempt": attempts,
        }

        if evaluation["complete"]:
            return {
                "metadata": meta,
                "events": _event("completion_check", "ok", "deliverables satisfied"),
            }

        if attempts >= MAX_COMPLETION_ATTEMPTS:
            message = exhaustion_message(str(state.get("user_request") or ""), evaluation)
            return {
                "metadata": meta,
                "status": "insufficient_data",
                "final_text": message,
                "events": _event("completion_check", "exhausted", message[:300]),
            }

        finish = finish_work_plan(
            user_request=str(state.get("user_request") or ""),
            evaluation=evaluation,
            attempt=attempts,
        )
        plan = WorkflowPlan.model_validate(state.get("plan") or {})
        new_index = len(plan.steps)
        plan.steps.extend(finish.steps)
        meta["completion_attempts"] = attempts + 1
        return {
            "plan": plan.model_dump(mode="json"),
            "task_index": new_index,
            "status": "running",
            "metadata": meta,
            "events": _event(
                "completion_check",
                "retry",
                f"attempt {attempts + 1}: " + ", ".join(evaluation.get("missing") or []),
            ),
        }

    def finalize(self, state: KITurbState) -> Dict[str, Any]:
        from .turn_memory import update_turn_memory

        if state.get("errors"):
            text = state.get("final_text") or ("Errors: " + "; ".join(state["errors"]))
            status = "failed"
        else:
            results = state.get("task_results") or []
            # Prefer the latest step with text — this turn only (accumulators reset on new turns).
            text = ""
            for item in reversed(results):
                if isinstance(item, dict) and str(item.get("text") or "").strip():
                    text = str(item["text"]).strip()
                    break
            if not text:
                text = str(state.get("final_text") or "").strip()
            text = text or "KI-TURB workflow completed."
            status = state.get("status") if state.get("status") in {"accepted", "rejected", "insufficient_data"} else "completed"

        memory = update_turn_memory(
            self.session_context.get("turn_memory"),
            user_request=str(state.get("user_request") or ""),
            plan=state.get("plan") if isinstance(state.get("plan"), dict) else {},
            task_results=list(state.get("task_results") or []),
            artifacts=list(state.get("artifacts") or []),
            session_context=self.session_context,
            final_text=text,
            status=status,
        )
        self.session_context["turn_memory"] = memory
        if memory.get("job_id"):
            _remember_simulation_job(self.session_context, str(memory["job_id"]))
        if memory.get("manifest_path"):
            self.session_context["manifest_path"] = memory["manifest_path"]
        return {"status": status, "final_text": text, "metadata": {**(state.get("metadata") or {}), "turn_memory": memory}}


def build_app_graph(
    *,
    router: RequestRouter,
    role_agents: Dict[str, Any],
    hit_services: HITGraphServices,
    checkpointer: Any,
    project_root: str | Path,
    session_context: Dict[str, Any],
    engineering_services: EngineeringGraphServices | None = None,
):
    from langgraph.graph import END, START, StateGraph
    nodes = AppGraphNodes(router, project_root, session_context)
    if engineering_services is not None:
        eng_services = engineering_services
    else:
        from .settings import LangGraphSettings

        settings = getattr(hit_services, "settings", None)
        if settings is None:
            settings = LangGraphSettings.from_environment(project_root, "ollama")
        eng_services = EngineeringGraphServices.default(
            settings,
            project_root,
            session_context,
            getattr(hit_services, "role_factory", None),
        )
    graph = StateGraph(KITurbState)
    graph.add_node("plan", nodes.plan)
    graph.add_node("hit_workflow", build_hit_subgraph(hit_services))
    graph.add_node("engineering_workflow", build_engineering_subgraph(eng_services))
    graph.add_node("prepare_step", nodes.prepare_step)
    graph.add_node("execute_step", nodes.execute_step)
    for role in ROLES:
        graph.add_node(f"{role}_agent", role_agents[role])
    graph.add_node("collect_step", nodes.collect_step)
    graph.add_node("recover_step", nodes.recover_step)
    graph.add_node("poll_simulation", nodes.poll_simulation)
    graph.add_node("completion_check", nodes.completion_check)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "plan")

    def route_plan(state: KITurbState) -> str:
        if state.get("errors"):
            return "finalize"
        kind = (state.get("plan") or {}).get("kind")
        if kind == "hit_workflow":
            return "hit_workflow"
        if kind == "engineering_workflow":
            return "engineering_workflow"
        return "prepare_step"

    graph.add_conditional_edges(
        "plan",
        route_plan,
        {
            "hit_workflow": "hit_workflow",
            "engineering_workflow": "engineering_workflow",
            "prepare_step": "prepare_step",
            "finalize": "finalize",
        },
    )
    graph.add_edge("hit_workflow", "finalize")

    def route_after_engineering(state: KITurbState) -> str:
        meta = state.get("metadata") or {}
        if meta.get("engineering_fallback") and (state.get("plan") or {}).get("kind") == "agent_workflow":
            return "prepare_step"
        return "finalize"

    graph.add_conditional_edges(
        "engineering_workflow",
        route_after_engineering,
        {"prepare_step": "prepare_step", "finalize": "finalize"},
    )

    def route_role(state: KITurbState) -> str:
        step = nodes._step(state)
        if step.tool:
            return "execute_step"
        role = state.get("active_role", "orchestrator")
        return f"{role}_agent" if role in ROLES else "orchestrator_agent"

    route_targets = {f"{role}_agent": f"{role}_agent" for role in ROLES}
    route_targets["execute_step"] = "execute_step"
    graph.add_conditional_edges("prepare_step", route_role, route_targets)
    graph.add_edge("execute_step", "collect_step")
    for role in ROLES:
        graph.add_edge(f"{role}_agent", "collect_step")

    def next_step(state: KITurbState) -> str:
        if state.get("errors"):
            return "recover_step"
        plan = WorkflowPlan.model_validate(state["plan"])
        index = int(state.get("task_index", 0))
        if index < len(plan.steps) and plan.steps[index].tool == "supervise_simulation":
            return "poll_simulation"
        return "prepare_step" if index < len(plan.steps) else "completion_check"

    def after_recover(state: KITurbState) -> str:
        if state.get("errors") or state.get("status") == "failed":
            return "finalize"
        return "prepare_step"

    def after_poll(state: KITurbState) -> str:
        if state.get("errors"):
            return "recover_step"
        if AppGraphNodes._is_supervise_step(state):
            return "poll_simulation"
        plan = WorkflowPlan.model_validate(state["plan"])
        index = int(state.get("task_index", 0))
        return "prepare_step" if index < len(plan.steps) else "completion_check"

    def after_completion_check(state: KITurbState) -> str:
        plan = WorkflowPlan.model_validate(state.get("plan") or {})
        index = int(state.get("task_index", 0))
        meta = state.get("metadata") or {}
        check = meta.get("completion_check") if isinstance(meta, dict) else None
        if (
            state.get("status") == "running"
            and isinstance(check, dict)
            and not check.get("complete")
            and index < len(plan.steps)
        ):
            return "prepare_step"
        return "finalize"

    graph.add_conditional_edges(
        "collect_step",
        next_step,
        {
            "prepare_step": "prepare_step",
            "poll_simulation": "poll_simulation",
            "recover_step": "recover_step",
            "completion_check": "completion_check",
        },
    )
    graph.add_conditional_edges(
        "recover_step",
        after_recover,
        {"prepare_step": "prepare_step", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "poll_simulation",
        after_poll,
        {
            "poll_simulation": "poll_simulation",
            "prepare_step": "prepare_step",
            "recover_step": "recover_step",
            "completion_check": "completion_check",
        },
    )
    graph.add_conditional_edges(
        "completion_check",
        after_completion_check,
        {"prepare_step": "prepare_step", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, name="kiturb")


__all__ = ["build_app_graph", "AppGraphNodes", "ROLES"]
