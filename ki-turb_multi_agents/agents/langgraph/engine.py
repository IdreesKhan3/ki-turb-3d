"""UI-compatible single LangChain/LangGraph engine for all KI-TURB requests."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .app_graph import ROLES, build_app_graph
from .checkpointing import CheckpointerHandle, create_checkpointer
from .compat import require_lang_dependencies
from .engineering_services import EngineeringGraphServices
from .hit_services import HITGraphServices
from .role_agents import RoleAgentFactory
from .router import RequestRouter
from .settings import LangGraphSettings
from .workflow_logging import format_workflow_events, format_workflow_update
from agents.intent_detection import get_plot_routing


def _jsonable(value: Any):
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _session_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "current_page", "selected_file", "selected_directory", "data_dir",
        "data_directory", "data_directories",
        "dataset_manifest_path", "manifest_path", "run_id", "report_sections",
        "hdf5_fortran_mode", "selection_mode", "hit_config", "requested_hit_config",
        "openlb_app_dir", "run_root", "simulation_job_id", "sim_workflow_job",
        "spectra_data_directory", "analysis_products_path",
        "engineering_plan", "engineering_step_index", "engineering_capability",
        "engineering_plan_approved", "engineering_intent",
        "turn_memory", "langgraph_thread_id",
    )
    result = {}
    for key in keys:
        if key in context:
            result[key] = _jsonable(context[key])
    return result


class KITurbGraphEngine:
    def __init__(
        self,
        project_root: str | Path,
        *,
        provider_name: str = "ollama",
        settings: Optional[LangGraphSettings] = None,
        log_callback=None,
        stream_callback=None,
        activity_render_callback=None,
        memory_checkpointer: bool = False,
    ):
        require_lang_dependencies(sqlite=not memory_checkpointer)
        self.project_root = Path(project_root).resolve()
        self.provider_name = provider_name
        self.settings = settings or LangGraphSettings.from_environment(self.project_root, provider_name)
        self.log_callback = log_callback or (lambda message: None)
        self.stream_callback = stream_callback
        self.activity_render_callback = activity_render_callback
        self.session_context: Dict[str, Any] = {}
        self.checkpointer_handle: CheckpointerHandle = create_checkpointer(self.settings.checkpoint_path, memory=memory_checkpointer)
        self.role_factory = RoleAgentFactory(self.settings.model, self.project_root, self.session_context)
        self.role_agents = {role: self.role_factory.create_role_agent(role) for role in ROLES}
        planner = self.role_factory.create_planner() if self.settings.use_llm_planner else None
        self.router = RequestRouter(planner, self.settings.max_plan_steps, project_root=self.project_root)
        self.hit_services = HITGraphServices.default(self.settings, self.project_root, self.role_factory)
        self.engineering_services = EngineeringGraphServices.default(
            self.settings,
            self.project_root,
            self.session_context,
            self.role_factory,
        )
        self.graph = build_app_graph(
            router=self.router,
            role_agents=self.role_agents,
            hit_services=self.hit_services,
            checkpointer=self.checkpointer_handle.saver,
            project_root=self.project_root,
            session_context=self.session_context,
            engineering_services=self.engineering_services,
        )

    def _openlb_app_dir(self) -> Path:
        rel = "cfd_solvers/SolverApps/kiTurbHIT3D"
        candidates = [
            self.project_root / rel,
            self.project_root.parent / rel,
            # Legacy fallback while older checkouts still have the in-tree example.
            self.project_root / "cfd_solvers/openLB/examples/kiTurbHIT3D",
            self.project_root.parent / "cfd_solvers/openLB/examples/kiTurbHIT3D",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]

    @staticmethod
    def _text_only_content(content: Any) -> str:
        """Flatten message content to plain text (drop multimodal parts)."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        piece = str(block.get("text") or "").strip()
                        if piece:
                            parts.append(piece)
                    elif block.get("type") in {"image_url", "image"}:
                        parts.append("[image omitted]")
                else:
                    piece = str(block).strip()
                    if piece:
                        parts.append(piece)
            return "\n".join(parts)
        return str(content)

    def _initial_state(self, user_message: str, chat_history, context: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        from langgraph.types import Overwrite

        messages = []
        for item in (chat_history or [])[-20:]:
            role = item.get("role")
            content = self._text_only_content(item.get("content"))
            if role in {"user", "assistant"} and content.strip():
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": self._text_only_content(user_message)})
        requested = context.get("hit_config") or context.get("requested_hit_config") or {}
        routing = get_plot_routing(user_message)
        # Same thread_id reuses the checkpointer; Overwrite resets add_messages /
        # accumulator channels so prior-turn state does not leak into this turn.
        return {
            "messages": Overwrite(messages),
            "workflow_version": 2,
            "thread_id": thread_id,
            "user_request": user_message,
            "chat_history": chat_history or [],
            "session_summary": _session_summary(context),
            "intent_override_text": routing.get("intent_override_text") or "",
            "prevent_tools": list(routing.get("prevent_tools") or []),
            "requested_config": requested if isinstance(requested, dict) else {},
            "status": "created",
            "approved": False,
            "require_approval": bool(context.get("require_execution_approval", self.settings.require_execution_approval)),
            "run_root": str(context.get("run_root") or self.settings.run_root),
            "openlb_app_dir": str(context.get("openlb_app_dir") or self._openlb_app_dir()),
            "final_text": "",
            "task_results": Overwrite([]),
            "artifacts": Overwrite([]),
            "warnings": Overwrite([]),
            "errors": Overwrite([]),
            "events": Overwrite([]),
            "task_index": 0,
            "plan": {},
            "metadata": {"provider": self.provider_name},
        }

    def _pending_response(self, result: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        interrupt_obj = result["__interrupt__"][0]
        value = _jsonable(getattr(interrupt_obj, "value", interrupt_obj))
        tool = "langgraph_approval"
        args: Dict[str, Any] = {}
        message = "KI-TURB requires approval to continue."
        kind = "approval"
        action_requests: list = []
        if isinstance(value, dict) and value.get("action_requests"):
            kind = "tool_review"
            action_requests = list(value["action_requests"])
            action = action_requests[0]
            tool = action.get("name", tool)
            args = action.get("arguments") or action.get("args") or {}
            if len(action_requests) == 1:
                message = action.get("description") or f"Approve tool '{tool}'?"
            else:
                names = ", ".join(f"`{item.get('name', 'tool')}`" for item in action_requests)
                message = f"Approve {len(action_requests)} tool calls: {names}"
        elif isinstance(value, dict):
            message = value.get("message", message)
            tool = value.get("kind", tool)
            if value.get("kind") == "engineering_plan_approval":
                kind = "engineering_plan_approval"
                args = {"engineering_plan": value.get("engineering_plan") or {}}
                message = value.get("message") or message
        return {
            "status": "pending_confirmation", "kind": "langgraph_interrupt",
            "tool": tool, "args": args, "message": message,
            "action_requests": action_requests,
            "langgraph_thread_id": thread_id,
            "langgraph_interrupt_type": kind,
            "langgraph_interrupt": value,
            "text": value.get("plan_text") if isinstance(value, dict) and value.get("plan_text") else message,
            "engineering_plan": value.get("engineering_plan") if isinstance(value, dict) else None,
        }

    def _log_workflow_update(
        self,
        node_name: str,
        update: Dict[str, Any],
        *,
        include_ai_text: bool = True,
    ) -> None:
        if not isinstance(update, dict):
            return
        from agents.tools.simulation._activity import flush_simulation_progress

        flush_simulation_progress(self.session_context)
        for event in format_workflow_events(
            node_name,
            update,
            include_ai_text=include_ai_text,
        ):
            self.log_callback(event)

    @staticmethod
    def _namespace_tuple(value: Any) -> Tuple[str, ...]:
        if isinstance(value, tuple):
            return tuple(str(item) for item in value)
        if isinstance(value, list):
            return tuple(str(item) for item in value)
        if value:
            return (str(value),)
        return ()

    @staticmethod
    def _agent_node(namespace: Tuple[str, ...], fallback: str = "agent") -> str:
        for item in reversed(namespace):
            name = item.split(":", 1)[0]
            if name.endswith("_agent"):
                return name
        return fallback

    @staticmethod
    def _message_text(message: Any) -> str:
        value = getattr(message, "content", "")
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in value
            )
        return str(value or "")

    @staticmethod
    def _unpack_stream_chunk(chunk: Any) -> Tuple[Tuple[str, ...], str, Any]:
        # LangGraph >= 1.1 v2 shape (accepted even though v2 is not required).
        if isinstance(chunk, dict) and chunk.get("type") and "data" in chunk:
            return (
                KITurbGraphEngine._namespace_tuple(chunk.get("ns")),
                str(chunk["type"]),
                chunk.get("data"),
            )
        # LangGraph v1: multiple modes + subgraphs -> (namespace, mode, data).
        if isinstance(chunk, tuple) and len(chunk) == 3:
            namespace, mode, data = chunk
            return KITurbGraphEngine._namespace_tuple(namespace), str(mode), data
        # LangGraph v1: multiple modes -> (mode, data), or one mode + subgraphs -> (namespace, data).
        if isinstance(chunk, tuple) and len(chunk) == 2:
            first, second = chunk
            if isinstance(first, str) and first in {"updates", "messages", "custom", "debug"}:
                return (), first, second
            return KITurbGraphEngine._namespace_tuple(first), "updates", second
        # Old/default single updates mode.
        return (), "updates", chunk

    def _graph_stream(self, input_value: Any, config: Dict[str, Any]) -> Iterable[Any]:
        modes = ["updates", "messages"]
        try:
            yield from self.graph.stream(
                input_value,
                config,
                stream_mode=modes,
                subgraphs=True,
            )
        except TypeError as exc:
            # Compatibility for older LangGraph releases that do not expose subgraphs=.
            if "subgraph" not in str(exc).lower():
                raise
            yield from self.graph.stream(input_value, config, stream_mode=modes)

    def _emit_token(self, namespace: Tuple[str, ...], data: Any) -> None:
        if self.stream_callback is None:
            return
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            return
        message, metadata = data
        # Only stream assistant tokens — never dump HumanMessage prompts or tool payloads.
        from langchain_core.messages import AIMessage

        if not isinstance(message, AIMessage):
            return
        text = self._message_text(message)
        if not text:
            return
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata_ns = metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns")
        effective_ns = namespace or self._namespace_tuple(metadata_ns)
        fallback = str(metadata.get("langgraph_node") or "agent")
        agent_node = self._agent_node(effective_ns, fallback=fallback)
        agent = agent_node.removesuffix("_agent")
        stream_id = "|".join(effective_ns) or f"{agent_node}:{metadata.get('langgraph_step', '')}"
        self.stream_callback(
            {
                "type": "token",
                "agent": agent,
                "stream_id": stream_id,
                "content": text,
            }
        )

    def _stream_graph(self, input_value: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        nested_agents_seen: set[str] = set()
        for raw_chunk in self._graph_stream(input_value, config):
            namespace, mode, data = self._unpack_stream_chunk(raw_chunk)
            if mode == "messages":
                self._emit_token(namespace, data)
                continue
            if mode != "updates" or not isinstance(data, dict):
                continue
            for node_name, update in data.items():
                display_node = self._agent_node(namespace, fallback=str(node_name))
                if namespace and display_node.endswith("_agent"):
                    nested_agents_seen.add(display_node)
                log_update = update
                # Nested create_agent emitted model/tool updates already. The parent
                # later returns the full message list, so suppress that duplicate replay.
                if (
                    not namespace
                    and str(node_name).endswith("_agent")
                    and str(node_name) in nested_agents_seen
                    and isinstance(update, dict)
                    and "messages" in update
                ):
                    log_update = dict(update)
                    log_update.pop("messages", None)
                self._log_workflow_update(
                    display_node,
                    log_update,
                    include_ai_text=self.stream_callback is None,
                )
        snapshot = self.graph.get_state(config)
        result = dict(snapshot.values or {})
        if snapshot.interrupts:
            result["__interrupt__"] = list(snapshot.interrupts)
        return result

    # KI_TURB_CONTEXT_ROUNDTRIP_V1
    def _sync_context_out(self, target: Dict[str, Any], thread_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        from agents.shared.session_context_sanitize import sanitize_session_context_for_persistence

        target.clear()
        target.update(sanitize_session_context_for_persistence(self.session_context))
        target["langgraph_thread_id"] = thread_id
        if result is not None:
            target["kiturb_workflow_state"] = result
            for key in ("manifest_path", "analysis_products_path", "report_path", "dashboard_path", "run_id"):
                if result.get(key):
                    target[key] = result[key]

    def run_chat_loop(self, user_message: str, *, chat_history=None, session_context: Optional[Dict[str, Any]] = None, resume_state: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
        context = session_context if session_context is not None else {}
        self.session_context.clear()
        self.session_context.update(context)
        thread_id = (resume_state or {}).get("langgraph_thread_id") or context.get("langgraph_thread_id") or f"kiturb-{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        self.log_callback({
            "type": "workflow_start",
            "kind": "system",
            "agent": "orchestrator",
            "status": "running",
            "title": "Starting workflow",
            "summary": "Preparing the agent team",
            "details": f"Thread: {thread_id}",
        })
        from agents.tools.simulation._activity import (
            ACTIVITY_CALLBACK_KEY,
            ACTIVITY_RENDER_CALLBACK_KEY,
        )

        self.session_context[ACTIVITY_CALLBACK_KEY] = self.log_callback
        if self.activity_render_callback is not None:
            self.session_context[ACTIVITY_RENDER_CALLBACK_KEY] = self.activity_render_callback
        try:
            return self._run_chat_loop_body(
                user_message,
                chat_history=chat_history,
                context=context,
                thread_id=thread_id,
                config=config,
                resume_state=resume_state,
            )
        finally:
            self.session_context.pop(ACTIVITY_CALLBACK_KEY, None)
            self.session_context.pop(ACTIVITY_RENDER_CALLBACK_KEY, None)

    def _run_chat_loop_body(
        self,
        user_message: str,
        *,
        chat_history,
        context: Dict[str, Any],
        thread_id: str,
        config: Dict[str, Any],
        resume_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        from langgraph.types import Command

        if resume_state and resume_state.get("langgraph_thread_id"):
            resume_value = resume_state.get("langgraph_resume_value")
            if resume_value is None:
                approved = bool(resume_state.get("langgraph_approved"))
                resume_value = {"approved": approved}
            result = self._stream_graph(Command(resume=resume_value), config)
        else:
            self.session_context.pop("_kiturb_vision_attached", None)
            context.pop("_kiturb_vision_attached", None)
            result = self._stream_graph(self._initial_state(user_message, chat_history, context, thread_id), config)
        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sync_context_out(context, thread_id, result)
            return self._pending_response(result, thread_id)
        self._sync_context_out(context, thread_id, result)
        artifacts = list(result.get("artifacts") or [])
        return {
            "text": result.get("final_text") or f"KI-TURB workflow finished with status {result.get('status', 'unknown')}.",
            "artifacts": artifacts,
            "artifact": artifacts[-1] if artifacts else None,
            "workflow_state": result,
            "langgraph_thread_id": thread_id,
        }

    def get_state(self, thread_id: str):
        return self.graph.get_state({"configurable": {"thread_id": thread_id}})

    def close(self) -> None:
        self.checkpointer_handle.close()


# Compatibility name used by the previous overlay and external imports.
LangGraphWorkflowEngine = KITurbGraphEngine

__all__ = ["KITurbGraphEngine", "LangGraphWorkflowEngine"]
