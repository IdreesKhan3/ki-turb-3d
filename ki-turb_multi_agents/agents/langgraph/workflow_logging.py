"""Format LangGraph workflow updates for the Autonomous Lab activity UI."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

KI_TURB_STRUCTURED_ACTIVITY_V2 = True


def _message_text(message: Any) -> str:
    value = getattr(message, "content", "")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in value
        )
    return str(value or "")


def _agent(node_name: str) -> str:
    return node_name.removesuffix("_agent")


def _one_line(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_json(value: Any, limit: int = 1800) -> str:
    try:
        text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n…"
    return text


def _args_summary(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    preferred = ("data_dir", "data_reference", "mode", "filepath", "query", "show_kolmogorov")
    parts: list[str] = []
    for key in preferred:
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, (dict, list)):
            value = _one_line(_safe_json(value, 160), 120)
        parts.append(f"{key.replace('_', ' ')}: {value}")
        if len(parts) == 3:
            break
    return " · ".join(parts) or f"{len(args)} parameters"


def _result_summary(text: str) -> tuple[str, str, str]:
    raw = text.strip()
    if not raw:
        return "success", "Completed", ""
    if raw.lower().startswith("error:"):
        summary = raw.split(":", 1)[1].strip() or "Tool failed"
        return "error", _one_line(summary, 220), raw
    try:
        payload = json.loads(raw)
    except Exception:
        lowered = raw.lower()
        status = "error" if "error" in lowered or "failed" in lowered else "success"
        return status, _one_line(raw, 220), raw if len(raw) > 220 else ""
    if not isinstance(payload, dict):
        return "success", _one_line(payload, 220), _safe_json(payload)
    status_value = str(payload.get("status") or "success").lower()
    status = "error" if status_value in {"error", "failed", "failure"} else "warning" if status_value == "warning" else "success"
    message = payload.get("message") or payload.get("summary") or payload.get("error")
    if not message:
        artifact_title = payload.get("artifact_title")
        message = f"Created {artifact_title}" if artifact_title else "Completed"
    details_payload = dict(payload)
    if "artifact_content" in details_payload:
        artifact_content = str(details_payload["artifact_content"])
        details_payload["artifact_content"] = (
            artifact_content[:900] + "…" if len(artifact_content) > 900 else artifact_content
        )
    return status, _one_line(message, 220), _safe_json(details_payload)


def format_workflow_events(
    node_name: str,
    update: dict[str, Any],
    *,
    include_ai_text: bool = True,
) -> list[dict[str, Any]]:
    """Return structured events consumed by the compact Streamlit renderer."""
    if not update:
        return []
    events: list[dict[str, Any]] = []
    agent = _agent(node_name)
    if node_name in {"poll_simulation", "execute_step"}:
        agent = "simulation"

    for raw_event in update.get("events") or []:
        stage = str(raw_event.get("stage") or "activity").replace("_", " ").title()
        message = str(raw_event.get("message") or "").strip()
        raw_status = str(raw_event.get("status") or "info").lower()
        status = "success" if raw_status in {"ok", "done", "complete", "completed", "success"} else raw_status
        if node_name == "poll_simulation" and message:
            events.append({
                "type": "tool_result",
                "kind": "tool",
                "agent": "simulation",
                "status": status,
                "title": "supervise_simulation",
                "summary": _one_line(message),
                "tool": "supervise_simulation",
            })
            continue
        if message:
            events.append({
                "type": "stage",
                "kind": "progress",
                "agent": agent,
                "status": status,
                "title": stage,
                "summary": _one_line(message),
            })

    if node_name == "plan" and update.get("plan"):
        plan = update["plan"]
        steps = plan.get("steps") or []
        roles = [str(step.get("role") or "agent").replace("_", " ").title() for step in steps]
        route = " → ".join(roles[:5]) or "Direct response"
        details_lines = []
        rationale = str(plan.get("rationale") or "").strip()
        if rationale:
            details_lines.append(rationale)
        for index, step in enumerate(steps, 1):
            details_lines.append(
                f"{index}. {str(step.get('role') or 'agent').replace('_', ' ').title()}: "
                f"{str(step.get('instruction') or '').strip()}"
            )
        events.append({
            "type": "plan",
            "kind": "plan",
            "agent": "orchestrator",
            "status": "success",
            "title": "Workflow planned",
            "summary": f"{len(steps)} step{'s' if len(steps) != 1 else ''} · {route}",
            "details": "\n".join(details_lines),
        })

    if node_name.endswith("_agent"):
        try:
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        except ImportError:
            AIMessage = HumanMessage = ToolMessage = ()  # type: ignore[assignment]

        for message in update.get("messages") or []:
            if isinstance(message, HumanMessage):
                text = _message_text(message).strip()
                if text:
                    events.append({
                        "type": "task",
                        "kind": "task",
                        "agent": agent,
                        "status": "running",
                        "title": "Task received",
                        "summary": _one_line(text),
                        "details": text if len(text) > 220 else "",
                    })
            elif isinstance(message, AIMessage):
                text = _message_text(message).strip()
                if text and include_ai_text:
                    events.append({
                        "type": "response",
                        "kind": "response",
                        "agent": agent,
                        "status": "success",
                        "title": "Response prepared",
                        "summary": _one_line(text),
                        "details": text if len(text) > 220 else "",
                    })
                for tool_call in getattr(message, "tool_calls", None) or []:
                    if isinstance(tool_call, dict):
                        name = str(tool_call.get("name") or "tool")
                        args = tool_call.get("args") or tool_call.get("arguments") or {}
                        call_id = str(tool_call.get("id") or "")
                    else:
                        name = str(getattr(tool_call, "name", "tool"))
                        args = getattr(tool_call, "args", {})
                        call_id = str(getattr(tool_call, "id", ""))
                    events.append({
                        "type": "tool_start",
                        "kind": "tool",
                        "agent": agent,
                        "status": "running",
                        "title": name,
                        "summary": _args_summary(args) or "Running tool",
                        "details": "Arguments\n" + _safe_json(args),
                        "tool": name,
                        "call_id": call_id,
                    })
            elif isinstance(message, ToolMessage):
                name = str(getattr(message, "name", "tool"))
                text = _message_text(message).strip()
                status, summary, details = _result_summary(text)
                events.append({
                    "type": "tool_result",
                    "kind": "tool",
                    "agent": agent,
                    "status": status,
                    "title": name,
                    "summary": summary,
                    "details": "Result\n" + details if details else "",
                    "tool": name,
                    "call_id": str(getattr(message, "tool_call_id", "") or ""),
                })

    if node_name == "collect_step":
        results = update.get("task_results") or []
        if results:
            latest = results[-1]
            role = str(latest.get("role") or "agent")
            text = str(latest.get("text") or "").strip()
            events.append({
                "type": "step_complete",
                "kind": "handoff",
                "agent": role,
                "status": "success",
                "title": "Step completed",
                "summary": _one_line(text) if text else "Result handed back to the orchestrator",
                "details": text if len(text) > 220 else "",
            })

    if update.get("final_text") and node_name in {"finalize", "summarize_hit"}:
        text = str(update["final_text"]).strip()
        if text:
            events.append({
                "type": "final",
                "kind": "final",
                "agent": "orchestrator",
                "status": "success",
                "title": "Final response ready",
                "summary": _one_line(text),
            })

    for error in _iter_state_list(update.get("errors")):
        events.append({
            "type": "error",
            "kind": "error",
            "agent": agent,
            "status": "error",
            "title": "Action failed",
            "summary": _one_line(error),
            "details": str(error),
        })
    for warning in _iter_state_list(update.get("warnings")):
        events.append({
            "type": "warning",
            "kind": "warning",
            "agent": agent,
            "status": "warning",
            "title": "Needs attention",
            "summary": _one_line(warning),
            "details": str(warning),
        })
    return events


def _iter_state_list(value):
    """Iterate list channels; LangGraph Overwrite([...]) must not be iterated raw."""
    if value is None:
        return []
    if type(value).__name__ == "Overwrite":
        inner = getattr(value, "value", None)
        return list(inner or [])
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


# Backward-compatible Markdown formatter retained for tests and non-UI callers.
def format_workflow_update(
    node_name: str,
    update: dict[str, Any],
    *,
    include_ai_text: bool = True,
) -> list[str]:
    lines: list[str] = []
    for event in format_workflow_events(node_name, update, include_ai_text=include_ai_text):
        kind = str(event.get("kind") or event.get("type") or "activity").upper()
        agent = str(event.get("agent") or "agent")
        title = str(event.get("title") or "Activity")
        summary = str(event.get("summary") or "")
        if event.get("type") == "plan":
            lines.append(f"**[PLAN]** {title}\n\n{summary}")
        elif event.get("type") == "tool_start":
            lines.append(f"**[{agent}]** → calling `{title}`\n\n{summary}")
        elif event.get("type") == "tool_result":
            lines.append(f"**[{agent}]** `{title}` result\n\n{summary}")
        else:
            lines.append(f"**[{kind}]** {agent}: {title}" + (f"\n\n{summary}" if summary else ""))
    return lines


__all__ = ["format_workflow_events", "format_workflow_update"]
