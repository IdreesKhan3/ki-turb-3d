"""
Tool confirmation UI for Autonomous Lab.
Shows Accept/Reject buttons with advanced diff preview when the agent requests
file modifications (modify_file, write_file, create_file).

FLOW: User Accept/Reject -> execute tool (if approved) -> resume agent loop ->
      sync_context_to_session() -> append assistant message to chat.
      Sync mappings live in session_sync.py (sectionized by page).

On Accept of file edits, original content is snapshotted so the user can
Retrieve (restore) the previous version afterward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import streamlit as st

from .diff_preview import get_diff_for_pending_tool
from .live_activity import LiveActivityRenderer
from .chat_figures import store_plotly_artifact_in_message
from .retrieve import (
    apply_retrieve_batch,
    batch_summary_lines,
    capture_retrieve_batch,
    push_retrieve_batch,
)

_REJECT_MESSAGE = (
    "User rejected the operation. Do not retry this exact tool call or the same "
    "file write/edit. Continue with a different approach or stop."
)


def init_confirmation_state():
    """Initialize session state keys for tool confirmation flow."""
    if "lab_pending_tool" not in st.session_state:
        st.session_state.lab_pending_tool = None
    if "lab_tool_confirmation_choice" not in st.session_state:
        st.session_state.lab_tool_confirmation_choice = None
    if "lab_confirm_decisions" not in st.session_state:
        st.session_state.lab_confirm_decisions = {}
    if "lab_confirm_inflight" not in st.session_state:
        st.session_state.lab_confirm_inflight = None
    if "lab_retrieve_stack" not in st.session_state:
        st.session_state.lab_retrieve_stack = []
    # Legacy single-slot key (migrated into the stack when present).
    if "lab_revert_data" not in st.session_state:
        st.session_state.lab_revert_data = None
    if "engineering_plan" not in st.session_state:
        st.session_state.engineering_plan = None
    if "engineering_step_index" not in st.session_state:
        st.session_state.engineering_step_index = 0
    if "engineering_plan_approved" not in st.session_state:
        st.session_state.engineering_plan_approved = False


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def pending_fingerprint(pending: Dict[str, Any]) -> str:
    """Stable id for a pending confirmation (tool + args + thread)."""
    actions = []
    for action in _action_requests_from_pending(pending):
        actions.append({
            "name": action.get("name") or action.get("tool") or "",
            "args": action.get("args") or action.get("arguments") or {},
        })
    if not actions:
        actions = [{
            "name": pending.get("tool") or "",
            "args": pending.get("args") or {},
        }]
    payload = {
        "thread": pending.get("langgraph_thread_id") or "",
        "kind": pending.get("langgraph_interrupt_type") or pending.get("kind") or "",
        "actions": actions,
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8", errors="replace")).hexdigest()
    return digest[:20]


def _record_decision(fingerprint: str, choice: str) -> None:
    history = st.session_state.setdefault("lab_confirm_decisions", {})
    prev = dict(history.get(fingerprint) or {})
    prev["decision"] = "approved" if choice == "approved" else "rejected"
    prev["count"] = int(prev.get("count") or 0) + 1
    history[fingerprint] = prev


def auto_choice_for_duplicate_pending(pending: Dict[str, Any]) -> Optional[str]:
    """
    If this exact pending action was already decided, return that choice so the
    UI does not ask again (breaks Accept/Reject loops when the agent re-interrupts).
    """
    fingerprint = pending_fingerprint(pending)
    prev = (st.session_state.get("lab_confirm_decisions") or {}).get(fingerprint) or {}
    decision = prev.get("decision")
    if decision not in {"approved", "rejected"}:
        return None
    # Already decided this exact payload — replay without showing buttons again.
    return decision


def _set_confirmation_choice(choice: str) -> None:
    st.session_state.lab_tool_confirmation_choice = choice


def _migrate_legacy_revert_slot() -> None:
    """Move old lab_revert_data into lab_retrieve_stack once."""
    legacy = st.session_state.get("lab_revert_data")
    if not legacy:
        return
    stack = list(st.session_state.get("lab_retrieve_stack") or [])
    # Avoid double-migrating the same legacy payload.
    if not any(b.get("id") == "legacy-revert" for b in stack):
        entry = {
            "kind": "modify" if legacy.get("file_existed", True) else "create",
            "tool": legacy.get("tool") or "modify_file",
            "filepath": legacy["filepath"],
            "new_filepath": None,
            "filename": legacy.get("filename") or Path(legacy["filepath"]).name,
            "file_existed": legacy.get("file_existed", True),
            "original_content": legacy.get("original_content", ""),
            "summary": f"change `{legacy.get('filename') or Path(legacy['filepath']).name}`",
        }
        stack = push_retrieve_batch(stack, {
            "id": "legacy-revert",
            "captured_at": 0,
            "entries": [entry],
            "label": entry["filename"],
            "count": 1,
        })
        st.session_state.lab_retrieve_stack = stack
    st.session_state.lab_revert_data = None


def _infer_language(filename: str) -> str:
    """Infer code language from filename for syntax highlighting."""
    if filename.endswith(".py"):
        return "python"
    if filename.endswith((".f90", ".f", ".f95")):
        return "fortran"
    if filename.endswith((".js", ".ts", ".tsx", ".jsx")):
        return "javascript"
    if filename.endswith((".json", ".yaml", ".yml")):
        return "json" if "json" in filename else "yaml"
    return "text"


def _action_requests_from_pending(pending: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return all HITL action requests stored in a pending confirmation payload."""
    requests = pending.get("action_requests") or []
    if requests:
        return list(requests)
    interrupt = pending.get("langgraph_interrupt") or {}
    requests = interrupt.get("action_requests") or []
    if requests:
        return list(requests)
    tool = pending.get("tool")
    if tool:
        return [{
            "name": tool,
            "args": pending.get("args", {}),
            "description": pending.get("message", ""),
        }]
    return []


def _build_langgraph_resume_value(pending: Dict[str, Any], approved: bool) -> Dict[str, Any]:
    """Build a HITL resume payload with one decision per hanging tool call."""
    action_requests = _action_requests_from_pending(pending)
    if not action_requests:
        return {"approved": approved}
    decision: Dict[str, Any] = (
        {"type": "approve"}
        if approved
        else {"type": "reject", "message": _REJECT_MESSAGE}
    )
    return {"decisions": [dict(decision) for _ in action_requests]}


def render_tool_confirmation_ui(pending: Dict[str, Any], project_root: Optional[Path] = None) -> None:
    """Render the confirmation prompt with advanced diff preview and Accept/Reject buttons."""
    if project_root is None:
        project_root = Path.cwd()
    tool = pending.get("tool", "")
    args = pending.get("args", {})
    message = pending.get("message", "Unknown action")
    action_requests = _action_requests_from_pending(pending)

    if pending.get("activity"):
        with st.expander("Agent activity before confirmation", expanded=False):
            st.markdown(pending["activity"])

    interrupt = pending.get("langgraph_interrupt") or {}
    if (
        pending.get("langgraph_interrupt_type") == "engineering_plan_approval"
        or interrupt.get("kind") == "engineering_plan_approval"
        or pending.get("tool") == "engineering_plan_approval"
    ):
        plan = pending.get("engineering_plan") or interrupt.get("engineering_plan") or args.get("engineering_plan")
        plan_text = interrupt.get("plan_text") or pending.get("text") or message
        st.info("**Engineering plan approval** — review create/modify/verify before any edits.")
        if plan_text:
            with st.expander("Engineering plan", expanded=True):
                st.markdown(plan_text)
        if isinstance(plan, dict):
            st.session_state.engineering_plan = plan
            cols = st.columns(3)
            cols[0].metric("Create", len(plan.get("create") or []))
            cols[1].metric("Modify", len(plan.get("modify") or []))
            cols[2].metric("Steps", len(plan.get("steps") or []))
        fp = pending_fingerprint(pending)
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            if st.button(
                "✓ Approve plan",
                key=f"lab_confirm_accept_{fp}",
                type="primary",
                on_click=_set_confirmation_choice,
                args=("approved",),
            ):
                st.session_state.engineering_plan_approved = True
                st.rerun()
        with col2:
            if st.button(
                "✗ Reject plan",
                key=f"lab_confirm_reject_{fp}",
                on_click=_set_confirmation_choice,
                args=("rejected",),
            ):
                st.session_state.engineering_plan_approved = False
                st.rerun()
        st.markdown("---")
        return

    if len(action_requests) > 1:
        st.warning(f"**Agent wants to run {len(action_requests)} tools** (Accept/Reject applies to all)")
        for index, action in enumerate(action_requests, 1):
            name = action.get("name", "tool")
            desc = action.get("description") or f"Run `{name}`"
            st.markdown(f"{index}. **{name}** — {desc}")
        # Diff preview for the first file-modifying action, if any.
        for action in action_requests:
            name = action.get("name", "")
            action_args = action.get("args") or action.get("arguments") or {}
            diff_data = get_diff_for_pending_tool(name, action_args, project_root) if project_root else None
            if diff_data:
                tool = name
                args = action_args
                message = action.get("description") or message
                break
    else:
        st.warning(f"**Agent wants to:** {message}")

    # Advanced diff preview for file operations
    diff_data = get_diff_for_pending_tool(tool, args, project_root) if project_root else None
    if diff_data:
        filename = diff_data.get("filename", "file")
        lang = _infer_language(filename)

        if "diff_text" in diff_data:
            # modify_file or write_file: show unified diff 
            current_content = diff_data.get("current_content", "")
            new_content = diff_data.get("new_content", "")
            diff_text = diff_data.get("diff_text", "")
            mode_hint = diff_data.get("mode", "")
            if mode_hint == "search_replace":
                st.caption("🔍 Surgical edit (search/replace mode)")

            with st.expander(f"📊 Unified diff — {filename}", expanded=True):
                if diff_text and diff_text != "(no changes)":
                    st.code(diff_text, language="diff")
                else:
                    st.info("No changes detected")

            tab1, tab2 = st.tabs(["📄 Current", "✨ Proposed"])
            with tab1:
                if current_content:
                    st.code(current_content, language=lang)
                else:
                    st.info("(new file)")
            with tab2:
                if new_content:
                    st.code(new_content, language=lang)
                else:
                    st.info("(empty)")
        else:
            # create_file: show content preview
            content = diff_data.get("content", "")
            line_count = diff_data.get("line_count", 0)
            if line_count > 0:
                st.caption(f"📊 {line_count} lines")
            if content:
                with st.expander(f"📄 Preview — {filename}", expanded=True):
                    st.code(content, language=lang)

    fp = pending_fingerprint(pending)
    col1, col2, st_space = st.columns([1, 1, 2])
    with col1:
        if st.button(
            "✓ Accept",
            key=f"lab_confirm_accept_{fp}",
            type="primary",
            on_click=_set_confirmation_choice,
            args=("approved",),
        ):
            st.rerun()
    with col2:
        if st.button(
            "✗ Reject",
            key=f"lab_confirm_reject_{fp}",
            on_click=_set_confirmation_choice,
            args=("rejected",),
        ):
            st.rerun()
    # Hint: Retrieve becomes available after Accept for file edits.
    if any(
        (a.get("name") or a.get("tool")) in {
            "write_file", "modify_file", "create_file", "delete_file", "rename_file",
        }
        for a in action_requests
    ):
        st.caption("After Accept, you can **Retrieve** the previous version if you change your mind.")
    st.markdown("---")


def _store_retrieve_snapshot(pending: Dict[str, Any], project_root: Path) -> None:
    """Snapshot file state before approved tool_review actions run."""
    if pending.get("langgraph_interrupt_type") != "tool_review":
        return
    batch = capture_retrieve_batch(_action_requests_from_pending(pending), project_root)
    if not batch:
        return
    stack = list(st.session_state.get("lab_retrieve_stack") or [])
    st.session_state.lab_retrieve_stack = push_retrieve_batch(stack, batch)
    st.session_state.lab_retrieve_panel_dismissed = False
    # Keep legacy slot pointing at the newest single-file batch for older callers.
    entry = batch["entries"][0]
    st.session_state.lab_revert_data = {
        "filepath": entry["filepath"],
        "original_content": entry.get("original_content") or "",
        "file_existed": entry.get("file_existed", True),
        "filename": entry.get("filename"),
        "tool": entry.get("tool"),
        "batch_id": batch["id"],
    }


def handle_tool_confirmation_resume(
    pending: Dict[str, Any],
    choice: str,
    project_root: Path,
    build_session_context: Callable[[], dict],
    *,
    _depth: int = 0,
) -> bool:
    """
    Handle resume after user clicked Accept or Reject.
    Executes the tool (if approved), resumes the agent loop, appends assistant message.
    On Accept of file edits, snapshots originals so the user can Retrieve them.
    Returns True if handled (caller should rerun), False otherwise.
    """
    if not pending or choice not in {"approved", "rejected"}:
        st.session_state.lab_tool_confirmation_choice = None
        return False

    fingerprint = pending_fingerprint(pending)
    # Consume pending immediately so a failed/partial resume cannot re-show the same buttons.
    st.session_state.lab_pending_tool = None
    st.session_state.lab_tool_confirmation_choice = None
    st.session_state.lab_confirm_inflight = {
        "fingerprint": fingerprint,
        "choice": choice,
        "tool": pending.get("tool"),
    }
    _record_decision(fingerprint, choice)

    session_context = build_session_context()
    team = None
    live_activity = LiveActivityRenderer(st.empty())
    response_data: Any = {"text": "Resume failed before the agent returned a result.", "status": "error"}

    try:
        if pending.get("kind") != "langgraph_interrupt":
            raise ValueError("Unsupported pending confirmation payload; expected langgraph_interrupt.")

        approved = choice == "approved"
        interrupt = pending.get("langgraph_interrupt") or {}
        if approved and (
            pending.get("langgraph_interrupt_type") == "engineering_plan_approval"
            or interrupt.get("kind") == "engineering_plan_approval"
        ):
            plan = pending.get("engineering_plan") or interrupt.get("engineering_plan")
            if plan:
                st.session_state.engineering_plan = plan
                session_context["engineering_plan"] = plan
            st.session_state.engineering_plan_approved = True
            session_context["engineering_plan_approved"] = True
        # Capture *before* resume so originals exist on disk.
        if approved:
            try:
                _store_retrieve_snapshot(pending, project_root)
            except Exception:
                pass
        if pending.get("langgraph_interrupt_type") == "tool_review":
            resume_value = _build_langgraph_resume_value(pending, approved)
        else:
            resume_value = {"approved": approved}
        resume_state = {
            "langgraph_thread_id": pending.get("langgraph_thread_id"),
            "langgraph_resume_value": resume_value,
        }

        from agents import UnifiedTeam

        with st.status("Resuming agent workflow...", expanded=True) as status:
            activity_log = st.empty()
            live_activity = LiveActivityRenderer(activity_log)
            team = UnifiedTeam(
                log_callback=live_activity.log,
                stream_callback=live_activity.stream,
                activity_render_callback=lambda *, force=True: live_activity.render(force=force),
                project_root=project_root,
                provider_name=st.session_state.lab_llm_provider,
            )
            response_data = team.run_chat_loop(
                "",
                chat_history=st.session_state.lab_chat_history,
                session_context=session_context,
                resume_state=resume_state,
            )
            live_activity.render(force=True)
            try:
                status.update(label="Done", state="complete", expanded=False)
            except Exception:
                pass
    except Exception as exc:
        response_data = {
            "text": (
                "Agent resume failed or was interrupted. "
                f"Pending confirmation was cleared — click Reset chat if this persists. "
                f"Error: {type(exc).__name__}: {exc}"
            ),
            "status": "error",
        }
        st.session_state.lab_chat_history.append({
            "role": "assistant",
            "content": response_data["text"],
            "activity": live_activity.snapshot(),
        })
        return True
    finally:
        st.session_state.lab_tool_confirmation_choice = None
        st.session_state.lab_confirm_inflight = None
        if team is not None:
            try:
                team.close()
            except Exception:
                pass

    # Another tool may need approval before the workflow can finish.
    if isinstance(response_data, dict) and response_data.get("status") == "pending_confirmation":
        from pages.AutonomousLab.session_sync import sync_context_to_session
        sync_context_to_session(session_context)
        pending_response = dict(response_data)
        pending_response["activity"] = live_activity.snapshot()
        next_fp = pending_fingerprint(pending_response)

        # Same interrupt came back after a decision — break the loop.
        if next_fp == fingerprint:
            if _depth < 1 and choice == "approved":
                # One silent retry can clear flaky resume; then stop.
                return handle_tool_confirmation_resume(
                    pending_response,
                    "approved",
                    project_root,
                    build_session_context,
                    _depth=_depth + 1,
                )
            st.session_state.lab_pending_tool = None
            st.session_state.lab_chat_history.append({
                "role": "assistant",
                "content": (
                    f"Stopped a confirmation loop on `{pending_response.get('tool') or 'tool'}`. "
                    "Your previous Accept/Reject was kept; the same approval request returned again. "
                    "Use **Reset** if the agent is stuck, then retry with a clearer instruction."
                ),
                "activity": live_activity.snapshot(),
            })
            return True

        st.session_state.lab_pending_tool = pending_response
        # If this exact next payload was already decided earlier, auto-apply on next run.
        return True

    # --- Sync agent results to session (see session_sync.py for page-order mappings) ---
    from pages.AutonomousLab.session_sync import sync_context_to_session
    sync_context_to_session(session_context)

    text_content = response_data.get("text", response_data) if isinstance(response_data, dict) else response_data
    artifact = response_data.get("artifact") if isinstance(response_data, dict) else None
    artifacts = response_data.get("artifacts") if isinstance(response_data, dict) else None
    if not artifact and isinstance(artifacts, list) and artifacts:
        artifact = artifacts[-1]

    msg_entry = {
        "role": "assistant",
        "content": text_content,
        "activity": live_activity.snapshot(),
    }
    if artifact and artifact.get("artifact_type") == "plotly_figure":
        try:
            content = store_plotly_artifact_in_message(msg_entry, artifact.get("artifact_content"))
            st.session_state["last_figure_json"] = content
        except Exception:
            pass
    elif artifact and artifact.get("artifact_type") == "downloadable_file":
        try:
            import base64
            b64 = artifact.get("content_base64")
            fname = artifact.get("filename", "download")
            mime = artifact.get("mime_type", "application/octet-stream")
            if b64:
                data = base64.b64decode(b64)
                msg_entry["download"] = {"data": data, "filename": fname, "mime": mime}
        except Exception:
            pass

    st.session_state.lab_chat_history.append(msg_entry)
    return True


def handle_retrieve_action(project_root: Path, *, batch_index: int = -1) -> bool:
    """
    Retrieve (restore) a snapshotted agent file change.
    ``batch_index=-1`` restores the newest batch.
    Returns True if restored (caller should rerun), False otherwise.
    """
    _ = project_root  # entries already store absolute paths
    _migrate_legacy_revert_slot()
    stack = list(st.session_state.get("lab_retrieve_stack") or [])
    if not stack:
        return False
    if batch_index < 0:
        batch_index = len(stack) + batch_index
    if batch_index < 0 or batch_index >= len(stack):
        return False
    batch = stack[batch_index]
    try:
        results = apply_retrieve_batch(batch)
    except Exception as e:
        st.error(f"Retrieve failed: {e}")
        return False
    # Drop the restored batch and everything newer (they may depend on it).
    st.session_state.lab_retrieve_stack = stack[:batch_index]
    st.session_state.lab_revert_data = None
    summary = "; ".join(results[:3])
    if len(results) > 3:
        summary += f" (+{len(results) - 3} more)"
    st.session_state.lab_retrieve_success = f"Retrieved previous version — {summary}"
    st.session_state.lab_revert_success = st.session_state.lab_retrieve_success
    return True


def handle_revert_action(project_root: Path) -> bool:
    """Backward-compatible alias for handle_retrieve_action."""
    return handle_retrieve_action(project_root, batch_index=-1)


def render_retrieve_ui(project_root: Path) -> None:
    """
    Show Retrieve controls after the agent applied file modifications.
    Lets the user restore the pre-accept snapshot (undo Accept).
    """
    _ = project_root
    _migrate_legacy_revert_slot()

    success = st.session_state.get("lab_retrieve_success") or st.session_state.get("lab_revert_success")
    if success:
        st.success(success)
        st.session_state.lab_retrieve_success = None
        st.session_state.lab_revert_success = None

    stack = list(st.session_state.get("lab_retrieve_stack") or [])
    if not stack:
        return

    latest = stack[-1]
    lines = batch_summary_lines(latest)
    count = int(latest.get("count") or len(lines) or 1)
    label = latest.get("label") or (lines[0] if lines else "file")
    dismissed = bool(st.session_state.get("lab_retrieve_panel_dismissed"))

    if not dismissed:
        st.markdown(
            f"""
<div style="
  border: 1px solid rgba(46, 125, 50, 0.35);
  background: linear-gradient(135deg, rgba(46, 125, 50, 0.08), rgba(25, 118, 210, 0.06));
  border-radius: 10px;
  padding: 0.85rem 1rem;
  margin: 0.35rem 0 0.75rem 0;
">
  <div style="font-weight: 600; margin-bottom: 0.25rem;">
    Retrieve previous version
  </div>
  <div style="opacity: 0.9; font-size: 0.92rem;">
    Agent changed <code>{label}</code>
    {" (" + str(count) + " files)" if count > 1 else ""}.
    Restore the state from before you accepted.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        col_btn, col_dismiss, _ = st.columns([1.4, 1, 2])
        with col_btn:
            if st.button(
                "↩ Retrieve previous version",
                key="lab_retrieve_btn",
                type="primary",
                help="Restore files to how they were before the last accepted agent edit",
            ):
                if handle_retrieve_action(Path("."), batch_index=-1):
                    st.rerun()
        with col_dismiss:
            if st.button("Keep changes", key="lab_retrieve_dismiss", type="secondary"):
                st.session_state.lab_retrieve_panel_dismissed = True
                st.rerun()

        if lines:
            with st.expander("What will be restored", expanded=count > 1):
                for line in lines:
                    st.markdown(f"- {line}")
    else:
        st.caption(f"Retrieve available for `{label}` — open history below if you need it.")

    if dismissed or len(stack) > 1:
        title = (
            f"Retrieve history ({len(stack)})"
            if dismissed
            else f"Earlier changes ({len(stack) - 1} more)"
        )
        start = len(stack) - 1 if dismissed else len(stack) - 2
        with st.expander(title, expanded=False):
            for idx in range(start, -1, -1):
                batch = stack[idx]
                blines = batch_summary_lines(batch)
                blabel = batch.get("label") or (blines[0] if blines else f"batch {idx + 1}")
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.caption(blabel + (" · latest" if idx == len(stack) - 1 else ""))
                    for line in blines[:4]:
                        st.markdown(f"- {line}")
                with c2:
                    if st.button(
                        "Retrieve",
                        key=f"lab_retrieve_batch_{batch.get('id', idx)}",
                        help="Also discards newer retrieve snapshots after this one",
                    ):
                        if handle_retrieve_action(Path("."), batch_index=idx):
                            st.rerun()

def render_revert_ui(project_root: Path) -> None:
    """Backward-compatible alias — Retrieve UI (formerly Revert)."""
    render_retrieve_ui(project_root)
