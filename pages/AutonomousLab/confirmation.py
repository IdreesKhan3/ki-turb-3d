"""
Tool confirmation UI for Autonomous Lab.
Shows Accept/Reject buttons with advanced diff preview when the agent requests
file modifications (modify_file, write_file, create_file).
"""

import streamlit as st
import plotly.io as pio
from pathlib import Path
from typing import Callable, Dict, Any, Optional

from .diff_preview import get_diff_for_pending_tool


def init_confirmation_state():
    """Initialize session state keys for tool confirmation flow."""
    if "lab_pending_tool" not in st.session_state:
        st.session_state.lab_pending_tool = None
    if "lab_tool_confirmation_choice" not in st.session_state:
        st.session_state.lab_tool_confirmation_choice = None
    if "lab_revert_data" not in st.session_state:
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


def render_tool_confirmation_ui(pending: Dict[str, Any], project_root: Optional[Path] = None) -> None:
    """Render the confirmation prompt with advanced diff preview and Accept/Reject buttons."""
    if project_root is None:
        project_root = Path.cwd()
    tool = pending.get("tool", "")
    args = pending.get("args", {})
    message = pending.get("message", "Unknown action")

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

    col1, col2, st_space = st.columns([1, 1, 2])
    with col1:
        if st.button("✓ Accept", key="lab_confirm_accept", type="primary"):
            st.session_state.lab_tool_confirmation_choice = "approved"
            st.rerun()
    with col2:
        if st.button("✗ Reject", key="lab_confirm_reject"):
            st.session_state.lab_tool_confirmation_choice = "rejected"
            st.rerun()
    st.markdown("---")


def _resolve_filepath(filepath: str, project_root: Path) -> Path:
    """Resolve filepath relative to project root."""
    p = Path(filepath)
    if not p.is_absolute():
        p = project_root / p
    return p


def _capture_revert_data(
    tool: str,
    args: Dict[str, Any],
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    Capture original state before applying a file modification, for revert support.
    Returns dict with filepath, original_content, file_existed, filename; or None if not revertable.
    """
    from .diff_preview import get_diff_for_pending_tool
    filepath = args.get("filepath", "")
    if not filepath:
        return None
    path = _resolve_filepath(filepath, project_root)
    diff_data = get_diff_for_pending_tool(tool, args, project_root)
    if not diff_data:
        return None
    if tool == "modify_file":
        original_content = diff_data.get("current_content", "")
        return {
            "filepath": str(path),
            "original_content": original_content,
            "file_existed": True,
            "filename": diff_data.get("filename", path.name),
            "tool": tool,
        }
    if tool == "write_file":
        original_content = diff_data.get("current_content", "")
        file_existed = path.exists() and path.is_file() if path else False
        return {
            "filepath": str(path),
            "original_content": original_content,
            "file_existed": file_existed,
            "filename": diff_data.get("filename", path.name),
            "tool": tool,
        }
    if tool == "create_file":
        return {
            "filepath": str(path),
            "original_content": "",
            "file_existed": False,
            "filename": diff_data.get("filename", path.name),
            "tool": tool,
        }
    return None


def handle_tool_confirmation_resume(
    pending: Dict[str, Any],
    choice: str,
    project_root: Path,
    build_session_context: Callable[[], dict],
) -> bool:
    """
    Handle resume after user clicked Accept or Reject.
    Executes the tool (if approved), resumes the agent loop, appends assistant message.
    Stores revert data on successful Accept for modify_file, write_file, create_file.
    Returns True if handled (caller should rerun), False otherwise.
    """
    session_context = build_session_context()
    if choice == "approved":
        tool_name = pending.get("tool", "")
        args = pending.get("args", {})
        revert_data = _capture_revert_data(tool_name, args, project_root)
        session_context["tool_confirmation_approved"] = True
        from agents.tools import execute_tool
        tool_result = execute_tool(
            tool_name, args, project_root,
            session_context=session_context,
        )
        if revert_data and "Error" not in str(tool_result):
            st.session_state.lab_revert_data = revert_data
    else:
        tool_result = "User rejected the operation."

    resume_state = {
        "messages": pending.get("messages", []),
        "last_assistant_response": pending.get("last_assistant_response", ""),
        "tool_result": str(tool_result),
    }

    logs = []
    from agents import UnifiedTeam
    team = UnifiedTeam(
        log_callback=lambda m: logs.append(m) or st.markdown(m),
        project_root=project_root,
        provider_name=st.session_state.lab_llm_provider,
    )
    with st.spinner("Resuming..."):
        response_data = team.run_chat_loop(
            "",
            chat_history=st.session_state.lab_chat_history,
            session_context=session_context,
            resume_state=resume_state,
        )

    st.session_state.lab_pending_tool = None
    st.session_state.lab_tool_confirmation_choice = None
    if session_context.get("plot_styles"):
        st.session_state.plot_styles.update(session_context["plot_styles"])

    text_content = response_data.get("text", response_data) if isinstance(response_data, dict) else response_data
    artifact = response_data.get("artifact") if isinstance(response_data, dict) else None

    msg_entry = {"role": "assistant", "content": text_content}
    if artifact and artifact.get("artifact_type") == "plotly_figure":
        try:
            import json
            content = artifact.get("artifact_content")
            if isinstance(content, dict):
                content = json.dumps(content)
            fig = pio.from_json(content)
            msg_entry["figure"] = fig
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


def handle_revert_action(project_root: Path) -> bool:
    """
    Revert the last applied file modification.
    Restores original content (or deletes file if it was newly created).
    Returns True if reverted (caller should rerun), False otherwise.
    """
    revert_data = st.session_state.get("lab_revert_data")
    if not revert_data:
        return False
    filepath = Path(revert_data["filepath"])
    original_content = revert_data.get("original_content", "")
    file_existed = revert_data.get("file_existed", True)
    filename = revert_data.get("filename", filepath.name)
    try:
        if file_existed:
            filepath.write_text(original_content, encoding="utf-8")
        else:
            if filepath.exists() and filepath.is_file():
                filepath.unlink()
    except Exception as e:
        st.error(f"Revert failed: {e}")
        return False
    st.session_state.lab_revert_data = None
    st.session_state.lab_revert_success = f"↩ Reverted `{filename}`"
    return True


def render_revert_ui(project_root: Path) -> None:
    """
    Render the Revert button when a revertable modification was just applied.
    Shows a prominent '↩ Revert last change' button (similar to edit-undo UX).
    """
    if st.session_state.get("lab_revert_success"):
        st.success(st.session_state.lab_revert_success)
        st.session_state.lab_revert_success = None
    revert_data = st.session_state.get("lab_revert_data")
    if not revert_data:
        return
    filename = revert_data.get("filename", "file")
    st.info(f"Last change: `{filename}` — you can revert it.")
    if st.button("↩ Revert last change", key="lab_revert_btn", type="secondary"):
        if handle_revert_action(project_root):
            st.rerun()
