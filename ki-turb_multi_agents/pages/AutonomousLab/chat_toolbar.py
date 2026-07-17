"""
Compact chat session controls — Copy / Export / Reset.

"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import streamlit as st


def format_chat_transcript(history: List[Dict[str, Any]]) -> str:
    """Plain-text transcript of the current conversation (copy/export)."""
    lines: List[str] = [
        "KI-TURB Autonomous Lab — conversation transcript",
        f"Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    for i, msg in enumerate(history or [], 1):
        role = str(msg.get("role") or "unknown").strip().lower()
        label = "User" if role == "user" else "Assistant"
        content = str(msg.get("content") or "").rstrip()
        lines.append(f"—— {i}. {label} ——")
        lines.append(content or "(empty)")
        if msg.get("activity"):
            lines.append("")
            lines.append("[Agent activity]")
            lines.append(str(msg["activity"]).rstrip())
        tables = msg.get("tables") or []
        for j, table_md in enumerate(tables, 1):
            lines.append("")
            lines.append(f"[Table {j}]")
            lines.append(str(table_md).rstrip())
        for j, block in enumerate(msg.get("markdown_blocks") or [], 1):
            lines.append("")
            title = block.get("title") or f"Markdown {j}"
            lines.append(f"[{title}]")
            lines.append(str(block.get("content") or "").rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def reset_lab_chat() -> None:
    """Clear conversation and related confirmation/retrieve state (keep LLM provider)."""
    st.session_state.lab_chat_history = []
    st.session_state.lab_pending_tool = None
    st.session_state.lab_tool_confirmation_choice = None
    st.session_state.lab_confirm_decisions = {}
    st.session_state.lab_confirm_inflight = None
    st.session_state.lab_retrieve_stack = []
    st.session_state.lab_revert_data = None
    st.session_state.lab_retrieve_success = None
    st.session_state.lab_revert_success = None
    st.session_state.lab_retrieve_panel_dismissed = False
    st.session_state.lab_artifact_history = []
    st.session_state.lab_agent_data_cache = {}
    for key in (
        "engineering_plan",
        "engineering_step_index",
        "engineering_plan_approved",
        "engineering_intent",
        "lab_team",
        "lab_example_query",
        "_lab_show_copy_panel",
        "_lab_copy_transcript",
        "_lab_confirm_reset",
        "_lab_do_clipboard_copy",
    ):
        if key in st.session_state:
            del st.session_state[key]


def render_chat_toolbar() -> None:
    """Compact action strip placed directly above the compose input."""
    history = list(st.session_state.get("lab_chat_history") or [])
    n_msgs = len(history)
    empty = n_msgs == 0
    transcript = format_chat_transcript(history) if not empty else ""

    st.markdown(
        f"""
<div class="lab-compose-actions">
  <span class="lab-compose-actions__label">Chat tools</span>
  <span class="lab-compose-actions__count">{n_msgs} messages</span>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 4], gap="small")
    with c1:
        if st.button(
            "Copy",
            key="lab_chat_copy_all",
            type="secondary",
            disabled=empty,
            use_container_width=True,
            help="Show transcript with copy control",
        ):
            st.session_state["_lab_show_copy_panel"] = True
            st.session_state["_lab_copy_transcript"] = transcript
            st.rerun()
    with c2:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Export",
            data=transcript.encode("utf-8") if transcript else b"",
            file_name=f"kiturb_lab_chat_{stamp}.txt",
            mime="text/plain",
            key="lab_chat_export_txt",
            disabled=empty,
            use_container_width=True,
            help="Download conversation as .txt",
        )
    with c3:
        if st.button(
            "Reset",
            key="lab_chat_reset",
            type="secondary",
            use_container_width=True,
            help="Clear chat, approvals, and retrieve history",
        ):
            st.session_state["_lab_confirm_reset"] = True
            st.rerun()

    if st.session_state.get("_lab_show_copy_panel") and st.session_state.get("_lab_copy_transcript"):
        with st.expander("Transcript — use the copy icon", expanded=True):
            st.code(st.session_state["_lab_copy_transcript"], language="markdown")
            if st.button("Hide transcript", key="lab_chat_hide_copy"):
                st.session_state["_lab_show_copy_panel"] = False
                st.rerun()

    if st.session_state.get("_lab_confirm_reset"):
        st.markdown(
            """
<div class="lab-reset-confirm">
  <strong>Reset this session?</strong>
  <span>Clears messages, pending Accept/Reject, and Retrieve. Provider kept.</span>
</div>
""",
            unsafe_allow_html=True,
        )
        a, b, _ = st.columns([1, 1, 4], gap="small")
        with a:
            if st.button("Confirm reset", key="lab_chat_reset_yes", type="primary", use_container_width=True):
                reset_lab_chat()
                st.rerun()
        with b:
            if st.button("Cancel", key="lab_chat_reset_no", use_container_width=True):
                st.session_state.pop("_lab_confirm_reset", None)
                st.rerun()
