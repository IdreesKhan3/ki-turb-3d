"""
Autonomous Lab — 5 LLM-driven agents with chat interface.
Chat with agents, general writing, or web search.
"""

import streamlit as st
import plotly.io as pio
from pathlib import Path
import sys
import os

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from utils.app_navigation import render_app_navigation_iframe
from utils.multimodal_input import (
    render_multimodal_input,
    get_input_text,
    is_voice_input,
    is_image_input,
)
from agents import UnifiedTeam
from agents.shared.llm_provider import get_llm_provider, get_available_providers

# Import from AutonomousLab module
from pages.AutonomousLab import (
    inject_chat_styles,
    build_session_context,
    init_confirmation_state,
    render_tool_confirmation_ui,
    handle_tool_confirmation_resume,
    render_revert_ui,
)

st.set_page_config(page_title="Autonomous Lab", page_icon="🧠")

# Max artifacts (figures, tables, user images) to remember for agent context
LAB_ARTIFACT_HISTORY_MAX = 15


def init_session_state():
    """Initialize session state for Autonomous Lab."""
    if "lab_chat_history" not in st.session_state:
        st.session_state.lab_chat_history = []
    if "lab_llm_provider" not in st.session_state:
        st.session_state.lab_llm_provider = "gemini"
    if "lab_artifact_history" not in st.session_state:
        st.session_state.lab_artifact_history = []
    if "lab_agent_data_cache" not in st.session_state:
        st.session_state.lab_agent_data_cache = {}
    init_confirmation_state()
    # Ensure exports directory exists so saved figures have a place to go
    (project_root / "exports").mkdir(exist_ok=True)


def render_sidebar():
    """Render sidebar with LLM provider selection and examples."""
    st.subheader("LLM Provider")

    provider_name = st.session_state.lab_llm_provider
    available_providers = get_available_providers()
    provider_options = []
    provider_status = {}

    provider_order = ["gemini", "ollama"]
    provider_labels = ["Gemini", "Ollama"]
    for provider in provider_order:
        provider_status[provider] = available_providers.get(provider, False)

    selected_label = st.radio(
        "Select Provider",
        options=provider_labels,
        index=provider_order.index(provider_name) if provider_name in provider_order else 0,
        key="lab_llm_provider_radio",
        help="Gemini (cloud). Ollama (local).",
    )
    selected_provider = provider_order[provider_labels.index(selected_label)]

    if selected_provider != provider_name:
        st.session_state.lab_llm_provider = selected_provider
        if "lab_team" in st.session_state:
            del st.session_state.lab_team
        st.info(f"Switched to {selected_provider.title()}.")

    if provider_status.get(selected_provider, False):
        st.success("Ready")
        if selected_provider == "ollama":
            try:
                llm = get_llm_provider("ollama")
                model_name = llm.model if hasattr(llm, "model") else "Unknown"
                st.caption(f"Model: **{model_name}**")
            except Exception:
                st.caption("Model: Not detected")
        if selected_provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
                st.caption(f"API Key: {masked}")
            else:
                st.warning("GOOGLE_API_KEY not found")
    else:
        if selected_provider == "ollama":
            st.warning("Ollama not running. Install: curl -fsSL https://ollama.com/install.sh | sh")
        elif selected_provider == "gemini":
            st.warning("Set GOOGLE_API_KEY environment variable.")

def render_chat_history():
    """Render chat history (text, figures, tables)."""
    st.markdown("---")
    st.subheader("Conversation")

    if not st.session_state.lab_chat_history:
        st.info("Start a conversation. Chat with agents, request writing, or web search.")
    else:
        for i, msg in enumerate(st.session_state.lab_chat_history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="♾️"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🌌"):
                    st.write(msg["content"])
                    # Multiple figures (multi-task response)
                    figures = msg.get("figures") or ([msg["figure"]] if msg.get("figure") else [])
                    for j, fig in enumerate(figures):
                        st.plotly_chart(fig, use_container_width=True, key=f"lab_fig_{i}_{j}")
                    # Multiple tables (clearly labeled so they are visible)
                    tables = msg.get("tables") or []
                    for j, table_md in enumerate(tables):
                        st.markdown("---")
                        st.markdown(f"**Summary table** {j + 1}" if len(tables) > 1 else "**Summary table**")
                        st.markdown(table_md)
                    # Markdown blocks (e.g. Theory & Equations)
                    markdown_blocks = msg.get("markdown_blocks") or []
                    for j, block in enumerate(markdown_blocks):
                        st.markdown("---")
                        st.markdown(f"**{block.get('title', 'Theory & Equations')}**")
                        st.markdown(block.get("content", ""))
                    # Report HTML blocks (compiled report preview — rendered like PDF/HTML)
                    # Use st.html (no iframe) when available to avoid duplicate sidebar; wrap in container for scrolling
                    report_html_blocks = msg.get("report_html_blocks") or []
                    for j, block in enumerate(report_html_blocks):
                        st.markdown("---")
                        st.markdown(f"**{block.get('title', 'Report Preview')}**")
                        html_content = block.get("content", "")
                        try:
                            with st.container(height=800):
                                st.html(html_content)
                        except AttributeError:
                            st.components.v1.html(html_content, height=800, scrolling=True)
                    if msg.get("download"):
                        d = msg["download"]
                        st.download_button(
                            f"Download {d.get('filename', 'file')}",
                            data=d["data"],
                            file_name=d.get("filename", "download"),
                            mime=d.get("mime", "application/octet-stream"),
                            key=f"lab_dl_{i}",
                        )


def main():
    init_session_state()
    inject_theme_css()
    inject_chat_styles()

    st.title("Autonomous Lab")
    st.markdown(
        "Five agents: Orchestrator, Data Steward, Analyst, Visualizer, Reviewer. "
        "Chat with agents, ask questions, clear doubts, or request plots and analysis."
    )

    # Browse App Pages — main area so it's visible (sidebar was too cramped)
    render_app_navigation_iframe(key_prefix="lab", in_sidebar=False)

    with st.sidebar:
        render_sidebar()

    render_chat_history()

    # Session context builder (uses main app data_directory from session state)
    def _session_context():
        return build_session_context()

    # Handle resume from tool confirmation (user clicked Accept or Reject)
    pending = st.session_state.lab_pending_tool
    choice = st.session_state.lab_tool_confirmation_choice
    if pending and choice:
        if handle_tool_confirmation_resume(pending, choice, project_root, _session_context):
            st.rerun()

    # Show confirmation UI when pending tool exists (no choice yet)
    if pending and not choice:
        render_tool_confirmation_ui(pending, project_root)

    # Revert UI: show after accepting a modification (similar to edit-undo UX)
    render_revert_ui(project_root)

    st.markdown("---")
    st.caption("AI responses may contain errors. Verify critical calculations and code before use.")

    example_query = st.session_state.get("lab_example_query", "")
    input_data = render_multimodal_input(
        key="lab_chat_input",
        placeholder="Chat with agents, write, or search the web",
        default_value=example_query,
        enabled_modes=["text", "voice", "image"],
        voice_enabled=True,
    )

    user_input = get_input_text(input_data) if input_data else ""
    if example_query:
        del st.session_state.lab_example_query

    if input_data:
        if is_voice_input(input_data):
            st.caption("Voice input")
        elif is_image_input(input_data):
            st.caption("Image uploaded")
            files = input_data.get("files", [])
            if files:
                try:
                    import base64
                    for file in files[:1]:
                        if "data" in file:
                            base64_data = file["data"]
                            if "," in base64_data:
                                base64_data = base64_data.split(",", 1)[1]
                            image_bytes = base64.b64decode(base64_data)
                            st.image(image_bytes, caption=file.get("name", "Uploaded image"), width=300)
                except Exception as e:
                    st.caption(f"Could not display image: {str(e)}")

    has_text = bool(user_input and user_input.strip())
    has_image = bool(input_data and is_image_input(input_data))

    # Skip normal flow when waiting for tool confirmation
    waiting_confirmation = bool(st.session_state.lab_pending_tool and not st.session_state.lab_tool_confirmation_choice)
    if input_data is not None and (has_text or has_image) and not waiting_confirmation:
        if not user_input or not user_input.strip():
            user_input = "[Image uploaded]" if has_image else ""

        if user_input:
            st.session_state.lab_chat_history.append({"role": "user", "content": user_input})

            # Remember user-uploaded images so agents can see and explain them
            if has_image and input_data and input_data.get("files"):
                try:
                    import base64
                    from utils.image_processor import _decode_base64_robust, _infer_mime_type_from_filename
                    file = input_data["files"][0]
                    raw = file.get("data", "")
                    if raw:
                        if "," in raw and raw.startswith("data:"):
                            from utils.image_processor import _parse_data_uri
                            mime_type, b64 = _parse_data_uri(raw)
                            image_bytes = _decode_base64_robust(b64)
                        else:
                            mime_type = _infer_mime_type_from_filename(file.get("name", ""))
                            image_bytes = _decode_base64_robust(raw)
                        entry = {
                            "type": "image",
                            "figure_image": {"mime_type": mime_type, "data": image_bytes},
                            "caption": (user_input or "User uploaded image")[:200],
                        }
                        st.session_state.lab_artifact_history.append(entry)
                        st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                except Exception:
                    pass

            logs = []
            provider = st.session_state.lab_llm_provider

            def collect_log(msg):
                logs.append(msg)
                st.markdown(msg)

            team = UnifiedTeam(
                log_callback=collect_log,
                project_root=project_root,
                provider_name=provider,
            )

            with st.status("🤖 Agents working...", expanded=True) as status:
                try:
                    chat_history = st.session_state.lab_chat_history[:-1]
                    session_context = _session_context()
                    response_data = team.run_chat_loop(
                        user_input,
                        chat_history=chat_history,
                        session_context=session_context,
                    )

                    # Pending confirmation: store and show UI on next render
                    if isinstance(response_data, dict) and response_data.get("status") == "pending_confirmation":
                        st.session_state.lab_pending_tool = response_data
                        st.rerun()

                    text_content = response_data.get("text", response_data) if isinstance(response_data, dict) else response_data
                    # Multi-task: support both single artifact and list of artifacts
                    artifacts = response_data.get("artifacts") if isinstance(response_data, dict) else None
                    if not artifacts and isinstance(response_data, dict) and response_data.get("artifact"):
                        artifacts = [response_data["artifact"]]
                    artifact = artifacts[-1] if artifacts else None

                    # Sync agent results to session (see session_sync.py for page-order mappings)
                    from pages.AutonomousLab.session_sync import sync_context_to_session
                    sync_context_to_session(session_context)

                    msg_entry = {"role": "assistant", "content": text_content}

                    # Fallback: user asked to see report but agent returned outline. Show compiled report anyway
                    # (same as manual Report page: figures, tables, text rendered)
                    _show_report_kw = ("show report", "see the report", "display report", "preview report",
                                        "compiled report", "full report", "show me the report", "how it looks",
                                        "report with figures", "report with tables")
                    _ctx_for_preview = _session_context()  # Use synced session (report_sections from agent run)
                    if (any(kw in user_input.lower() for kw in _show_report_kw)
                            and not any(a.get("artifact_type") == "report_html" for a in (artifacts or []))
                            and _ctx_for_preview.get("report_sections")):
                        try:
                            from agents.tools import execute_tool
                            prev = execute_tool("preview_report", {}, project_root, _ctx_for_preview)
                            if isinstance(prev, dict) and prev.get("artifact_type") == "report_html":
                                msg_entry.setdefault("report_html_blocks", []).append({
                                    "content": prev.get("artifact_content", ""),
                                    "title": "Report Preview",
                                })
                                msg_entry["content"] = prev.get("message", "Report preview (compiled form).")
                        except Exception:
                            pass
                    msg_figures = []
                    msg_tables = []
                    msg_markdown_blocks = []
                    for a in (artifacts or []):
                        if a.get("artifact_type") == "plotly_figure":
                            try:
                                import json
                                from utils.image_processor import plotly_figure_to_image_dict, extract_figure_data_for_agent
                                content = a.get("artifact_content")
                                if isinstance(content, dict):
                                    content = json.dumps(content)
                                fig = pio.from_json(content)
                                msg_figures.append(fig)
                                st.session_state["last_figure_json"] = content
                                img_dict = plotly_figure_to_image_dict(fig)
                                fig_data_str = extract_figure_data_for_agent(fig)
                                if img_dict or fig_data_str:
                                    st.session_state.lab_artifact_history.append({
                                        "type": "figure",
                                        "figure_image": img_dict,
                                        "figure_data": fig_data_str,
                                        "caption": (text_content or "")[:200],
                                        "source_file": a.get("source_file"),
                                        "tool_name": a.get("tool_name"),
                                    })
                                    st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                            except Exception as ex:
                                st.caption(f"Could not render figure: {ex}")
                        elif a.get("artifact_type") == "markdown_table":
                            table_md = a.get("artifact_content") or ""
                            if table_md:
                                msg_tables.append(table_md)
                                st.session_state.lab_artifact_history.append({
                                    "type": "table",
                                    "table_md": table_md,
                                    "caption": (text_content or "")[:200],
                                })
                                st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                        elif a.get("artifact_type") == "markdown":
                            content = a.get("artifact_content") or ""
                            title = a.get("artifact_title", "Theory & Equations")
                            if content:
                                msg_markdown_blocks.append({"content": content, "title": title})
                                st.session_state.lab_artifact_history.append({
                                    "type": "markdown",
                                    "content": content,
                                    "title": title,
                                    "caption": (text_content or "")[:200],
                                })
                                st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                        elif a.get("artifact_type") == "report_html":
                            html_content = a.get("artifact_content") or ""
                            title = a.get("artifact_title", "Report Preview")
                            if html_content:
                                msg_entry.setdefault("report_html_blocks", []).append({"content": html_content, "title": title})
                        elif a.get("artifact_type") == "downloadable_file" and not msg_entry.get("download"):
                            try:
                                import base64
                                b64 = a.get("content_base64")
                                fname = a.get("filename", "download")
                                mime = a.get("mime_type", "application/octet-stream")
                                if b64:
                                    data = base64.b64decode(b64)
                                    msg_entry["download"] = {"data": data, "filename": fname, "mime": mime}
                            except Exception as ex:
                                st.caption(f"Could not prepare download: {ex}")
                    if msg_figures:
                        msg_entry["figures"] = msg_figures
                        if len(msg_figures) == 1:
                            msg_entry["figure"] = msg_figures[0]
                    if msg_tables:
                        msg_entry["tables"] = msg_tables
                    if msg_markdown_blocks:
                        msg_entry["markdown_blocks"] = msg_markdown_blocks
                    # Fallback: if user asked to save/export and agent didn't call export_figure, do it here
                    # Only trigger on explicit save/export/download intent — NOT on "pdf" from "PDFs page" or "Dissipation Rate PDF" (probability density)
                    if not msg_entry.get("download") and "last_figure_json" in st.session_state:
                        save_intent = any(kw in user_input.lower() for kw in ("save", "export", "download"))
                        if save_intent:
                            try:
                                import base64
                                fig = pio.from_json(st.session_state["last_figure_json"])
                                fmt = "png"
                                ul = user_input.lower()
                                # Use "pdf" only when it clearly means file format, not probability density (PDFs page)
                                pdf_as_format = any(p in ul for p in ("save as pdf", "export to pdf", "export as pdf", "to pdf", "as pdf", "pdf format", "download pdf"))
                                pdfs_page_context = any(p in ul for p in ("pdfs page", "pdf page", "dissipation pdf", "vorticity pdf", "velocity pdf", "enstrophy pdf", "joint pdf", "r-q", "probability density"))
                                if pdf_as_format and not pdfs_page_context:
                                    fmt = "pdf"
                                elif "svg" in ul and not pdfs_page_context:
                                    fmt = "svg"
                                elif "html" in ul:
                                    fmt = "html"
                                fname = f"figure.{fmt}"
                                out_dir = project_root / "exports"
                                out_dir.mkdir(exist_ok=True)
                                out_path = out_dir / fname
                                if fmt == "html":
                                    content = fig.to_html().encode("utf-8")
                                    mime = "text/html"
                                else:
                                    content = fig.to_image(format="jpeg" if fmt == "jpg" else fmt, scale=2)
                                    mime = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml", "jpeg": "image/jpeg"}.get(fmt, "application/octet-stream")
                                out_path.write_bytes(content)
                                b64 = base64.b64encode(content).decode("ascii")
                                msg_entry["download"] = {"data": content, "filename": fname, "mime": mime}
                                msg_entry["content"] = (msg_entry.get("content", "") or "").strip()
                                if msg_entry["content"]:
                                    msg_entry["content"] += f"\n\nFigure saved to `exports/{fname}`."
                                else:
                                    msg_entry["content"] = f"Figure saved to `exports/{fname}`. Use the download button below."
                            except Exception as ex:
                                msg_entry["content"] = (msg_entry.get("content", "") or "") + f"\n\n(Could not auto-export: {ex})"
                    st.session_state.lab_chat_history.append(msg_entry)
                    try:
                        status.update(label="✅ Done", state="complete")
                    except Exception:
                        pass
                except Exception as e:
                    st.session_state.lab_chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                    })
                    try:
                        status.update(label="❌ Error", state="error")
                    except Exception:
                        pass

            st.rerun()


if __name__ == "__main__":
    main()
