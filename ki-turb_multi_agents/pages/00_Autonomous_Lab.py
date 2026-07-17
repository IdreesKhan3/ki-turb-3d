"""
Autonomous Lab — seven-role LangGraph team with chat interface.

Roles: Orchestrator, Data Steward, Simulation, Analyst, Visualizer, Reviewer, Engineer.
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
from pages.AutonomousLab.multimodal_input import (
    render_multimodal_input,
    get_input_text,
    is_voice_input,
    is_image_input,
)
from agents import UnifiedTeam
from agents.shared.llm_provider import get_llm_provider, get_available_providers
from pages.AutonomousLab.live_activity import LiveActivityRenderer

from pages.AutonomousLab import (
    inject_chat_styles,
    build_session_context,
    init_confirmation_state,
    render_tool_confirmation_ui,
    handle_tool_confirmation_resume,
    render_revert_ui,
    render_simulation_workflow,
)
from pages.AutonomousLab.confirmation import auto_choice_for_duplicate_pending
from pages.AutonomousLab.chat_figures import figures_from_message, store_plotly_artifact_in_message
from pages.AutonomousLab.chat_toolbar import render_chat_toolbar

st.set_page_config(page_title="Autonomous Lab", page_icon="🧠")

# Max artifacts (figures, tables, user images) to remember for agent context
LAB_ARTIFACT_HISTORY_MAX = 15


def init_session_state():
    """Initialize session state for Autonomous Lab."""
    if "lab_chat_history" not in st.session_state:
        st.session_state.lab_chat_history = []
    if "lab_llm_provider" not in st.session_state:
        st.session_state.lab_llm_provider = "deepseek"
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
    provider_status = {}

    provider_order = ["deepseek", "gemini", "ollama"]
    provider_labels = ["DeepSeek", "Gemini", "Ollama"]
    for provider in provider_order:
        provider_status[provider] = available_providers.get(provider, False)

    selected_label = st.radio(
        "Select Provider",
        options=provider_labels,
        index=provider_order.index(provider_name) if provider_name in provider_order else 0,
        key="lab_llm_provider_radio",
        help="DeepSeek (cloud, default). Gemini (cloud). Ollama (local).",
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
            if not os.getenv("GOOGLE_API_KEY"):
                st.warning("GOOGLE_API_KEY not found")
        if selected_provider == "deepseek":
            try:
                llm = get_llm_provider("deepseek")
                model_name = llm.model if hasattr(llm, "model") else "Unknown"
                st.caption(f"Model: **{model_name}**")
            except Exception:
                st.caption("Model: Not detected")
            if not os.getenv("DEEPSEEK_API_KEY"):
                st.warning("DEEPSEEK_API_KEY not found")
    else:
        if selected_provider == "ollama":
            st.warning("Ollama not running. Install: curl -fsSL https://ollama.com/install.sh | sh")
        elif selected_provider == "gemini":
            st.warning("Set GOOGLE_API_KEY environment variable.")
        elif selected_provider == "deepseek":
            st.warning("Set DEEPSEEK_API_KEY environment variable.")

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
                    for j, img in enumerate(msg.get("image_files") or []):
                        data = img.get("data")
                        if data is not None:
                            st.image(
                                data,
                                caption=img.get("caption") or "Attached image",
                                use_container_width=True,
                            )
            else:
                with st.chat_message("assistant", avatar="🌌"):
                    st.write(msg["content"])
                    if msg.get("activity"):
                        with st.expander("Agent activity", expanded=False):
                            st.markdown(msg["activity"])
                    # Multiple figures (multi-task response)
                    for j, fig in enumerate(figures_from_message(msg)):
                        st.plotly_chart(fig, use_container_width=True, key=f"lab_fig_{i}_{j}")
                    # Image files from read_document (PNG, JPG, etc. from disk)
                    image_files = msg.get("image_files") or []
                    for j, img in enumerate(image_files):
                        st.markdown("---")
                        st.caption(img.get("caption", "Image from disk"))
                        st.image(img.get("data"), use_container_width=True)
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
        "Seven agents: Orchestrator, Data Steward, Simulation, Analyst, Visualizer, "
        "Reviewer, and Engineer. Run solvers, analyze data, plot results, edit files, "
        "or improve the KI-TURB product itself."
    )

    render_app_navigation_iframe(key_prefix="lab", in_sidebar=False)

    with st.sidebar:
        render_sidebar()
        st.markdown("---")
        render_simulation_workflow(project_root)

    render_chat_history()

    # Session context builder (uses main app data_directory from session state)
    def _session_context():
        return build_session_context()

    # Handle resume from tool confirmation (user clicked Accept or Reject).
    # Re-read session keys each step so we never use stale locals after consume-once.
    pending = st.session_state.lab_pending_tool
    choice = st.session_state.lab_tool_confirmation_choice
    if pending and not choice:
        auto_choice = auto_choice_for_duplicate_pending(pending)
        if auto_choice:
            st.session_state.lab_tool_confirmation_choice = auto_choice
            choice = auto_choice
    if pending and choice:
        if handle_tool_confirmation_resume(pending, choice, project_root, _session_context):
            st.rerun()
        return

    pending = st.session_state.lab_pending_tool
    choice = st.session_state.lab_tool_confirmation_choice
    # Show confirmation UI when pending tool exists (no choice yet)
    if pending and not choice:
        render_tool_confirmation_ui(pending, project_root)

    # Retrieve UI only when not mid-confirmation (avoid overlapping Accept UIs)
    if not (pending and not choice):
        render_revert_ui(project_root)

    st.markdown("---")
    st.caption("AI responses may contain errors. Verify critical calculations and code before use.")

    render_chat_toolbar()

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
            turn_images: list = []
            user_msg: dict = {"role": "user", "content": user_input}

            if has_image and input_data and input_data.get("files"):
                try:
                    from agents.shared.image_processor import (
                        _decode_base64_robust,
                        _infer_mime_type_from_filename,
                        _parse_data_uri,
                        normalize_image_dict,
                    )
                    for file in (input_data.get("files") or [])[:3]:
                        raw = file.get("data", "")
                        if not raw:
                            continue
                        if isinstance(raw, str) and raw.startswith("data:") and "," in raw:
                            mime_type, b64 = _parse_data_uri(raw)
                            image_bytes = _decode_base64_robust(b64)
                        else:
                            mime_type = _infer_mime_type_from_filename(file.get("name", ""))
                            image_bytes = _decode_base64_robust(raw) if isinstance(raw, str) else bytes(raw)
                        img_dict = normalize_image_dict({"mime_type": mime_type, "data": image_bytes})
                        if not img_dict:
                            continue
                        turn_images.append(img_dict)
                        entry = {
                            "type": "image",
                            "figure_image": img_dict,
                            "caption": (user_input or file.get("name") or "User uploaded image")[:200],
                        }
                        st.session_state.lab_artifact_history.append(entry)
                    st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                    if turn_images:
                        src_files = list(input_data.get("files") or [])
                        user_msg["image_files"] = []
                        for i, img in enumerate(turn_images):
                            name = "Attached image"
                            if i < len(src_files) and isinstance(src_files[i], dict):
                                name = str(src_files[i].get("name") or name)
                            user_msg["image_files"].append({
                                "data": img["data"],
                                "caption": name,
                                "mime_type": img["mime_type"],
                            })
                        if user_input.strip() in {"", "[Image uploaded]"}:
                            user_msg["content"] = (
                                "Please analyze the attached image(s) and explain what you see "
                                "(figures, equations, UI, or errors)."
                            )
                            user_input = user_msg["content"]
                except Exception:
                    turn_images = []

            st.session_state.lab_chat_history.append(user_msg)

            provider = st.session_state.lab_llm_provider

            with st.status("Running agent workflow...", expanded=True) as status:
                activity_log = st.empty()
                live_activity = LiveActivityRenderer(activity_log)
                team = UnifiedTeam(
                    log_callback=live_activity.log,
                    stream_callback=live_activity.stream,
                    activity_render_callback=lambda *, force=True: live_activity.render(force=force),
                    project_root=project_root,
                    provider_name=provider,
                )
                try:
                    chat_history = st.session_state.lab_chat_history[:-1]
                    session_context = _session_context()
                    if turn_images:
                        session_context["turn_images"] = turn_images
                    response_data = team.run_chat_loop(
                        user_input,
                        chat_history=chat_history,
                        session_context=session_context,
                    )

                    # Pending confirmation: store and show UI on next render
                    if isinstance(response_data, dict) and response_data.get("status") == "pending_confirmation":
                        from pages.AutonomousLab.session_sync import sync_context_to_session
                        sync_context_to_session(session_context)
                        pending_response = dict(response_data)
                        pending_response["activity"] = live_activity.snapshot()
                        st.session_state.lab_pending_tool = pending_response
                        st.rerun()

                    text_content = response_data.get("text", response_data) if isinstance(response_data, dict) else response_data
                    # Multi-task: support both single artifact and list of artifacts
                    artifacts = response_data.get("artifacts") if isinstance(response_data, dict) else None
                    if not artifacts and isinstance(response_data, dict) and response_data.get("artifact"):
                        artifacts = [response_data["artifact"]]

                    # Sync agent results to session (see session_sync.py for page-order mappings)
                    from pages.AutonomousLab.session_sync import sync_context_to_session
                    sync_context_to_session(session_context)

                    msg_entry = {
                        "role": "assistant",
                        "content": text_content,
                        "activity": live_activity.snapshot(),
                    }

                    # If the user requested a report preview and none was returned, compile one from session sections.
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
                                from agents.shared.image_processor import plotly_figure_to_image_dict, extract_figure_data_for_agent
                                content = store_plotly_artifact_in_message(msg_entry, a.get("artifact_content"))
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
                        elif a.get("artifact_type") == "image_file":
                            img_dict = a.get("figure_image")
                            if img_dict and img_dict.get("data"):
                                msg_entry.setdefault("image_files", []).append({
                                    "data": img_dict["data"],
                                    "caption": a.get("artifact_title", "Image from disk"),
                                })
                                st.session_state.lab_artifact_history.append({
                                    "type": "image",
                                    "figure_image": img_dict,
                                    "caption": (text_content or a.get("artifact_title", ""))[:200],
                                    "source": "document",
                                })
                                st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                        elif a.get("artifact_type") in ("pdf_document", "word_document", "excel_document"):
                            content = a.get("artifact_content") or ""
                            title = a.get("artifact_title", "Document")
                            if content:
                                if a.get("artifact_type") == "excel_document":
                                    msg_tables.append(content)
                                    st.session_state.lab_artifact_history.append({
                                        "type": "table",
                                        "table_md": content,
                                        "caption": (text_content or title)[:200],
                                    })
                                else:
                                    msg_markdown_blocks.append({"content": content, "title": title})
                                    st.session_state.lab_artifact_history.append({
                                        "type": "markdown",
                                        "content": content,
                                        "title": title,
                                        "caption": (text_content or title)[:200],
                                    })
                                st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                            for img_dict in a.get("figure_images") or []:
                                if img_dict and img_dict.get("data"):
                                    msg_entry.setdefault("image_files", []).append({
                                        "data": img_dict["data"],
                                        "caption": title,
                                    })
                                    st.session_state.lab_artifact_history.append({
                                        "type": "image",
                                        "figure_image": img_dict,
                                        "caption": (text_content or title)[:200],
                                        "source": "document",
                                    })
                                    st.session_state.lab_artifact_history = st.session_state.lab_artifact_history[-LAB_ARTIFACT_HISTORY_MAX:]
                    if msg_figures:
                        if len(msg_figures) == 1:
                            pass
                    if msg_tables:
                        msg_entry["tables"] = msg_tables
                    if msg_markdown_blocks:
                        msg_entry["markdown_blocks"] = msg_markdown_blocks
                    # Export last figure when the user explicitly requests save/export/download.
                    if not msg_entry.get("download") and "last_figure_json" in st.session_state:
                        save_intent = any(kw in user_input.lower() for kw in ("save", "export", "download"))
                        if save_intent:
                            try:
                                import base64
                                fig = pio.from_json(st.session_state["last_figure_json"])
                                fmt = "png"
                                ul = user_input.lower()
                                # Prefer file-format "pdf" over turbulence PDF (probability density) phrasing.
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
                    live_activity.render(force=True)
                    try:
                        status.update(label="Done", state="complete", expanded=False)
                    except Exception:
                        pass
                except Exception as e:
                    live_activity.render(force=True)
                    st.session_state.lab_chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                        "activity": live_activity.snapshot(),
                    })
                    try:
                        status.update(label="❌ Error", state="error")
                    except Exception:
                        pass

            st.rerun()


if __name__ == "__main__":
    main()
