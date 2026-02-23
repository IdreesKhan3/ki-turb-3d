"""
Session context builder for Autonomous Lab.
Builds the session_context dict passed to agents (data paths, plot styles, etc.).

ORGANIZATION: Sections follow page order (page_schema). When adding a new page:
  1. Add section below in correct order
  2. Add corresponding SYNC_* in session_sync.py
"""

import streamlit as st
import plotly.io as pio
from pathlib import Path
from typing import Optional

from utils.image_processor import plotly_figure_to_image_dict, extract_figure_data_for_agent

# Max artifacts to include in context (figures, tables, user images)
ARTIFACT_HISTORY_MAX = 15


def build_session_context(
    data_dir: Optional[str] = None,
    data_path: Optional[Path] = None,
) -> dict:
    """Build session context for agent runs (data paths, plot styles, axis labels, etc.)."""
    from utils.plot_style import default_plot_style

    ctx = {}
    if data_dir and data_path and data_path.exists():
        ctx["data_directory"] = str(data_path)
    elif getattr(st.session_state, "data_directory", None):
        ctx["data_directory"] = str(st.session_state.data_directory)
    if getattr(st.session_state, "data_directories", []):
        ctx["data_directories"] = [str(d) for d in st.session_state.data_directories]
    if getattr(st.session_state, "all_loaded_files", {}):
        ctx["all_loaded_files"] = st.session_state.all_loaded_files

    # --- PAGE 06 — Energy Spectra ---
    plot_styles = st.session_state.setdefault("plot_styles", {})
    spectra_names = ("Raw Energy Spectrum", "Normalized Spectrum", "Time Evolution")
    defaults = {"x_axis_type": "log", "y_axis_type": "log", "line_width": 2.4}
    for name in spectra_names:
        if name not in plot_styles:
            s = default_plot_style()
            s.update(defaults)
            plot_styles[name] = s
    ctx["spectra_plot_styles"] = {n: plot_styles[n] for n in spectra_names}
    ctx["style_config"] = plot_styles["Raw Energy Spectrum"]
    ctx["spectra_style"] = plot_styles["Raw Energy Spectrum"]
    ctx["axis_labels_raw"] = st.session_state.setdefault(
        "axis_labels_raw", {"x": "Wavenumber k", "y": "Energy spectrum E(k)"}
    )
    ctx["axis_labels_norm"] = st.session_state.setdefault(
        "axis_labels_norm", {"x": "Normalized wavenumber kη", "y": "Normalized spectrum E<sub>norm</sub>(kη)"}
    )
    ctx["spectrum_legend_names"] = st.session_state.setdefault("spectrum_legend_names", {})
    ctx["norm_legend_names"] = st.session_state.setdefault("norm_legend_names", {})
    ctx["spectra_options"] = st.session_state.setdefault("spectra_options", {
        "show_std": True, "show_error_bars": True, "pope_scaling_prefix": None,
        "kmin": 3.0, "kmax": 20.0, "kolm_scale_factor": 1.0,
    })

    # Persist agent data cache across messages so "modify previous figure" works without recompute
    ctx["agent_data_cache"] = st.session_state.setdefault("lab_agent_data_cache", {})

    # --- PAGES 04–05 — Real Isotropy, Spectral Isotropy ---
    plot_styles = st.session_state.setdefault("plot_styles", {})
    isotropy_names = ("IC(k) Time-Avg", "Energy Fractions (A)", "Lumley Triangle (B)")
    for name in isotropy_names:
        if name not in plot_styles:
            s = default_plot_style()
            s.update({
                "x_axis_type": "log" if "IC" in name else "linear",
                "y_axis_type": "log" if "IC" in name else "linear",
                "line_width": 2.2,
            })
            plot_styles[name] = s
    ctx["isotropy_plot_styles"] = {n: plot_styles[n] for n in isotropy_names if n in plot_styles}
    ctx["axis_labels_spec_iso"] = st.session_state.setdefault("axis_labels_spec_iso", {"x": "k", "y": "IC(k)"})
    ctx["axis_labels_real_iso"] = st.session_state.setdefault("axis_labels_real_iso", {"x": "t/t0", "y": "Energy fraction"})
    ctx["axis_labels_lumley"] = st.session_state.setdefault("axis_labels_lumley", {"x": "ξ", "y": "η"})

    # --- PAGE 09 — PDFs ---
    ctx["axis_labels_pdfs"] = st.session_state.setdefault("axis_labels_pdfs", {})
    ctx["legend_titles_pdfs"] = st.session_state.setdefault("legend_titles_pdfs", {})
    plot_styles = st.session_state.setdefault("plot_styles", {})
    # Must match pages/09_PDFs.py plot_names for style sync
    pdfs_plot_names = (
        "Velocity PDF", "R-Q Topological Space", "Vorticity PDF", "Enstrophy PDF",
        "Velocity Magnitude PDF", "Dissipation PDF",
        "Velocity-Dissipation Joint PDF", "Velocity-Enstrophy Joint PDF", "Dissipation-Enstrophy Joint PDF",
    )
    for name in pdfs_plot_names:
        if name not in plot_styles:
            s = default_plot_style()
            s.update({"line_width": 2.4, "per_sim_style_comparison": {}})
            plot_styles[name] = s
    ctx["pdfs_plot_styles"] = {n: plot_styles[n] for n in pdfs_plot_names if n in plot_styles}
    ctx["pdfs_style_config"] = plot_styles.get("Velocity Magnitude PDF", default_plot_style())
    # File selections (agent uses when file_paths not specified)
    ctx["pdfs_selected_files_ud"] = st.session_state.get("joint_pdf_files_ud")
    ctx["pdfs_selected_files_uo"] = st.session_state.get("joint_pdf_files_uo")
    ctx["pdfs_selected_files_do"] = st.session_state.get("joint_pdf_files_do")
    ctx["pdfs_selected_files_rq"] = st.session_state.get("joint_pdf_files_rq")
    ctx["pdfs_selected_files_dissipation"] = st.session_state.get("dissipation_file_select")
    ctx["pdfs_selected_files_vorticity"] = st.session_state.get("vorticity_file_select")
    ctx["pdfs_selected_files_enstrophy"] = st.session_state.get("enstrophy_file_select")
    ctx["pdfs_selected_files_velocity_components"] = st.session_state.get("velocity_pdf_file_select")
    ctx["pdfs_selected_files_velocity_magnitude"] = st.session_state.get("velocity_mag_file_select")
    # Bins and normalize (agent uses when not specified in tool args)
    ctx["pdfs_bins_dissipation"] = st.session_state.get("dissipation_pdf_bins")
    ctx["pdfs_normalize_dissipation"] = st.session_state.get("dissipation_normalize")
    ctx["pdfs_bins_vorticity"] = st.session_state.get("vorticity_pdf_bins")
    ctx["pdfs_normalize_vorticity"] = st.session_state.get("vorticity_normalize")
    ctx["pdfs_bins_velocity_components"] = st.session_state.get("velocity_pdf_bins")
    ctx["pdfs_normalize_velocity_components"] = st.session_state.get("velocity_pdf_normalize")
    ctx["pdfs_bins_velocity_magnitude"] = st.session_state.get("velocity_mag_pdf_bins")
    ctx["pdfs_normalize_velocity_magnitude"] = st.session_state.get("velocity_mag_normalize")
    ctx["pdfs_bins_joint"] = st.session_state.get("joint_pdf_bins")
    ctx["pdfs_bins_rq"] = st.session_state.get("joint_pdf_rq_bins")
    ctx["pdfs_normalize_joint"] = st.session_state.get("joint_pdf_normalize")
    ctx["pdfs_log_scale_joint"] = st.session_state.get("joint_pdf_log_scale")
    ctx["pdfs_log_scale_rq"] = st.session_state.get("joint_pdf_rq_log_scale")
    # Page-level: viscosity and grid spacing (agent uses when not in tool args)
    ctx["pdfs_nu"] = st.session_state.get("pdfs_nu_input")
    ctx["pdfs_dx_override"] = st.session_state.get("pdfs_dx_override")

    # --- PAGE 12 — Report Generator ---
    ctx["report_sections"] = st.session_state.get("report_sections") or []

    # --- Shared: last figure, artifact history ---
    if "last_figure_json" in st.session_state:
        try:
            fig = pio.from_json(st.session_state["last_figure_json"])
            ctx["last_figure"] = fig
            # For agent vision: convert figure to image dict (mime_type + data bytes)
            img_dict = plotly_figure_to_image_dict(fig)
            if img_dict:
                ctx["last_figure_image"] = img_dict
            # For precise physics explanation: extract trace data, axis labels, ranges
            ctx["last_figure_data"] = extract_figure_data_for_agent(fig)
        except Exception:
            pass

    # Full artifact history so agents can remember and explain any previous figure/table/image
    artifact_history = getattr(st.session_state, "lab_artifact_history", [])
    if artifact_history:
        ctx["artifact_history"] = artifact_history[-ARTIFACT_HISTORY_MAX:]

    # LLM provider for generation tools (generate_content, generate_code)
    ctx["llm_provider_name"] = getattr(st.session_state, "lab_llm_provider", "gemini")

    return ctx
