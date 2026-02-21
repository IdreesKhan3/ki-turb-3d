"""
Session context builder for Autonomous Lab.
Builds the session_context dict passed to agents (data paths, plot styles, etc.).
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
