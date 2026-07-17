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

from agents.shared.image_processor import plotly_figure_to_image_dict, extract_figure_data_for_agent

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
    if getattr(st.session_state, "dataset_manifest_path", None):
        ctx["manifest_path"] = str(st.session_state.dataset_manifest_path)
    if getattr(st.session_state, "sim_workflow_job", None):
        ctx["simulation_job_id"] = str(st.session_state.sim_workflow_job)
    if getattr(st.session_state, "dataset_manifest", None):
        ctx["dataset_manifest"] = st.session_state.dataset_manifest
    if getattr(st.session_state, "spectra_data_directory", None):
        ctx["spectra_data_directory"] = str(st.session_state.spectra_data_directory)
    if getattr(st.session_state, "analysis_products_path", None):
        ctx["analysis_products_path"] = str(st.session_state.analysis_products_path)
    if getattr(st.session_state, "analysis_products", None):
        ctx["analysis_products"] = st.session_state.analysis_products

    # --- PAGE 02 — Theory Equations ---
    ctx["d3q19_settings"] = st.session_state.get("d3q19_settings")
    ctx["mrt_nu"] = st.session_state.get("mrt_nu")
    ctx["mrt_tau"] = st.session_state.get("mrt_tau")

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
    ctx["spectra_start_idx"] = st.session_state.get("energy_start_idx")
    ctx["spectra_end_idx"] = st.session_state.get("energy_end_idx")
    ctx["spectra_view_mode"] = st.session_state.get("energy_view_mode")
    ctx["spectra_every_n"] = st.session_state.get("energy_evol_every_n")
    ctx["spectra_error_display"] = st.session_state.get("energy_error_display")

    # Persist agent data cache across messages so "modify previous figure" works without recompute
    ctx["agent_data_cache"] = st.session_state.setdefault("lab_agent_data_cache", {})

    # --- PAGES 04–05 — Real Isotropy, Spectral Isotropy ---
    plot_styles = st.session_state.setdefault("plot_styles", {})
    spec_iso_plot_names = ("IC(k) Time-Avg", "Component Spectra")
    isotropy_names = ("IC(k) Time-Avg", "Energy Fractions (A)", "Lumley Triangle (B)", "Component Spectra")
    for name in isotropy_names:
        if name not in plot_styles:
            s = default_plot_style()
            s.update({"line_width": 2.2})
            if name == "IC(k) Time-Avg":
                s.update({"x_axis_type": "log", "y_axis_type": "linear"})
            elif name == "Component Spectra":
                s.update({"x_axis_type": "log", "y_axis_type": "log"})
            else:
                s.update({"x_axis_type": "linear", "y_axis_type": "linear"})
            plot_styles[name] = s
    ctx["isotropy_plot_styles"] = {n: plot_styles[n] for n in spec_iso_plot_names if n in plot_styles}
    ctx["axis_labels_spec_iso"] = st.session_state.setdefault("axis_labels_spec_iso", {"k": "k", "ic": "IC(k)", "ek": "E<sub>ii</sub>(k)"})
    ctx["spec_iso_sim_legend_names"] = st.session_state.setdefault("spec_iso_sim_legend_names", {})
    ctx["spec_iso_legends"] = st.session_state.setdefault("spec_iso_legends", {
        "IC": "IC(k) (time-avg)", "IC_snap": "IC(k) snapshots",
        "E11": "E<sub>11</sub>(k)", "E22": "E<sub>22</sub>(k)", "E33": "E<sub>33</sub>(k)",
    })
    if "Component Spectra" not in plot_styles:
        s = default_plot_style()
        s.update({"x_axis_type": "log", "y_axis_type": "log", "line_width": 2.2})
        plot_styles["Component Spectra"] = s
    ctx["component_spectra_style_config"] = plot_styles.get("Component Spectra")
    ctx["spec_iso_start_idx"] = st.session_state.get("speciso_start_idx")
    ctx["spec_iso_end_idx"] = st.session_state.get("speciso_end_idx")
    ctx["spec_iso_show_snap"] = st.session_state.get("speciso_show_snap")
    ctx["spec_iso_error_display"] = st.session_state.get("speciso_error_display")
    ctx["spec_iso_show_component"] = st.session_state.get("speciso_show_component")
    ctx["spec_iso_show_curves"] = st.session_state.get("speciso_show_curves")
    # Real Isotropy: page structure (time, energy_frac, lumley_x, lumley_y, bij, cross, dev, convergence)
    real_iso_plot_names = (
        "Energy Fractions (A)",
        "Lumley Triangle (B)",
        "Diagonal b_ii (C)",
        "Cross-correlations (D)",
        "Deviations (E)",
        "Convergence (F)",
    )
    for name in real_iso_plot_names:
        if name not in plot_styles:
            s = default_plot_style()
            s.update({"line_width": 2.2, "x_axis_type": "linear", "y_axis_type": "linear"})
            if name in ("Cross-correlations (D)", "Deviations (E)", "Convergence (F)"):
                s["y_axis_type"] = "log"
            if name == "Lumley Triangle (B)":
                s["line_width"] = 1.5
            if name == "Diagonal b_ii (C)":
                s["line_width"] = 1.6
            plot_styles[name] = s
    ctx["real_iso_plot_styles"] = {n: plot_styles[n] for n in real_iso_plot_names}
    ctx["axis_labels_real_iso"] = st.session_state.setdefault("axis_labels_real_iso", {
        "time": "t/t₀", "energy_frac": "Energy fraction", "bij": "Anisotropy tensor b<sub>ij</sub>",
        "cross": "Cross-correlations / Anisotropy index", "dev": "Absolute deviation",
        "convergence": "Running standard deviation",
        "lumley_x": "ξ = (III<sub>b</sub>/2)<sup>1/3</sup>", "lumley_y": "η = (-II<sub>b</sub>/3)<sup>1/2</sup>",
    })
    ctx["real_iso_legends"] = st.session_state.setdefault("real_iso_legends", {
        "Ex": "E<sub>x</sub>/E<sub>tot</sub>", "Ey": "E<sub>y</sub>/E<sub>tot</sub>", "Ez": "E<sub>z</sub>/E<sub>tot</sub>",
        "b11": "b<sub>11</sub>", "b22": "b<sub>22</sub>", "b33": "b<sub>33</sub>",
        "b12": "|b<sub>12</sub>|", "b13": "|b<sub>13</sub>|", "b23": "|b<sub>23</sub>|", "anis": "Anisotropy index",
        "devx": "devx", "devy": "devy", "devz": "devz", "maxdev": "Max deviation",
    })
    ctx["real_iso_tol_a"] = st.session_state.get("tol_a")
    ctx["real_iso_tol_c"] = st.session_state.get("tol_c")
    ctx["real_iso_tol_d"] = st.session_state.get("tol_d")
    ctx["real_iso_tol_e"] = st.session_state.get("tol_e")
    ctx["real_iso_norm_x"] = st.session_state.get("real_iso_norm_x")
    ctx["real_iso_x_norm"] = st.session_state.get("real_iso_x_norm")
    ctx["real_iso_stationary_iter"] = st.session_state.get("real_iso_stationary_iter")
    ctx["real_iso_ma_win"] = st.session_state.get("real_iso_ma_win")

    # --- PAGE 07 — Flatness ---
    ctx["axis_labels_flatness"] = st.session_state.setdefault("axis_labels_flatness", {
        "x": "Separation distance r",
        "y": "Longitudinal flatness F<sub>L</sub>(r)",
    })
    ctx["flatness_legend_names"] = st.session_state.setdefault("flatness_legend_names", {})
    ctx["flatness_style_config"] = plot_styles.get("Flatness Factors")
    ctx["flatness_style_configs"] = {"Flatness Factors": plot_styles.get("Flatness Factors")}
    ctx["flatness_start_idx"] = st.session_state.get("flatness_start_idx")
    ctx["flatness_end_idx"] = st.session_state.get("flatness_end_idx")
    ctx["flatness_num_errorbars"] = st.session_state.get("flatness_num_errorbars")
    ctx["flatness_error_display"] = st.session_state.get("flatness_error_display")
    ctx["flatness_show_ref"] = st.session_state.get("flatness_show_ref")

    # --- PAGE 08 — Structure Functions ---
    struct_plot_names = ("S_p(r) vs r", "ESS (S_p vs S_3)", "ESS Inset", "Anomalies (ξₚ − p/3)")
    for name in struct_plot_names:
        if name not in plot_styles:
            s = default_plot_style()
            s.update({"x_axis_type": "log", "y_axis_type": "log", "line_width": 2.4, "per_sim_style_structure": {}})
            plot_styles[name] = s
    ctx["structure_style_configs"] = {
        "S_p(r) vs r": plot_styles.get("S_p(r) vs r"),
        "ESS (S_p vs S_3)": plot_styles.get("ESS (S_p vs S_3)"),
        "ESS Inset": plot_styles.get("ESS Inset"),
        "Anomalies (ξₚ − p/3)": plot_styles.get("Anomalies (ξₚ − p/3)"),
    }
    ctx["axis_labels_structure"] = st.session_state.setdefault("axis_labels_structure", {
        "x_r": "Separation distance r", "y_sp": "Structure functions S<sub>p</sub>(r)",
        "x_ess": "S<sub>3</sub>(r)", "y_ess": "S<sub>p</sub>(r)", "x_anom": "p", "y_anom": "ξ<sub>p</sub> - p/3",
    })
    ctx["structure_sim_legend_names"] = st.session_state.setdefault("structure_legend_names", {})
    ctx["structure_start_idx"] = st.session_state.get("struct_start_idx")
    ctx["structure_end_idx"] = st.session_state.get("struct_end_idx")

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

    # --- PAGE 10 — Other Turbulence Stats ---
    turb_stats_plot_name = "Custom Multi-Trace Plot"
    plot_styles = st.session_state.setdefault("plot_styles", {})
    if turb_stats_plot_name not in plot_styles:
        ps_turb = st.session_state.get("plot_style") or default_plot_style()
        if isinstance(ps_turb, dict):
            plot_styles[turb_stats_plot_name] = dict(ps_turb)
        else:
            s = default_plot_style()
            s.update({"line_width": 2.2, "x_axis_type": "linear", "y_axis_type": "linear"})
            plot_styles[turb_stats_plot_name] = s
    ctx["turb_stats_plot_styles"] = {turb_stats_plot_name: plot_styles[turb_stats_plot_name]}
    ctx["custom_plot_legend_names"] = st.session_state.setdefault("custom_plot_legend_names", {})
    ctx["custom_plot_axis_labels"] = st.session_state.setdefault("custom_plot_axis_labels", {"x": "X", "y": "Y"})
    ctx["custom_plot_traces"] = st.session_state.get("custom_plot_traces", [])
    ctx["turb_stats_use_abs"] = st.session_state.get("plot_use_abs")
    ctx["turb_stats_smooth_window"] = st.session_state.get("plot_smooth")
    ctx["turb_stats_normalize_x"] = st.session_state.get("plot_norm_x")
    ctx["turb_stats_x_norm"] = st.session_state.get("plot_x_norm")
    ctx["turb_stats_normalize_y"] = st.session_state.get("plot_norm_y")

    # --- PAGE 11 — 3D Volume Viewer ---
    volume_3d_plot_name = "3D Volume"
    plot_styles = st.session_state.setdefault("plot_styles", {})
    if volume_3d_plot_name not in plot_styles:
        ps_3d = st.session_state.get("plot_style_3d") or default_plot_style()
        if isinstance(ps_3d, dict):
            plot_styles[volume_3d_plot_name] = dict(ps_3d)
        else:
            s = default_plot_style()
            s.update({"line_width": 2.2})
            plot_styles[volume_3d_plot_name] = s
    ctx["volume_3d_plot_styles"] = {volume_3d_plot_name: plot_styles[volume_3d_plot_name]}
    ctx["vol3d_field_type"] = st.session_state.get("field_type")
    ctx["vol3d_downsample"] = st.session_state.get("downsample")
    ctx["vol3d_show_vol"] = st.session_state.get("show_vol")
    ctx["vol3d_show_slices"] = st.session_state.get("show_slices")
    ctx["vol3d_show_surface"] = st.session_state.get("show_surface")
    ctx["vol3d_show_iso"] = st.session_state.get("show_iso")
    ctx["vol3d_colormap"] = st.session_state.get("colormap")
    ctx["vol3d_color_max"] = st.session_state.get("color_max")
    ctx["vol3d_vrange"] = st.session_state.get("vrange")
    ctx["vol3d_vol_opacity"] = st.session_state.get("vol_opacity")
    ctx["vol3d_vol_surfaces"] = st.session_state.get("vol_surfaces")
    ctx["vol3d_iso_opacity"] = st.session_state.get("iso_opacity")
    ctx["vol3d_surface_opacity"] = st.session_state.get("surface_opacity")
    ctx["vol3d_slice_x"] = st.session_state.get("slice_x")
    ctx["vol3d_slice_y"] = st.session_state.get("slice_y")
    ctx["vol3d_slice_z"] = st.session_state.get("slice_z")
    ctx["vol3d_slice_opacity"] = st.session_state.get("slice_opacity")
    ctx["vol3d_use_clip"] = st.session_state.get("use_clip")
    ctx["vol3d_clip_x"] = st.session_state.get("clip_x")
    ctx["vol3d_clip_y"] = st.session_state.get("clip_y")
    ctx["vol3d_clip_z"] = st.session_state.get("clip_z")
    ctx["vol3d_show_axes"] = st.session_state.get("show_axes_3d")
    ctx["vol3d_show_axis_labels"] = st.session_state.get("show_axis_labels_3d")
    ctx["vol3d_camera_preset"] = st.session_state.get("camera_preset")
    ctx["vol3d_spacing_choice"] = st.session_state.get("vol3d_spacing_choice")
    ctx["vol3d_dx_override"] = st.session_state.get("vol3d_dx_override")
    ctx["vol3d_file_index"] = st.session_state.get("file_index")
    ctx["vol3d_file_type_selection"] = st.session_state.get("file_type_selection")

    # --- PAGE 12 — Report Generator ---
    # Use st.session_state list directly so modifications persist when sync is skipped (e.g. pending confirmation).
    ctx["report_sections"] = st.session_state.setdefault("report_sections", [])
    ctx["report_title"] = st.session_state.get("report_title")
    ctx["report_author"] = st.session_state.get("report_author")
    ctx["report_include_toc"] = st.session_state.get("report_include_toc", True)

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
    ctx["llm_provider_name"] = getattr(st.session_state, "lab_llm_provider", "deepseek")

    # Engineering workflow (plan → approve → step-by-step continue)
    for key in (
        "engineering_plan",
        "engineering_step_index",
        "engineering_capability",
        "engineering_plan_approved",
        "engineering_intent",
        "engineering_context",
    ):
        if getattr(st.session_state, key, None) is not None:
            ctx[key] = st.session_state[key]

    # Same-chat continuity anchors
    if getattr(st.session_state, "lab_turn_memory", None) is not None:
        ctx["turn_memory"] = st.session_state.lab_turn_memory
    if getattr(st.session_state, "langgraph_thread_id", None):
        ctx["langgraph_thread_id"] = st.session_state.langgraph_thread_id

    # KI-TURB-owned OpenLB HIT app (outside upstream OpenLB tree)
    rel = "cfd_solvers/SolverApps/kiTurbHIT3D"
    legacy = "cfd_solvers/openLB/examples/kiTurbHIT3D"
    for candidate in (
        Path.cwd() / rel,
        Path.cwd().parent / rel,
        Path.cwd() / legacy,
        Path.cwd().parent / legacy,
    ):
        if candidate.is_dir():
            ctx["openlb_app_dir"] = str(candidate.resolve())
            break

    return ctx
