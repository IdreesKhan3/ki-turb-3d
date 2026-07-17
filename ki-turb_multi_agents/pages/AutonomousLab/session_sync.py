"""
Centralized sync: copy agent results from session_context back to st.session_state.

Add new mappings in the correct page section below. Page order matches page_schema.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from agents.tools._shared import update_data_directory_in_context
# =============================================================================
# SYNC MAPPINGS — (session_context key, st.session_state key)
# Add new pages in page order. Use 1:1 copy when value is not None.
# =============================================================================

# --- PAGE 00 — App / Shared ---
# Data paths used by main page, PDFs, and other pages. Syncing ensures that when
# the agent runs (e.g. from Autonomous Lab), the data directory it used is
# written back so manual pages (PDFs, etc.) can render without "select data directory".
SYNC_APP_LEVEL: List[Tuple[str, str]] = [
    ("data_directory", "data_directory"),
    ("data_directories", "data_directories"),
    ("simulation_job_id", "sim_workflow_job"),
    ("manifest_path", "dataset_manifest_path"),
    ("dataset_manifest", "dataset_manifest"),
    ("all_loaded_files", "all_loaded_files"),
    ("spectra_data_directory", "spectra_data_directory"),
    ("stats_data_directory", "stats_data_directory"),
    ("isotropy_data_directory", "isotropy_data_directory"),
    ("structure_functions_data_directory", "structure_functions_data_directory"),
    ("analysis_products_path", "analysis_products_path"),
    ("analysis_products", "analysis_products"),
    ("engineering_plan", "engineering_plan"),
    ("engineering_step_index", "engineering_step_index"),
    ("engineering_capability", "engineering_capability"),
    ("engineering_plan_approved", "engineering_plan_approved"),
    ("engineering_intent", "engineering_intent"),
    ("engineering_context", "engineering_context"),
    ("turn_memory", "lab_turn_memory"),
    ("langgraph_thread_id", "langgraph_thread_id"),
]

# --- PAGE 01 — Overview ---
# (not needed yet)

# --- PAGE 02 — Theory Equations ---
SYNC_PAGE_THEORY: List[Tuple[str, str]] = [
    ("d3q19_settings", "d3q19_settings"),
    ("mrt_nu", "mrt_nu"),
    ("mrt_tau", "mrt_tau"),
]

# --- PAGE 04 — Real Isotropy ---
SYNC_PAGE_REAL_ISOTROPY: List[Tuple[str, str]] = [
    ("axis_labels_real_iso", "axis_labels_real_iso"),
    ("real_iso_legends", "real_iso_legends"),
    ("real_iso_tol_a", "tol_a"),
    ("real_iso_tol_c", "tol_c"),
    ("real_iso_tol_d", "tol_d"),
    ("real_iso_tol_e", "tol_e"),
    ("real_iso_norm_x", "real_iso_norm_x"),
    ("real_iso_x_norm", "real_iso_x_norm"),
    ("real_iso_stationary_iter", "real_iso_stationary_iter"),
    ("real_iso_ma_win", "real_iso_ma_win"),
]

# --- PAGE 05 — Spectral Isotropy ---
SYNC_PAGE_SPECTRAL_ISOTROPY: List[Tuple[str, str]] = [
    ("axis_labels_spec_iso", "axis_labels_spec_iso"),
    ("spec_iso_sim_legend_names", "spec_iso_sim_legend_names"),
    ("spec_iso_legends", "spec_iso_legends"),
    ("spec_iso_start_idx", "speciso_start_idx"),
    ("spec_iso_end_idx", "speciso_end_idx"),
    ("spec_iso_show_snap", "speciso_show_snap"),
    ("spec_iso_error_display", "speciso_error_display"),
    ("spec_iso_show_component", "speciso_show_component"),
    ("spec_iso_show_curves", "speciso_show_curves"),
]

# --- PAGE 06 — Energy Spectra ---
SYNC_PAGE_SPECTRA: List[Tuple[str, str]] = [
    ("axis_labels_raw", "axis_labels_raw"),
    ("axis_labels_norm", "axis_labels_norm"),
    ("spectrum_legend_names", "spectrum_legend_names"),
    ("norm_legend_names", "norm_legend_names"),
    ("spectra_options", "spectra_options"),
    ("spectra_start_idx", "energy_start_idx"),
    ("spectra_end_idx", "energy_end_idx"),
    ("spectra_view_mode", "energy_view_mode"),
    ("spectra_every_n", "energy_evol_every_n"),
    ("spectra_error_display", "energy_error_display"),
]

# --- PAGE 07 — Flatness ---
SYNC_PAGE_FLATNESS: List[Tuple[str, str]] = [
    ("axis_labels_flatness", "axis_labels_flatness"),
    ("flatness_legend_names", "flatness_legend_names"),
    ("flatness_start_idx", "flatness_start_idx"),
    ("flatness_end_idx", "flatness_end_idx"),
    ("flatness_num_errorbars", "flatness_num_errorbars"),
    ("flatness_error_display", "flatness_error_display"),
    ("flatness_show_ref", "flatness_show_ref"),
]
# flatness_style_configs synced via plot_styles merge

# --- PAGE 08 — Structure Functions ---
SYNC_PAGE_STRUCTURE: List[Tuple[str, str]] = [
    ("axis_labels_structure", "axis_labels_structure"),
    ("structure_sim_legend_names", "structure_legend_names"),
    ("structure_start_idx", "struct_start_idx"),
    ("structure_end_idx", "struct_end_idx"),
]

# --- PAGE 09 — PDFs ---
SYNC_PAGE_PDFS: List[Tuple[str, str]] = [
    ("axis_labels_pdfs", "axis_labels_pdfs"),
    ("legend_titles_pdfs", "legend_titles_pdfs"),
    # File selections
    ("pdfs_selected_files_ud", "joint_pdf_files_ud"),
    ("pdfs_selected_files_uo", "joint_pdf_files_uo"),
    ("pdfs_selected_files_do", "joint_pdf_files_do"),
    ("pdfs_selected_files_rq", "joint_pdf_files_rq"),
    ("pdfs_selected_files_dissipation", "dissipation_file_select"),
    ("pdfs_selected_files_vorticity", "vorticity_file_select"),
    ("pdfs_selected_files_enstrophy", "enstrophy_file_select"),
    ("pdfs_selected_files_velocity_components", "velocity_pdf_file_select"),
    ("pdfs_selected_files_velocity_magnitude", "velocity_mag_file_select"),
    # Bins and normalize (shared with manual page)
    ("pdfs_bins_dissipation", "dissipation_pdf_bins"),
    ("pdfs_normalize_dissipation", "dissipation_normalize"),
    ("pdfs_bins_vorticity", "vorticity_pdf_bins"),
    ("pdfs_normalize_vorticity", "vorticity_normalize"),
    ("pdfs_bins_velocity_components", "velocity_pdf_bins"),
    ("pdfs_normalize_velocity_components", "velocity_pdf_normalize"),
    ("pdfs_bins_velocity_magnitude", "velocity_mag_pdf_bins"),
    ("pdfs_normalize_velocity_magnitude", "velocity_mag_normalize"),
    ("pdfs_bins_joint", "joint_pdf_bins"),
    ("pdfs_bins_rq", "joint_pdf_rq_bins"),
    ("pdfs_normalize_joint", "joint_pdf_normalize"),
    # Log scale for joint PDFs
    ("pdfs_log_scale_joint", "joint_pdf_log_scale"),
    ("pdfs_log_scale_rq", "joint_pdf_rq_log_scale"),
    # Page-level: viscosity and grid spacing
    ("pdfs_nu", "pdfs_nu_input"),
    ("pdfs_dx_override", "pdfs_dx_override"),
]

# --- PAGE 10 — Other Turbulence Stats ---
SYNC_PAGE_OTHER_TURB_STATS: List[Tuple[str, str]] = [
    ("custom_plot_legend_names", "custom_plot_legend_names"),
    ("custom_plot_axis_labels", "custom_plot_axis_labels"),
    ("custom_plot_traces", "custom_plot_traces"),
    ("turb_stats_use_abs", "plot_use_abs"),
    ("turb_stats_smooth_window", "plot_smooth"),
    ("turb_stats_normalize_x", "plot_norm_x"),
    ("turb_stats_x_norm", "plot_x_norm"),
    ("turb_stats_normalize_y", "plot_norm_y"),
]

# --- PAGE 11 — 3D Volume Viewer ---
SYNC_PAGE_3D_VOLUME: List[Tuple[str, str]] = [
    ("vol3d_field_type", "field_type"),
    ("vol3d_downsample", "downsample"),
    ("vol3d_show_vol", "show_vol"),
    ("vol3d_show_slices", "show_slices"),
    ("vol3d_show_surface", "show_surface"),
    ("vol3d_show_iso", "show_iso"),
    ("vol3d_colormap", "colormap"),
    ("vol3d_color_max", "color_max"),
    ("vol3d_vrange", "vrange"),
    ("vol3d_vol_opacity", "vol_opacity"),
    ("vol3d_vol_surfaces", "vol_surfaces"),
    ("vol3d_iso_opacity", "iso_opacity"),
    ("vol3d_surface_opacity", "surface_opacity"),
    ("vol3d_slice_x", "slice_x"),
    ("vol3d_slice_y", "slice_y"),
    ("vol3d_slice_z", "slice_z"),
    ("vol3d_slice_opacity", "slice_opacity"),
    ("vol3d_use_clip", "use_clip"),
    ("vol3d_clip_x", "clip_x"),
    ("vol3d_clip_y", "clip_y"),
    ("vol3d_clip_z", "clip_z"),
    ("vol3d_show_axes", "show_axes_3d"),
    ("vol3d_show_axis_labels", "show_axis_labels_3d"),
    ("vol3d_camera_preset", "camera_preset"),
    ("vol3d_spacing_choice", "vol3d_spacing_choice"),
    ("vol3d_dx_override", "vol3d_dx_override"),
    ("vol3d_file_index", "file_index"),
    ("vol3d_file_type_selection", "file_type_selection"),
]

# --- PAGE 12 — Report Generator ---
SYNC_PAGE_REPORT: List[Tuple[str, str]] = [
    ("report_sections", "report_sections"),
    ("report_title", "report_title"),
    ("report_author", "report_author"),
    ("report_include_toc", "report_include_toc"),
]


def _all_mappings() -> List[Tuple[str, str]]:
    """Collect all mappings in page order."""
    mappings: List[Tuple[str, str]] = []
    mappings.extend(SYNC_APP_LEVEL)
    mappings.extend(SYNC_PAGE_THEORY)
    mappings.extend(SYNC_PAGE_REAL_ISOTROPY)
    mappings.extend(SYNC_PAGE_SPECTRAL_ISOTROPY)
    mappings.extend(SYNC_PAGE_SPECTRA)
    mappings.extend(SYNC_PAGE_FLATNESS)
    mappings.extend(SYNC_PAGE_STRUCTURE)
    mappings.extend(SYNC_PAGE_PDFS)
    mappings.extend(SYNC_PAGE_OTHER_TURB_STATS)
    mappings.extend(SYNC_PAGE_3D_VOLUME)
    mappings.extend(SYNC_PAGE_REPORT)
    return mappings


def sync_context_to_session(session_context: Dict[str, Any]) -> None:
    """
    Copy agent results from session_context back to st.session_state.
    Call after agent run (normal flow and confirmation resume).
    """
    from agents.shared.session_context_sanitize import sanitize_session_context_for_persistence

    session_context = sanitize_session_context_for_persistence(session_context)
    if not session_context:
        return
    # plot_styles: merge update (shared across pages)
    if session_context.get("plot_styles"):
        st.session_state.setdefault("plot_styles", {}).update(session_context["plot_styles"])
    # Flatness: sync flatness_style_configs to plot_styles
    flatness_configs = session_context.get("flatness_style_configs")
    if flatness_configs:
        for plot_name, style in flatness_configs.items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[plot_name] = style
    elif session_context.get("flatness_style_config") is not None:
        st.session_state.setdefault("plot_styles", {})["Flatness Factors"] = session_context["flatness_style_config"]
    # Spectral isotropy: sync isotropy_plot_styles to plot_styles (IC(k) Time-Avg, Component Spectra)
    if session_context.get("isotropy_plot_styles"):
        for name, style in session_context["isotropy_plot_styles"].items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[name] = style
    # Real isotropy: sync real_iso_plot_styles to plot_styles (A–F subplots)
    if session_context.get("real_iso_plot_styles"):
        for name, style in session_context["real_iso_plot_styles"].items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[name] = style
    # Energy spectra: sync spectra_plot_styles to plot_styles (Raw, Normalized, Time Evolution)
    if session_context.get("spectra_plot_styles"):
        for name, style in session_context["spectra_plot_styles"].items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[name] = style
    # Structure functions: sync all subplot configs the agent plotted (sp, ess, anomalies)
    struct_configs = session_context.get("structure_style_configs")
    if struct_configs:
        for plot_name, style in struct_configs.items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[plot_name] = style
    # PDFs: sync pdfs_plot_styles to plot_styles (all 9 subplots)
    if session_context.get("pdfs_plot_styles"):
        for name, style in session_context["pdfs_plot_styles"].items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[name] = style
    # Other Turbulence Stats: sync turb_stats_plot_styles to plot_styles (Custom Multi-Trace Plot)
    if session_context.get("turb_stats_plot_styles"):
        for name, style in session_context["turb_stats_plot_styles"].items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[name] = style
        # Also update plot_style so manual page displays agent styles
        custom_plot_style = session_context.get("turb_stats_plot_styles", {}).get("Custom Multi-Trace Plot")
        if isinstance(custom_plot_style, dict):
            st.session_state.plot_style = custom_plot_style
    # 3D Volume Viewer: sync volume_3d_plot_styles to plot_styles and plot_style_3d
    if session_context.get("volume_3d_plot_styles"):
        for name, style in session_context["volume_3d_plot_styles"].items():
            if isinstance(style, dict):
                st.session_state.setdefault("plot_styles", {})[name] = style
        vol_3d_style = session_context.get("volume_3d_plot_styles", {}).get("3D Volume")
        if isinstance(vol_3d_style, dict):
            st.session_state.plot_style_3d = vol_3d_style
    # Simple 1:1 copies from all page sections
    for ctx_key, ss_key in _all_mappings():
        if session_context.get(ctx_key) is not None:
            st.session_state[ss_key] = session_context[ctx_key]

    # 3D Volume: sync slider_index to match file_index (time step slider uses slider_index)
    fi = session_context.get("vol3d_file_index")
    if fi is not None:
        st.session_state["slider_index"] = fi

    # 3D Volume: store agent options for manual page to apply on load (ensures widgets display correctly)
    if any(session_context.get(k) is not None for k in ("vol3d_field_type", "vol3d_show_iso", "vol3d_show_slices")):
        st.session_state["vol3d_from_agent"] = True

    # 3D Volume: sync iso thresholds (manual page uses different keys per field type)
    ft = session_context.get("vol3d_field_type")
    iso_log = session_context.get("vol3d_iso_value_log10")
    iso_val = session_context.get("vol3d_iso_value")
    if ft == "Q_S^S" and iso_log is not None:
        st.session_state["log_qss_threshold"] = iso_log
    elif ft == "Q Invariant" and iso_log is not None:
        st.session_state["log_Q_Invariant_threshold"] = iso_log
    elif ft == "R Invariant" and iso_log is not None:
        st.session_state["log_R_Invariant_threshold"] = iso_log
    elif iso_val is not None:
        st.session_state["iso_value"] = iso_val

    # 3D Volume: sync file_type_selection from vol3d_file_type_filter
    ffilter = session_context.get("vol3d_file_type_filter")
    if ffilter and isinstance(ffilter, str):
        data_dirs = st.session_state.get("data_directories") or []
        if not data_dirs and st.session_state.get("data_directory"):
            data_dirs = [st.session_state.data_directory]
        if data_dirs:
            try:
                from pages.VolumeViewer3D.file_loading import collect_volume_files
                vti_files, hdf5_files, _ = collect_volume_files(data_dirs)
                opts = []
                if vti_files:
                    opts.append(f"VTI ({len(vti_files)} files)")
                if hdf5_files:
                    opts.append(f"HDF5 ({len(hdf5_files)} files)")
                if vti_files and hdf5_files:
                    opts.append("Both (VTI + HDF5)")
                f = ffilter.strip().lower()
                if f == "vti" and any(o.startswith("VTI") for o in opts):
                    sel = next(o for o in opts if o.startswith("VTI"))
                    st.session_state["file_type_selection"] = sel
                    st.session_state["prev_file_type"] = sel
                elif f in ("hdf5", "h5") and any(o.startswith("HDF5") for o in opts):
                    sel = next(o for o in opts if o.startswith("HDF5"))
                    st.session_state["file_type_selection"] = sel
                    st.session_state["prev_file_type"] = sel
                elif f == "both" and opts:
                    sel = opts[-1] if len(opts) > 1 else opts[0]
                    st.session_state["file_type_selection"] = sel
                    st.session_state["prev_file_type"] = sel
            except Exception:
                pass
    # When agent sets data_directory, mark app as "loaded" so main page shows logo + confirmation
    if session_context.get("data_directory") or session_context.get("data_directories"):
        st.session_state.data_loaded = True
        # Ensure data_directories is set for single-dir case (pages expect it)
        if not st.session_state.get("data_directories") and st.session_state.get("data_directory"):
            st.session_state.data_directories = [st.session_state.data_directory]
        # When agent used multiple dirs (multi-sim), switch app to multi-directory mode so main page shows both
        data_dirs = st.session_state.get("data_directories") or []
        if len(data_dirs) > 1:
            st.session_state.multi_directory_mode = True
    if session_context.get("last_figure_json"):
        st.session_state["last_figure_json"] = session_context["last_figure_json"]
