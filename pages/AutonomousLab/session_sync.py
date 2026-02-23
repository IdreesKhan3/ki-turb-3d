"""
Centralized sync: copy agent results from session_context back to st.session_state.

Add new mappings in the correct page section below. Page order matches page_schema.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def update_data_directory_in_context(
    session_context: Optional[Dict[str, Any]], data_dir_path
) -> None:
    """Update session_context with data directory used. Call from any tool that uses data.
    Enables all manual pages (PDFs, Spectra, etc.) to render after agent run."""
    if not session_context or data_dir_path is None:
        return
    p = Path(data_dir_path).resolve()
    if p.exists():
        session_context["data_directory"] = str(p)


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
]

# --- PAGE 01 — Overview ---
# (no sync yet)

# --- PAGE 04 — Real Isotropy ---
SYNC_PAGE_REAL_ISOTROPY: List[Tuple[str, str]] = [
    ("axis_labels_real_iso", "axis_labels_real_iso"),
    ("real_iso_legends", "real_iso_legends"),
]

# --- PAGE 05 — Spectral Isotropy ---
# (add: axis_labels_spec_iso, spec_iso_sim_legend_names, spec_iso_legends, etc.)

# --- PAGE 06 — Energy Spectra ---
SYNC_PAGE_SPECTRA: List[Tuple[str, str]] = [
    ("axis_labels_raw", "axis_labels_raw"),
    ("axis_labels_norm", "axis_labels_norm"),
    ("spectrum_legend_names", "spectrum_legend_names"),
    ("norm_legend_names", "norm_legend_names"),
    ("spectra_options", "spectra_options"),
]

# --- PAGE 07 — Flatness ---
# (add: axis_labels_flatness, flatness_sim_legend_names, etc.)

# --- PAGE 08 — Structure Functions ---
# (add: axis_labels_structure, structure_sim_legend_names, etc.)

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
# (add when needed)

# --- PAGE 11 — 3D Volume Viewer ---
# (add when needed)

# --- PAGE 12 — Report Generator ---
SYNC_PAGE_REPORT: List[Tuple[str, str]] = [
    ("report_sections", "report_sections"),
]


def _all_mappings() -> List[Tuple[str, str]]:
    """Collect all mappings in page order."""
    mappings: List[Tuple[str, str]] = []
    mappings.extend(SYNC_APP_LEVEL)
    mappings.extend(SYNC_PAGE_REAL_ISOTROPY)
    mappings.extend(SYNC_PAGE_SPECTRA)
    mappings.extend(SYNC_PAGE_PDFS)
    mappings.extend(SYNC_PAGE_REPORT)
    return mappings


def sync_context_to_session(session_context: Dict[str, Any]) -> None:
    """
    Copy agent results from session_context back to st.session_state.
    Call after agent run (normal flow and confirmation resume).
    """
    if not session_context:
        return
    # plot_styles: merge update (shared across pages)
    if session_context.get("plot_styles"):
        st.session_state.setdefault("plot_styles", {}).update(session_context["plot_styles"])
    # Simple 1:1 copies from all page sections
    for ctx_key, ss_key in _all_mappings():
        if session_context.get(ctx_key) is not None:
            st.session_state[ss_key] = session_context[ctx_key]
    # When agent sets data_directory, mark app as "loaded" so main page shows logo + confirmation
    if session_context.get("data_directory") or session_context.get("data_directories"):
        st.session_state.data_loaded = True
        # Ensure data_directories is set for single-dir case (pages expect it)
        if not st.session_state.get("data_directories") and st.session_state.get("data_directory"):
            st.session_state.data_directories = [st.session_state.data_directory]
