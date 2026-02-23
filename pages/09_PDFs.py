"""
PDFs Page - Probability Density Functions for Multi-Simulation Analysis
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from data_readers.vti_reader import read_vti_file
from data_readers.hdf5_reader import read_hdf5_file
from pages.PDFs.vorticity_stats import render_vorticity_stats_tab
from pages.PDFs.velocity_magnitude_stats import render_velocity_magnitude_tab
from pages.PDFs.dissipation_stats import render_dissipation_tab
from pages.PDFs.joint_pdf_stats import render_joint_pdf_tab
from pages.PDFs.pdf_params import get_grid_spacing_options
from data_readers.parameter_reader import read_parameters
from utils.plot_style import resolve_line_style, apply_axis_limits, apply_figure_size
from pages.PDFs.pdfs_plot_style import (
    get_plot_style, apply_plot_style,
    _get_palette, plot_style_sidebar, export_panel, ensure_label_state
)
from utils.report_builder import capture_button

st.set_page_config(page_icon="⚫")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=True)
def _cached_read_vti(filepath: str):
    """Cached VTI file reading for performance"""
    abs_path = str(Path(filepath).resolve())
    return read_vti_file(abs_path)

@st.cache_data(show_spinner=True)
def _cached_read_hdf5(filepath: str, fortran_order: bool = True, _cache_version: str = "v2"):
    """Cached HDF5 file reading for performance
    
    fortran_order: If True, apply transpose for Fortran-written HDF5.
    _cache_version: Internal parameter to invalidate cache when reader is updated
    """
    abs_path = str(Path(filepath).resolve())
    return read_hdf5_file(abs_path, fortran_order=fortran_order)

def _load_velocity_file(filepath: str):
    """Load velocity data from either VTI or HDF5 file"""
    abs_filepath = str(Path(filepath).resolve())
    filepath_lower = abs_filepath.lower()
    fortran_order = st.session_state.get('hdf5_fortran_order', True)
    if filepath_lower.endswith(('.h5', '.hdf5')):
        return _cached_read_hdf5(abs_filepath, fortran_order=fortran_order)
    elif filepath_lower.endswith('.vti'):
        return _cached_read_vti(abs_filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Expected .vti, .h5, or .hdf5")


# -----------------------------
# Main
# -----------------------------
def main():
    inject_theme_css()
    st.title("PDFs")
    
    # Get data directories
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the main page.")
        return
    
    # Process ALL directories independently - collect files from all
    import glob
    from utils.file_detector import natural_sort_key
    
    all_vti_files = []
    all_hdf5_files = []
    
    for data_dir_path in data_dirs:
        # Resolve path to ensure it works regardless of how it was stored
        try:
            data_dir = Path(data_dir_path).resolve()
            if data_dir.exists() and data_dir.is_dir():
                # Collect files from THIS directory independently
                dir_vti = sorted(
                    glob.glob(str(data_dir / "*.vti")) + 
                    glob.glob(str(data_dir / "*.VTI")),
                    key=natural_sort_key
                )
                dir_hdf5 = sorted(
                    glob.glob(str(data_dir / "*.h5")) + 
                    glob.glob(str(data_dir / "*.H5")) +
                    glob.glob(str(data_dir / "*.hdf5")) + 
                    glob.glob(str(data_dir / "*.HDF5")),
                    key=natural_sort_key
                )
                all_vti_files.extend(dir_vti)
                all_hdf5_files.extend(dir_hdf5)
        except Exception:
            continue  # Skip invalid directories
    
    # Use first directory for metadata storage
    data_dir = Path(data_dirs[0]).resolve()
    
    # Initialize plot styles and label state
    st.session_state.setdefault("plot_styles", {})
    ensure_label_state()
    
    # Combine all files from all directories
    all_files = [Path(f).name for f in all_vti_files + all_hdf5_files]

    # Plot style sidebar
    plot_names = ["Velocity PDF", "R-Q Topological Space", "Vorticity PDF", "Enstrophy PDF", "Velocity Magnitude PDF", "Dissipation PDF", "Velocity-Dissipation Joint PDF", "Velocity-Enstrophy Joint PDF", "Dissipation-Enstrophy Joint PDF"]
    if all_files:
        plot_style_sidebar(data_dir, all_files, plot_names, include_label_panel=True)

    # Grid spacing (shared by all gradient-based PDF tabs) — read both files, user chooses
    spacing_options = get_grid_spacing_options(data_dir)
    with st.sidebar.expander("🔧 Advanced (grid spacing)", expanded=False):
        choice_labels = list(spacing_options.keys())
        default_idx = 0  # LBM first, NS second
        spacing_choice = st.radio(
            "Grid spacing source",
            choice_labels,
            index=min(default_idx, len(choice_labels) - 1),
            help="LBM: dx=1 (lattice units). NS: dx=L/nx from simulation.json. Choose based on your data.",
            key="pdfs_spacing_choice"
        )
        dx_selected, dy_selected, dz_selected = spacing_options[spacing_choice]
        st.caption("Used for gradients, strain rates, dissipation, vorticity (periodic BCs).")
        synced_dx = st.session_state.get("pdfs_dx_override")
        manual_dx_value = max(1e-6, float(synced_dx)) if synced_dx is not None else dx_selected
        manual_dx = st.number_input(
            "Or override dx (=dy=dz)",
            value=manual_dx_value,
            min_value=1e-6,
            step=0.001,
            format="%.6f",
            help="Optional: type a custom dx to override the selection above.",
            key="pdfs_dx_override"
        )
        # The number box remembers its last value when you switch LBM↔NS. So we trust
        # the radio choice unless you typed something different from both presets.
        use_override = not any(
            abs(manual_dx - v[0]) < 1e-9
            for v in spacing_options.values()
        )
        dx, dy, dz = (manual_dx, manual_dx, manual_dx) if use_override else (dx_selected, dy_selected, dz_selected)
        st.caption(f"Using dx = {dx:.6f}")

    # Physical Parameters (ν) — shared by Dissipation and Joint PDFs tabs, shown once
    st.sidebar.header("⚙️ Physical Parameters")
    nu_from_file = None
    param_source = None
    for candidate in (data_dir / "simulation.input", data_dir / "simulation.json"):
        if candidate.exists():
            try:
                params = read_parameters(str(candidate))
                if "nu" in params:
                    nu_from_file = params["nu"]
                    param_source = candidate.name
                    break
            except Exception as e:
                st.sidebar.warning(f"Error reading {candidate.name}: {e}")
    default_nu = nu_from_file if nu_from_file is not None else 0.004
    nu_value = st.session_state.get("pdfs_nu_input")
    if nu_value is not None:
        nu_value = max(0.0001, float(nu_value))
    else:
        nu_value = default_nu
    if nu_from_file is not None:
        st.sidebar.info(f"📄 Viscosity from {param_source}: {nu_from_file:.6f}")
    else:
        st.sidebar.warning("Viscosity not found in simulation.input or simulation.json. Please enter manually or check parameter file.")
    nu_help = "Kinematic viscosity used in dissipation calculation: ε = 2ν S_ij S_ij"
    if nu_from_file is not None:
        nu_help += f" (loaded from {param_source}, can be overridden)"
    else:
        nu_help += " (enter manually)"
    nu = st.sidebar.number_input(
        "ν (Kinematic Viscosity)",
        value=nu_value,
        min_value=0.0001,
        step=0.0001,
        format="%.6f",
        help=nu_help,
        key="pdfs_nu_input",
    )
    
    # Create tabs
    tabs = st.tabs([
        "Vorticity & Enstrophy PDFs",
        "Velocity Magnitude PDF",
        "Dissipation Rate PDF",
        "Joint PDFs"
    ])
    
    # ============================================
    # Tab: Vorticity & Enstrophy PDFs
    # ============================================
    with tabs[0]:
        render_vorticity_stats_tab(
            data_dirs,  # Pass all directories
            _load_velocity_file,
            dx=dx, dy=dy, dz=dz,
            get_plot_style_func=get_plot_style,
            apply_plot_style_func=apply_plot_style,
            get_palette_func=_get_palette,
            resolve_line_style_func=resolve_line_style,
            export_panel_func=export_panel,
            capture_button_func=capture_button
        )
    
    # ============================================
    # Tab: Velocity Magnitude PDF
    # ============================================
    with tabs[1]:
        render_velocity_magnitude_tab(
            data_dirs,  # Pass all directories
            _load_velocity_file,
            get_plot_style_func=get_plot_style,
            apply_plot_style_func=apply_plot_style,
            get_palette_func=_get_palette,
            resolve_line_style_func=resolve_line_style,
            export_panel_func=export_panel,
            capture_button_func=capture_button
        )
    
    # ============================================
    # Tab: Dissipation Rate PDF
    # ============================================
    with tabs[2]:
        render_dissipation_tab(
            data_dirs,  # Pass all directories
            _load_velocity_file,
            dx=dx, dy=dy, dz=dz, nu=nu,
            get_plot_style_func=get_plot_style,
            apply_plot_style_func=apply_plot_style,
            get_palette_func=_get_palette,
            resolve_line_style_func=resolve_line_style,
            export_panel_func=export_panel,
            capture_button_func=capture_button
        )
    
    # ============================================
    # Tab: Joint PDFs
    # ============================================
    with tabs[3]:
        render_joint_pdf_tab(
            data_dirs,  # Pass all directories
            _load_velocity_file,
            dx=dx, dy=dy, dz=dz, nu=nu,
            get_plot_style_func=get_plot_style,
            apply_plot_style_func=apply_plot_style,
            get_palette_func=_get_palette,
            resolve_line_style_func=resolve_line_style,
            export_panel_func=export_panel,
            capture_button_func=capture_button
        )


if __name__ == "__main__":
    main()
