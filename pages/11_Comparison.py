"""
Multi-Simulation Comparison Page
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from data_readers.vti_reader import read_vti_file
from data_readers.hdf5_reader import read_hdf5_file
from utils.topology_stats import render_topology_stats_tab
from utils.plot_style import resolve_line_style, apply_axis_limits, apply_figure_size
from utils.comparison_plot_style import (
    _load_ui_metadata, _save_ui_metadata, get_plot_style, apply_plot_style,
    _get_palette, plot_style_sidebar, export_panel
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
def _cached_read_hdf5(filepath: str, _cache_version: str = "v2"):
    """Cached HDF5 file reading for performance
    
    _cache_version: Internal parameter to invalidate cache when reader is updated
    """
    abs_path = str(Path(filepath).resolve())
    return read_hdf5_file(abs_path)

def _load_velocity_file(filepath: str):
    """Load velocity data from either VTI or HDF5 file"""
    abs_filepath = str(Path(filepath).resolve())
    filepath_lower = abs_filepath.lower()
    if filepath_lower.endswith(('.h5', '.hdf5')):
        return _cached_read_hdf5(abs_filepath)
    elif filepath_lower.endswith('.vti'):
        return _cached_read_vti(abs_filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Expected .vti, .h5, or .hdf5")


# -----------------------------
# Main
# -----------------------------
def main():
    inject_theme_css()
    st.title("🔀 Multi-Simulation Comparison")
    
    # Get data directories
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the main page.")
        return
    
    data_dir = Path(data_dirs[0])
    
    # Initialize plot styles
    st.session_state.setdefault("plot_styles", {})
    
    # Load UI metadata
    if st.session_state.get("_last_comparison_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        if "plot_styles" not in st.session_state:
            st.session_state.plot_styles = {}
        st.session_state["_last_comparison_dir"] = str(data_dir)
    
    # Get all files for sidebar
    import glob
    from utils.file_detector import natural_sort_key
    vti_files = sorted(
        glob.glob(str(data_dir / "*.vti")) + 
        glob.glob(str(data_dir / "*.VTI")),
        key=natural_sort_key
    )
    hdf5_files = sorted(
        glob.glob(str(data_dir / "*.h5")) + 
        glob.glob(str(data_dir / "*.H5")) +
        glob.glob(str(data_dir / "*.hdf5")) + 
        glob.glob(str(data_dir / "*.HDF5")),
        key=natural_sort_key
    )
    all_files = [Path(f).name for f in vti_files + hdf5_files]
    
    # Plot style sidebar
    plot_names = ["Velocity PDF", "R-Q Topological Space"]
    if all_files:
        plot_style_sidebar(data_dir, all_files, plot_names)
    
    # Create tabs
    tabs = st.tabs(["Topological & Statistical Distribution"])
    
    # ============================================
    # Tab: Topological & Statistical Distribution
    # ============================================
    with tabs[0]:
        render_topology_stats_tab(
            data_dir, 
            _load_velocity_file,
            get_plot_style_func=get_plot_style,
            apply_plot_style_func=apply_plot_style,
            get_palette_func=_get_palette,
            resolve_line_style_func=resolve_line_style,
            export_panel_func=export_panel,
            capture_button_func=capture_button
        )


if __name__ == "__main__":
    main()
