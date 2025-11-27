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
    
    # Create tabs
    tabs = st.tabs(["Topological & Statistical Distribution"])
    
    # ============================================
    # Tab: Topological & Statistical Distribution
    # ============================================
    with tabs[0]:
        render_topology_stats_tab(data_dir, _load_velocity_file)


if __name__ == "__main__":
    main()
