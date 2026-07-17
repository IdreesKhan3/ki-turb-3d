"""
3D Volume Viewer — File discovery, file type selection, time control.
"""

import re
import streamlit as st
from pathlib import Path
from typing import List, Optional, Tuple

from utils.file_detector import list_velocity_field_files, natural_sort_key

from .data_helpers import cached_read_vti, cached_read_hdf5


def collect_volume_files(data_dirs: List[str]) -> Tuple[List[str], List[str], Optional[Path]]:
    """
    Collect velocity VTI and HDF5 files from all data directories.

    Excludes OpenLB companion dumps such as ``density_*.vti`` / ``vorticity_*.vti``
    (those are not 3-component velocity and break ``read_vti_file``).

    Returns (vti_files, hdf5_files, data_dir) where data_dir is first dir for metadata.
    """
    all_vti: List[str] = []
    all_hdf5: List[str] = []
    data_dir = None

    for data_dir_path in data_dirs:
        try:
            d = Path(data_dir_path).resolve()
            if d.exists() and d.is_dir() and data_dir is None:
                data_dir = d
        except Exception:
            continue

    for path in list_velocity_field_files(data_dirs):
        suffix = Path(path).suffix.lower()
        if suffix == ".vti":
            all_vti.append(path)
        elif suffix in {".h5", ".hdf5"}:
            all_hdf5.append(path)

    all_vti = sorted(all_vti, key=natural_sort_key)
    all_hdf5 = sorted(all_hdf5, key=natural_sort_key)
    return all_vti, all_hdf5, data_dir


def extract_iterations(all_files: List[str]) -> List[Optional[int]]:
    """Extract iteration numbers from filenames. Returns list of int or None."""
    iterations = []
    for f in all_files:
        filename = Path(f).name
        match = re.search(r"_(\d+)\.(vti|h5|hdf5)", filename, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+)\.(vti|h5|hdf5)", filename, re.IGNORECASE)
        if match:
            iterations.append(int(match.group(1)))
        else:
            iterations.append(None)
    return iterations


def load_volume_data() -> Optional[dict]:
    """
    Load volume viewer state: collect files, render file type + time controls.
    Returns dict with keys: data_dir, all_files, file_index, selected_file, iterations,
    initial_load, vti_files, hdf5_files; or None on early exit.
    """
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]

    if not data_dirs:
        st.warning("Please select a data directory from the main page.")
        return None

    vti_files, hdf5_files, data_dir = collect_volume_files(data_dirs)
    has_vti = len(vti_files) > 0
    has_hdf5 = len(hdf5_files) > 0

    if not has_vti and not has_hdf5:
        st.error("No 3D velocity files found in any of the selected directories.")
        st.info(
            "Expected files: `*.vti`, `*.h5`, or `*.hdf5` "
            "(e.g., `velocity_50000.vti` or `velocity_50000.h5`)"
        )
        return None

    file_type_options = []
    if has_vti:
        file_type_options.append(f"VTI ({len(vti_files)} files)")
    if has_hdf5:
        file_type_options.append(f"HDF5 ({len(hdf5_files)} files)")
    if has_vti and has_hdf5:
        file_type_options.append("Both (VTI + HDF5)")

    if "file_type_selection" not in st.session_state:
        st.session_state.file_type_selection = (
            "Both (VTI + HDF5)" if "Both" in file_type_options else file_type_options[0]
        )

    st.sidebar.header("📁 File Selection")
    selected_file_type = st.sidebar.radio(
        "File Extension",
        options=file_type_options,
        index=file_type_options.index(st.session_state.file_type_selection)
        if st.session_state.file_type_selection in file_type_options
        else 0,
        key="file_type_radio",
    )
    st.session_state.file_type_selection = selected_file_type

    if selected_file_type.startswith("VTI"):
        all_files = vti_files
    elif selected_file_type.startswith("HDF5"):
        all_files = hdf5_files
    else:
        all_files = vti_files + hdf5_files

    iterations = extract_iterations(all_files)

    st.sidebar.header("⏱️ Time Control")
    st.sidebar.caption(f"Found {len(all_files)} files")

    if "prev_file_type" not in st.session_state:
        st.session_state.prev_file_type = selected_file_type
        st.session_state.file_index = 0
        st.session_state.initial_load = True
    elif st.session_state.prev_file_type != selected_file_type:
        st.session_state.file_index = 0
        st.session_state.prev_file_type = selected_file_type
        st.session_state.initial_load = True

    if "file_index" not in st.session_state:
        st.session_state.file_index = 0
        st.session_state.initial_load = True

    initial_load = st.session_state.get("initial_load", False)
    st.session_state.file_index = max(
        0, min(st.session_state.file_index, len(all_files) - 1)
    )

    col_t1, col_t2, col_t3 = st.sidebar.columns([1, 2, 1])

    if len(all_files) == 1:
        file_index = 0
        st.session_state.file_index = 0
        col_t2.caption("Time Step: 0 (1 file)")
    else:
        if col_t1.button("◀", key="prev_file", help="Previous time step"):
            if st.session_state.file_index > 0:
                st.session_state.file_index -= 1
                st.session_state.slider_index = st.session_state.file_index

        if col_t3.button("▶", key="next_file", help="Next time step"):
            if st.session_state.file_index < len(all_files) - 1:
                st.session_state.file_index += 1
                st.session_state.slider_index = st.session_state.file_index

        file_index = col_t2.slider(
            "Time Step",
            0,
            len(all_files) - 1,
            value=st.session_state.file_index,
            key="slider_index",
        )
        if file_index != st.session_state.file_index:
            st.session_state.file_index = file_index
        file_index = max(0, min(file_index, len(all_files) - 1))

    selected_file = all_files[file_index]
    filename = Path(selected_file).name
    iteration = iterations[file_index]

    st.sidebar.caption(f"File: {filename}")
    if iteration is not None:
        st.sidebar.caption(f"Iteration: {iteration}")
    else:
        st.sidebar.caption(f"Time Step: {file_index}")

    if initial_load:
        cached_read_vti.clear()
        cached_read_hdf5.clear()
        st.session_state.initial_load = False

    return {
        "data_dir": data_dir,
        "all_files": all_files,
        "file_index": file_index,
        "selected_file": selected_file,
        "iterations": iterations,
        "initial_load": initial_load,
        "vti_files": vti_files,
        "hdf5_files": hdf5_files,
    }
