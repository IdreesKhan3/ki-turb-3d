"""
Flatness — File discovery, grouping, session state.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.file_detector import detect_simulation_files, group_files_by_simulation, natural_sort_key

from .data_helpers import format_legend_name


DEFAULT_AXIS_LABELS = {
    "x": "Separation distance r",
    "y": "Longitudinal flatness F<sub>L</sub>(r)",
}


def init_session_state():
    """Initialize session state for Flatness page."""
    st.session_state.setdefault("flatness_legend_names", {})
    st.session_state.setdefault("axis_labels_flatness", DEFAULT_AXIS_LABELS.copy())
    st.session_state.setdefault("plot_styles", {})


def _load_flatness_groups(
    data_dirs: List[str],
) -> Tuple[Optional[Dict[str, List[str]]], Optional[Path]]:
    """
    Detect flatness files from data directories and group by simulation.
    Returns (sim_groups, data_dir) or (None, None) on failure.
    """
    all_flatness_files = []
    for data_dir_path in data_dirs:
        try:
            data_dir_obj = Path(data_dir_path).resolve()
            if data_dir_obj.exists() and data_dir_obj.is_dir():
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_flatness = files_dict.get("flatness", [])
                all_flatness_files.extend(dir_flatness)
        except Exception:
            continue

    if not all_flatness_files:
        return None, None

    data_dir = Path(data_dirs[0]).resolve()

    if len(data_dirs) > 1:
        sim_groups = {}
        for data_dir_path in data_dirs:
            data_dir_obj = Path(data_dir_path).resolve()
            dir_name = data_dir_obj.name
            data_dir_str = str(data_dir_obj)
            dir_flatness = [f for f in all_flatness_files if str(Path(f).resolve().parent) == data_dir_str]
            if not dir_flatness:
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_flatness = [str(f) for f in files_dict.get("flatness", [])]

            if dir_flatness:
                dir_sim_groups = group_files_by_simulation(
                    sorted([str(f) for f in dir_flatness], key=natural_sort_key),
                    r"(flatness_data\d+)_t\d+\.txt",
                )
                if not dir_sim_groups:
                    dir_sim_groups = group_files_by_simulation(
                        sorted([str(f) for f in dir_flatness], key=natural_sort_key),
                        r"(flatness_data\d+)_\d+\.txt",
                    )
                if not dir_sim_groups:
                    dir_sim_groups = group_files_by_simulation(
                        sorted([str(f) for f in dir_flatness], key=natural_sort_key),
                        r"(flatness\d+)_t\d+\.txt",
                    )
                if dir_sim_groups:
                    for key, files in dir_sim_groups.items():
                        new_key = f"{dir_name}_{key}" if key else dir_name
                        sim_groups[new_key] = files
                else:
                    sim_groups[dir_name] = sorted([str(f) for f in dir_flatness], key=natural_sort_key)
    else:
        sim_groups = group_files_by_simulation(
            sorted([str(f) for f in all_flatness_files], key=natural_sort_key),
            r"(flatness_data\d+)_t\d+\.txt",
        ) if all_flatness_files else {}
        if not sim_groups and all_flatness_files:
            sim_groups = group_files_by_simulation(
                sorted([str(f) for f in all_flatness_files], key=natural_sort_key),
                r"(flatness_data\d+)_\d+\.txt",
            )
        if not sim_groups and all_flatness_files:
            sim_groups = group_files_by_simulation(
                sorted([str(f) for f in all_flatness_files], key=natural_sort_key),
                r"(flatness\d+)_t\d+\.txt",
            )
        if not sim_groups and all_flatness_files:
            sim_groups = {"flatness": sorted([str(f) for f in all_flatness_files], key=natural_sort_key)}

    if not sim_groups:
        return None, data_dir
    return sim_groups, data_dir


def load_flatness_groups() -> Optional[Tuple[Path, Dict[str, List[str]]]]:
    """
    Load flatness files and group by simulation.
    Returns (data_dir, sim_groups) or None on early exit.
    """
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]

    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return None

    sim_groups, data_dir = _load_flatness_groups(data_dirs)

    if sim_groups is None:
        st.info("No flatness files found. Expected format: `flatness_data*_*.txt`")
        return None

    if not sim_groups:
        st.warning("Could not group flatness files by simulation type.")
        return None

    return (data_dir, sim_groups)


def render_legend_and_axis_labels(sim_groups: Dict[str, List[str]]):
    """Render legend names and axis labels in sidebar."""
    with st.sidebar.expander("🏷️ Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Legend names")
        for sim_prefix in sorted(sim_groups.keys()):
            st.session_state.flatness_legend_names.setdefault(
                sim_prefix, format_legend_name(sim_prefix)
            )
            st.session_state.flatness_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.flatness_legend_names[sim_prefix],
                key=f"legend_flat_{sim_prefix}",
            )
        st.markdown("---")
        st.markdown("### Axis labels")
        st.session_state.axis_labels_flatness["x"] = st.text_input(
            "X-axis label",
            value=st.session_state.axis_labels_flatness.get("x"),
            key="axis_flat_x",
        )
        st.session_state.axis_labels_flatness["y"] = st.text_input(
            "Y-axis label",
            value=st.session_state.axis_labels_flatness.get("y"),
            key="axis_flat_y",
        )
        if st.button("♻️ Reset labels/legends", key="flatness_reset_labels"):
            st.session_state.flatness_legend_names = {
                k: format_legend_name(k) for k in sim_groups.keys()
            }
            st.session_state.axis_labels_flatness = DEFAULT_AXIS_LABELS.copy()
            st.toast("Reset.")
            st.rerun()
