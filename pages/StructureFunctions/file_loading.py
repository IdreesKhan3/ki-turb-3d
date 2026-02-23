"""
Structure Functions — File discovery, grouping, session state.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.file_detector import detect_simulation_files, group_files_by_simulation, natural_sort_key


DEFAULT_AXIS_LABELS = {
    "x_r": "Separation distance r",
    "y_sp": "Structure functions S<sub>p</sub>(r)",
    "x_ess": "S<sub>3</sub>(r)",
    "y_ess": "S<sub>p</sub>(r)",
    "x_anom": "p",
    "y_anom": "ξ<sub>p</sub> - p/3",
    "x_inset": "p",
    "y_inset": "ξ<sub>p</sub> - p/3",
    "inset_legend_sl": "SL94",
    "inset_legend_b93": "B93",
}


def init_session_state():
    """Initialize session state for Structure Functions page."""
    st.session_state.setdefault("structure_legend_names", {})
    st.session_state.setdefault("axis_labels_structure", DEFAULT_AXIS_LABELS.copy())
    st.session_state.setdefault("plot_styles", {})


def _load_structure_groups(data_dirs: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Detect structure function files and group by simulation.
    Returns sim_groups: {prefix: {"kind": "bin"|"txt", "files": [...]}} or None.
    """
    all_bin_files = []
    all_txt_files = []
    for data_dir_path in data_dirs:
        try:
            data_dir_obj = Path(data_dir_path).resolve()
            if data_dir_obj.exists() and data_dir_obj.is_dir():
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_bin = files_dict.get("structure_functions_bin", [])
                dir_txt = files_dict.get("structure_functions_txt", [])
                all_bin_files.extend([str(f) for f in dir_bin])
                all_txt_files.extend([str(f) for f in dir_txt])
        except Exception:
            continue

    if not all_bin_files and not all_txt_files:
        return None

    sim_groups_bin: Dict[str, List[str]] = {}
    sim_groups_txt: Dict[str, List[str]] = {}

    if len(data_dirs) > 1:
        for data_dir_path in data_dirs:
            data_dir_obj = Path(data_dir_path).resolve()
            dir_name = data_dir_obj.name
            data_dir_str = str(data_dir_obj)
            dir_bin = [f for f in all_bin_files if str(Path(f).resolve().parent) == data_dir_str]
            dir_txt = [f for f in all_txt_files if str(Path(f).resolve().parent) == data_dir_str]
            if not dir_bin and not dir_txt:
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_bin = [str(f) for f in files_dict.get("structure_functions_bin", [])]
                dir_txt = [str(f) for f in files_dict.get("structure_functions_txt", [])]
            if dir_bin:
                dir_sim_groups_bin = group_files_by_simulation(
                    sorted([str(f) for f in dir_bin], key=natural_sort_key),
                    r"(structure_funcs\d+)_t\d+\.bin",
                )
                if not dir_sim_groups_bin:
                    dir_sim_groups_bin = group_files_by_simulation(
                        sorted([str(f) for f in dir_bin], key=natural_sort_key),
                        r"(structure_funcs_data\d+)_t\d+\.bin",
                    )
                if dir_sim_groups_bin:
                    for key, files in dir_sim_groups_bin.items():
                        new_key = f"{dir_name}_{key}" if key else dir_name
                        sim_groups_bin[new_key] = files
                else:
                    sim_groups_bin[dir_name] = sorted([str(f) for f in dir_bin], key=natural_sort_key)
            if dir_txt:
                dir_sim_groups_txt = group_files_by_simulation(
                    sorted([str(f) for f in dir_txt], key=natural_sort_key),
                    r"(structure_functions\d+)_t\d+\.txt",
                )
                if dir_sim_groups_txt:
                    for key, files in dir_sim_groups_txt.items():
                        new_key = f"{dir_name}_{key}" if key else dir_name
                        sim_groups_txt[new_key] = files
                else:
                    sim_groups_txt[dir_name] = sorted([str(f) for f in dir_txt], key=natural_sort_key)
    else:
        sim_groups_bin = group_files_by_simulation(
            sorted([str(f) for f in all_bin_files], key=natural_sort_key),
            r"(structure_funcs\d+)_t\d+\.bin",
        ) if all_bin_files else {}
        if not sim_groups_bin and all_bin_files:
            sim_groups_bin = group_files_by_simulation(
                sorted([str(f) for f in all_bin_files], key=natural_sort_key),
                r"(structure_funcs_data\d+)_t\d+\.bin",
            )
        sim_groups_txt = group_files_by_simulation(
            sorted([str(f) for f in all_txt_files], key=natural_sort_key),
            r"(structure_functions\d+)_t\d+\.txt",
        ) if all_txt_files else {}
        if not sim_groups_bin and not sim_groups_txt:
            if all_bin_files:
                sim_groups_bin["structure_funcs"] = sorted([str(f) for f in all_bin_files], key=natural_sort_key)
            elif all_txt_files:
                sim_groups_txt["structure_funcs"] = sorted([str(f) for f in all_txt_files], key=natural_sort_key)

    sim_groups: Dict[str, Dict[str, Any]] = {}
    for k, v in sim_groups_bin.items():
        sim_groups[k] = {"kind": "bin", "files": v}
    for k, v in sim_groups_txt.items():
        if k not in sim_groups:
            sim_groups[k] = {"kind": "txt", "files": v}
    return sim_groups if sim_groups else None


def load_structure_groups() -> Optional[Tuple[Path, Dict[str, Dict[str, Any]]]]:
    """
    Load structure function files and group by simulation.
    Returns (data_dir, sim_groups) or None on early exit.
    """
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]

    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return None

    sim_groups = _load_structure_groups(data_dirs)
    if sim_groups is None:
        st.info("No structure function files found. Expected `structure_funcs*_t*.bin` or `structure_functions*_t*.txt`.")
        return None

    if not sim_groups:
        st.error("No structure function files found or could not group files.")
        return None

    data_dir = Path(data_dirs[0]).resolve()
    return (data_dir, sim_groups)


def render_legend_and_axis_labels(sim_groups: Dict[str, Dict[str, Any]]):
    """Render legend names and axis labels in sidebar."""
    with st.sidebar.expander("🏷️ Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Legend names")
        for sim_prefix in sorted(sim_groups.keys()):
            st.session_state.structure_legend_names.setdefault(sim_prefix, sim_prefix.replace("_", " ").title())
            st.session_state.structure_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.structure_legend_names[sim_prefix],
                key=f"legend_struct_{sim_prefix}",
            )
        st.markdown("---")
        st.markdown("### Axis labels")
        st.session_state.axis_labels_structure["x_r"] = st.text_input("S_p plot x-label", st.session_state.axis_labels_structure.get("x_r", "Separation distance r"), key="ax_struct_xr")
        st.session_state.axis_labels_structure["y_sp"] = st.text_input("S_p plot y-label", st.session_state.axis_labels_structure.get("y_sp", "Structure functions S<sub>p</sub>(r)"), key="ax_struct_ysp")
        st.session_state.axis_labels_structure["x_ess"] = st.text_input("ESS x-label", st.session_state.axis_labels_structure.get("x_ess", "S<sub>3</sub>(r)"), key="ax_struct_xess")
        st.session_state.axis_labels_structure["y_ess"] = st.text_input("ESS y-label", st.session_state.axis_labels_structure.get("y_ess", "S<sub>p</sub>(r)"), key="ax_struct_yess")
        st.session_state.axis_labels_structure["x_anom"] = st.text_input("Anomaly x-label", st.session_state.axis_labels_structure.get("x_anom", "p"), key="ax_struct_xanom")
        st.session_state.axis_labels_structure["y_anom"] = st.text_input("Anomaly y-label", st.session_state.axis_labels_structure.get("y_anom", "ξ<sub>p</sub> - p/3"), key="ax_struct_yanom")
        st.markdown("---")
        st.markdown("### Inset labels")
        st.session_state.axis_labels_structure["x_inset"] = st.text_input("Inset x-label", st.session_state.axis_labels_structure.get("x_inset", "p"), key="ax_struct_x_inset")
        st.session_state.axis_labels_structure["y_inset"] = st.text_input("Inset y-label", st.session_state.axis_labels_structure.get("y_inset", "ξ<sub>p</sub> - p/3"), key="ax_struct_y_inset")
        st.session_state.axis_labels_structure["inset_legend_sl"] = st.text_input("Inset She-Leveque legend", st.session_state.axis_labels_structure.get("inset_legend_sl", "SL94"), key="ax_struct_legend_sl")
        st.session_state.axis_labels_structure["inset_legend_b93"] = st.text_input("Inset B93 legend", st.session_state.axis_labels_structure.get("inset_legend_b93", "B93"), key="ax_struct_legend_b93")
        if st.button("♻️ Reset labels/legends", key="structure_reset_labels"):
            st.session_state.structure_legend_names = {k: k.replace("_", " ").title() for k in sim_groups.keys()}
            st.session_state.axis_labels_structure.update(DEFAULT_AXIS_LABELS)
            st.toast("Reset.")
            st.rerun()
