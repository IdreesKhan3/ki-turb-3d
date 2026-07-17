"""
Structure Functions — File discovery, grouping, session state.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.file_detector import (
    detect_simulation_files,
    expand_analysis_search_dirs,
    group_files_by_simulation,
    natural_sort_key,
)


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


def _session_get(key: str, default=None):
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def _collect_structure_files(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    """Collect unique bin/txt structure-function files across session dirs."""
    all_bin: List[str] = []
    all_txt: List[str] = []
    seen: set[str] = set()

    search_roots: List[Path] = []
    seen_roots: set[str] = set()

    # Prefer the explicit OpenLB product folder when session knows it.
    preferred = _session_get("structure_functions_data_directory")
    if preferred:
        data_dirs = [str(preferred), *data_dirs]

    for data_dir_path in data_dirs:
        try:
            root = Path(data_dir_path).resolve()
        except Exception:
            continue
        if not root.is_dir():
            continue
        for candidate in expand_analysis_search_dirs(root):
            key = str(candidate.resolve())
            if key not in seen_roots:
                seen_roots.add(key)
                search_roots.append(candidate)

    for root in search_roots:
        try:
            files_dict = detect_simulation_files(str(root))
        except Exception:
            continue
        for f in files_dict.get("structure_functions_bin", []) or []:
            key = str(Path(f).resolve())
            if key not in seen:
                seen.add(key)
                all_bin.append(key)
        for f in files_dict.get("structure_functions_txt", []) or []:
            key = str(Path(f).resolve())
            if key not in seen:
                seen.add(key)
                all_txt.append(key)

    # Also use session index populated by manifest load.
    indexed = _session_get("all_loaded_files") or {}
    for item in indexed.get("structure_functions_bin") or []:
        path = item.get("full_path") if isinstance(item, dict) else str(item)
        if path:
            key = str(Path(path).resolve())
            if key not in seen:
                seen.add(key)
                all_bin.append(key)
    for item in indexed.get("structure_functions_txt") or []:
        path = item.get("full_path") if isinstance(item, dict) else str(item)
        if path:
            key = str(Path(path).resolve())
            if key not in seen:
                seen.add(key)
                all_txt.append(key)

    all_bin = sorted(all_bin, key=natural_sort_key)
    all_txt = sorted(all_txt, key=natural_sort_key)
    return all_bin, all_txt


def _load_structure_groups(data_dirs: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Detect structure function files and group by simulation prefix.

    Deduplicates across OpenLB raw/ + processed/ expansions so the page does not
    invent fake sims like ``raw_structure_functions1`` vs ``processed_…``.
    """
    all_bin_files, all_txt_files = _collect_structure_files(data_dirs)
    if not all_bin_files and not all_txt_files:
        return None

    sim_groups_bin: Dict[str, List[str]] = {}
    sim_groups_txt: Dict[str, List[str]] = {}

    if all_bin_files:
        sim_groups_bin = group_files_by_simulation(
            all_bin_files,
            r"(structure_funcs\d+)_t\d+\.bin",
        )
        if not sim_groups_bin:
            sim_groups_bin = group_files_by_simulation(
                all_bin_files,
                r"(structure_funcs_data\d+)_t\d+\.bin",
            )
        if not sim_groups_bin:
            sim_groups_bin = {"structure_funcs": all_bin_files}

    if all_txt_files:
        sim_groups_txt = group_files_by_simulation(
            all_txt_files,
            r"(structure_functions\d+)_t\d+\.txt",
        )
        if not sim_groups_txt:
            sim_groups_txt = {"structure_functions": all_txt_files}

    sim_groups: Dict[str, Dict[str, Any]] = {}
    for key, files in sim_groups_bin.items():
        sim_groups[key] = {"kind": "bin", "files": files}
    for key, files in sim_groups_txt.items():
        # Prefer bin group if the same prefix already exists.
        if key not in sim_groups:
            sim_groups[key] = {"kind": "txt", "files": files}
    return sim_groups if sim_groups else None


def load_structure_groups() -> Optional[Tuple[Path, Dict[str, Dict[str, Any]]]]:
    """
    Load structure function files and group by simulation.
    Returns (data_dir, sim_groups) or None on early exit.
    """
    data_dirs = list(st.session_state.get("data_directories", []) or [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]
    preferred = st.session_state.get("structure_functions_data_directory")
    if preferred and str(preferred) not in data_dirs:
        data_dirs = [str(preferred), *data_dirs]

    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return None

    sim_groups = _load_structure_groups(data_dirs)
    if sim_groups is None:
        st.info(
            "No structure function files found. Expected `structure_funcs*_t*.bin` "
            "or `structure_functions*.txt` (including OpenLB `processed/structure_functions/`)."
        )
        return None

    if not sim_groups:
        st.error("No structure function files found or could not group files.")
        return None

    # Prefer product folder for exports when available.
    data_dir = Path(
        preferred
        or next(iter(sim_groups.values()))["files"][0]
    ).resolve()
    if data_dir.is_file():
        data_dir = data_dir.parent
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
