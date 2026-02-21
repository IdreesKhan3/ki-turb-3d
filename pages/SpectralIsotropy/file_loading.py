"""
Spectral Isotropy — File discovery, grouping, session state.
"""

import glob
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.file_detector import detect_simulation_files, natural_sort_key, group_files_by_simulation


def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()


DEFAULT_LEGENDS = {
    "IC": "IC(k) (time-avg)",
    "IC_snap": "IC(k) snapshots",
    "E11": "E<sub>11</sub>(k)",
    "E22": "E<sub>22</sub>(k)",
    "E33": "E<sub>33</sub>(k)",
}
DEFAULT_AXIS_LABELS = {"k": "k", "ic": "IC(k)", "ek": "E<sub>ii</sub>(k)"}


def init_session_state():
    """Initialize session state for Spectral Isotropy page."""
    if "spec_iso_legends" not in st.session_state:
        st.session_state.spec_iso_legends = DEFAULT_LEGENDS.copy()
    else:
        for key, value in DEFAULT_LEGENDS.items():
            if key not in st.session_state.spec_iso_legends:
                st.session_state.spec_iso_legends[key] = value

    if "axis_labels_spec_iso" not in st.session_state:
        st.session_state.axis_labels_spec_iso = DEFAULT_AXIS_LABELS.copy()
    else:
        for key, value in DEFAULT_AXIS_LABELS.items():
            if key not in st.session_state.axis_labels_spec_iso:
                st.session_state.axis_labels_spec_iso[key] = value

    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    st.session_state.setdefault("spec_iso_sim_legend_names", {})


def load_ic_groups() -> Optional[Tuple[Path, Dict[str, List[str]]]]:
    """
    Load isotropy_coeff_*.dat files and group by simulation.
    Returns (data_dir, ic_groups) or None on early exit.
    """
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]

    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return None

    data_dir = Path(data_dirs[0]).resolve()
    ic_groups: Dict[str, List[str]] = {}

    for data_dir_path in data_dirs:
        try:
            dir_path = Path(data_dir_path).resolve()
            if not dir_path.exists() or not dir_path.is_dir():
                continue
            files = detect_simulation_files(str(dir_path))
            dir_ic_files = files.get("isotropy", [])
            if not dir_ic_files:
                dir_ic_files = glob.glob(str(dir_path / "isotropy_coeff_*.dat"))

            if not dir_ic_files:
                continue

            dir_name = dir_path.name
            dir_ic_str = [str(f) for f in dir_ic_files]
            grouped = group_files_by_simulation(dir_ic_str, r"(isotropy_coeff[_\w]*\d+)_\d+\.dat")
            if not grouped:
                grouped = group_files_by_simulation(dir_ic_str, r"isotropy_coeff_(\d+)_\d+\.dat")

            if grouped:
                for key, file_list in grouped.items():
                    new_key = f"{dir_name}_{key}" if len(data_dirs) > 1 else key
                    if new_key not in ic_groups:
                        ic_groups[new_key] = []
                    ic_groups[new_key].extend(file_list)
            else:
                group_key = dir_name if len(data_dirs) > 1 else "default"
                if group_key not in ic_groups:
                    ic_groups[group_key] = []
                ic_groups[group_key].extend(dir_ic_str)
        except Exception:
            continue

    for key in ic_groups:
        ic_groups[key] = sorted(ic_groups[key], key=natural_sort_key)

    if not ic_groups:
        st.info("No isotropy_coeff_*.dat files found in any of the selected directories.")
        return None

    return (data_dir, ic_groups)


def render_legend_and_axis_labels(ic_groups: Dict[str, List[str]]):
    """Render legend names and axis labels in sidebar."""
    with st.sidebar.expander("🏷️ Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Simulation legend names")
        for sim_prefix in sorted(ic_groups.keys()):
            st.session_state.spec_iso_sim_legend_names.setdefault(
                sim_prefix, _default_labelify(sim_prefix)
            )
            st.session_state.spec_iso_sim_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.spec_iso_sim_legend_names[sim_prefix],
                key=f"speciso_sim_leg_{sim_prefix}",
            )
        st.markdown("---")
        st.markdown("### Curve names")
        for k in st.session_state.spec_iso_legends:
            st.session_state.spec_iso_legends[k] = st.text_input(
                k, st.session_state.spec_iso_legends[k], key=f"speciso_leg_{k}"
            )
        st.markdown("---")
        st.markdown("### Axis labels")
        for k in st.session_state.axis_labels_spec_iso:
            st.session_state.axis_labels_spec_iso[k] = st.text_input(
                k, st.session_state.axis_labels_spec_iso[k], key=f"speciso_ax_{k}"
            )
        if st.button("♻️ Reset labels/legends", key="speciso_reset_labels"):
            st.session_state.spec_iso_sim_legend_names = {
                k: _default_labelify(k) for k in ic_groups.keys()
            }
            st.session_state.spec_iso_legends = DEFAULT_LEGENDS.copy()
            st.session_state.axis_labels_spec_iso = DEFAULT_AXIS_LABELS.copy()
            st.toast("Reset.")
            st.rerun()
