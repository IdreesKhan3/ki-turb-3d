"""
Energy Spectra — File discovery, grouping, session state.
"""

import glob
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.file_detector import natural_sort_key, group_files_by_simulation


def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()


def init_session_state():
    """Initialize session state for Energy Spectra page."""
    st.session_state.setdefault("spectrum_legend_names", {})
    st.session_state.setdefault("norm_legend_names", {})
    st.session_state.setdefault("axis_labels_raw", {"x": "Wavenumber k", "y": "Energy spectrum E(k)"})
    st.session_state.setdefault("axis_labels_norm", {
        "x": "Normalized wavenumber kη",
        "y": "Normalized spectrum E<sub>norm</sub>(kη)",
    })
    st.session_state.setdefault("plot_styles", {})
    st.session_state.setdefault("file_selection_mode", "directory")
    st.session_state.setdefault("custom_spectrum_files", [])
    st.session_state.setdefault("custom_norm_files", [])


def load_files_and_groups() -> Optional[Tuple[Path, Dict[str, List[str]], Dict[str, List[str]]]]:
    """
    Load spectrum/norm files and group by simulation.
    Renders file selection UI. Returns (data_dir, sim_groups, norm_groups) or None on early exit.
    """
    st.sidebar.header("📁 File Selection")
    file_mode = st.sidebar.radio(
        "Selection Mode",
        ["Directory (Auto-detect)", "Custom Files (Any location/name)"],
        index=0 if st.session_state.file_selection_mode == "directory" else 1,
        key="file_mode_radio",
    )
    st.session_state.file_selection_mode = "directory" if file_mode == "Directory (Auto-detect)" else "custom"

    spectrum_files: List[str] = []
    norm_files: List[str] = []
    data_dir: Path

    if st.session_state.file_selection_mode == "directory":
        data_dirs = st.session_state.get("data_directories", [])
        if not data_dirs and st.session_state.get("data_directory"):
            data_dirs = [st.session_state.data_directory]

        if not data_dirs:
            st.warning("Please select a data directory from the Overview page.")
            return None

        for data_dir_path in data_dirs:
            try:
                d = Path(data_dir_path).resolve()
                if d.exists() and d.is_dir():
                    dir_spectrum = sorted(glob.glob(str(d / "spectrum*.dat")), key=natural_sort_key)
                    dir_norm = sorted(glob.glob(str(d / "norm*.dat")), key=natural_sort_key)
                    spectrum_files.extend(dir_spectrum)
                    norm_files.extend(dir_norm)
            except Exception:
                continue

        if not spectrum_files and not norm_files:
            st.error("No spectrum*.dat or norm*.dat files found in the selected directories.")
            st.info(
                "Switch to 'Custom Files' mode to select files from any location, "
                "or select multiple directories in the main app."
            )
            return None

        data_dir = Path(data_dirs[0]).resolve()

    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Custom File Selection")
        st.sidebar.markdown("**Raw Spectrum Files** (k, E(k)):")
        raw_file_input = st.sidebar.text_area(
            "Enter file paths (one per line):",
            value="\n".join(st.session_state.custom_spectrum_files),
            height=100,
            help="Enter full paths to spectrum files, one per line. Files can be from any directory.",
            key="energy_raw_files",
        )
        st.sidebar.markdown("**Normalized Spectrum Files** (kη, E_norm, E_pope):")
        norm_file_input = st.sidebar.text_area(
            "Enter file paths (one per line):",
            value="\n".join(st.session_state.custom_norm_files),
            height=100,
            help="Enter full paths to normalized spectrum files, one per line.",
            key="energy_norm_files",
        )

        if st.sidebar.button("Load Custom Files", type="primary", key="energy_load_custom"):
            raw_paths = [p.strip() for p in raw_file_input.strip().split("\n") if p.strip()]
            norm_paths = [p.strip() for p in norm_file_input.strip().split("\n") if p.strip()]
            valid_raw = []
            valid_norm = []
            for p in raw_paths:
                path = Path(p)
                if path.exists() and path.is_file():
                    valid_raw.append(str(path.absolute()))
                else:
                    st.sidebar.warning(f"File not found: {p}")
            for p in norm_paths:
                path = Path(p)
                if path.exists() and path.is_file():
                    valid_norm.append(str(path.absolute()))
                else:
                    st.sidebar.warning(f"File not found: {p}")
            st.session_state.custom_spectrum_files = valid_raw
            st.session_state.custom_norm_files = valid_norm
            if valid_raw or valid_norm:
                st.sidebar.success(f"Loaded {len(valid_raw)} raw + {len(valid_norm)} normalized files")
            else:
                st.sidebar.error("No valid files found. Check paths.")

        spectrum_files = [str(Path(f).absolute()) for f in st.session_state.custom_spectrum_files if Path(f).exists()]
        norm_files = [str(Path(f).absolute()) for f in st.session_state.custom_norm_files if Path(f).exists()]

        if not spectrum_files and not norm_files:
            st.info("👈 Use the sidebar to enter file paths, then click 'Load Custom Files'.")
            return None

        if spectrum_files:
            data_dir = Path(spectrum_files[0]).parent
        elif norm_files:
            data_dir = Path(norm_files[0]).parent
        else:
            data_dir = Path.cwd()

    # Group files by simulation
    if st.session_state.file_selection_mode == "directory":
        data_dirs = st.session_state.get("data_directories", [])
        if not data_dirs and st.session_state.get("data_directory"):
            data_dirs = [st.session_state.data_directory]

        if len(data_dirs) > 1:
            sim_groups: Dict[str, List[str]] = {}
            norm_groups: Dict[str, List[str]] = {}
            for data_dir_path in data_dirs:
                d = Path(data_dir_path).resolve()
                dir_name = d.name
                dir_spectrum = [f for f in spectrum_files if Path(f).resolve().parent == d]
                dir_norm = [f for f in norm_files if Path(f).resolve().parent == d]
                dir_sim_groups = (
                    group_files_by_simulation(dir_spectrum, r"(spectrum[_\w]*\d+)_\d+\.dat")
                    if dir_spectrum else {}
                )
                dir_norm_groups = (
                    group_files_by_simulation(dir_norm, r"(norm[_\w]*\d+)_\d+\.dat")
                    if dir_norm else {}
                )
                if not dir_sim_groups and dir_spectrum:
                    dir_sim_groups = group_files_by_simulation(dir_spectrum, r"(spectrum\d+)_\d+\.dat")
                if not dir_norm_groups and dir_norm:
                    dir_norm_groups = group_files_by_simulation(dir_norm, r"(norm\d+)_\d+\.dat")
                for key, files in dir_sim_groups.items():
                    new_key = f"{dir_name}_{key}" if key else dir_name
                    sim_groups[new_key] = files
                for key, files in dir_norm_groups.items():
                    new_key = f"{dir_name}_{key}" if key else dir_name
                    norm_groups[new_key] = files
        else:
            sim_groups = (
                group_files_by_simulation(spectrum_files, r"(spectrum[_\w]*\d+)_\d+\.dat")
                if spectrum_files else {}
            )
            norm_groups = (
                group_files_by_simulation(norm_files, r"(norm[_\w]*\d+)_\d+\.dat")
                if norm_files else {}
            )
            if not sim_groups and spectrum_files:
                sim_groups = group_files_by_simulation(spectrum_files, r"(spectrum\d+)_\d+\.dat")
            if not norm_groups and norm_files:
                norm_groups = group_files_by_simulation(norm_files, r"(norm\d+)_\d+\.dat")
    else:
        sim_groups = {}
        unique_dirs = set(Path(str(f)).parent for f in spectrum_files)
        for f in spectrum_files:
            fpath = Path(f)
            group_key = fpath.parent.name if len(unique_dirs) > 1 else (
                fpath.stem.rsplit("_", 1)[0] if "_" in fpath.stem else fpath.stem
            )
            if group_key not in sim_groups:
                sim_groups[group_key] = []
            sim_groups[group_key].append(str(fpath))
        for key in sim_groups:
            sim_groups[key] = sorted(sim_groups[key], key=natural_sort_key)

        norm_groups = {}
        unique_norm_dirs = set(Path(str(f)).parent for f in norm_files)
        for f in norm_files:
            fpath = Path(f)
            stem = fpath.stem
            if len(unique_norm_dirs) > 1:
                group_key = fpath.parent.name
            else:
                group_key = stem.rsplit("_", 1)[0] if "_" in stem and stem.rsplit("_", 1)[1].isdigit() else stem
            if group_key not in norm_groups:
                norm_groups[group_key] = []
            norm_groups[group_key].append(str(fpath))
        for key in norm_groups:
            norm_groups[key] = sorted(norm_groups[key], key=natural_sort_key)

    return (data_dir, sim_groups, norm_groups)


def render_legend_and_axis_labels(
    sim_groups: Dict[str, List[str]],
    norm_groups: Dict[str, List[str]],
):
    """Render legend names and axis labels in sidebar."""
    with st.sidebar.expander("🏷️ Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Legend names")
        if sim_groups:
            st.markdown("**Raw spectra:**")
            for sim_prefix in sorted(sim_groups.keys()):
                st.session_state.spectrum_legend_names.setdefault(
                    sim_prefix, _default_labelify(sim_prefix)
                )
                st.session_state.spectrum_legend_names[sim_prefix] = st.text_input(
                    f"Name for `{sim_prefix}`",
                    value=st.session_state.spectrum_legend_names[sim_prefix],
                    key=f"legend_raw_{sim_prefix}",
                )
        if norm_groups:
            st.markdown("**Normalized spectra:**")
            for norm_prefix in sorted(norm_groups.keys()):
                st.session_state.norm_legend_names.setdefault(
                    norm_prefix, _default_labelify(norm_prefix)
                )
                st.session_state.norm_legend_names[norm_prefix] = st.text_input(
                    f"Name for `{norm_prefix}`",
                    value=st.session_state.norm_legend_names[norm_prefix],
                    key=f"legend_norm_{norm_prefix}",
                )
        st.markdown("---")
        st.markdown("### Axis labels")
        st.caption("Raw spectrum labels")
        st.session_state.axis_labels_raw["x"] = st.text_input(
            "Raw x-axis label",
            value=st.session_state.axis_labels_raw.get("x", "Wavenumber k"),
            key="axis_raw_x",
        )
        st.session_state.axis_labels_raw["y"] = st.text_input(
            "Raw y-axis label",
            value=st.session_state.axis_labels_raw.get("y", "Energy spectrum E(k)"),
            key="axis_raw_y",
        )
        st.caption("Normalized spectrum labels")
        st.session_state.axis_labels_norm["x"] = st.text_input(
            "Norm x-axis label",
            value=st.session_state.axis_labels_norm.get("x", "Normalized wavenumber kη"),
            key="axis_norm_x",
        )
        st.session_state.axis_labels_norm["y"] = st.text_input(
            "Norm y-axis label",
            value=st.session_state.axis_labels_norm.get("y", "Normalized spectrum E<sub>norm</sub>(kη)"),
            key="axis_norm_y",
        )
        st.markdown("---")
        if st.button("♻️ Reset labels/legends", key="energy_reset_labels"):
            st.session_state.spectrum_legend_names = {k: _default_labelify(k) for k in sim_groups.keys()}
            st.session_state.norm_legend_names = {k: _default_labelify(k) for k in norm_groups.keys()}
            st.session_state.axis_labels_raw = {"x": "Wavenumber k", "y": "Energy spectrum E(k)"}
            st.session_state.axis_labels_norm = {
                "x": "Normalized wavenumber kη",
                "y": "Normalized spectrum E<sub>norm</sub>(kη)",
            }
            st.toast("Reset.")
            st.rerun()
