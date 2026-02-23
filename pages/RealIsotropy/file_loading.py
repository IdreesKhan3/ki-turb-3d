"""
Real Isotropy — File discovery, data loading, session state.
"""

import glob
import re
import streamlit as st
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.file_detector import detect_simulation_files
from core_physics import (
    load_turbulence_data,
    load_reynolds_stress,
    anisotropy_tensor,
    invariants,
)


DEFAULT_LEGENDS = {
    "Ex": "E<sub>x</sub>/E<sub>tot</sub>",
    "Ey": "E<sub>y</sub>/E<sub>tot</sub>",
    "Ez": "E<sub>z</sub>/E<sub>tot</sub>",
    "b11": "b<sub>11</sub>",
    "b22": "b<sub>22</sub>",
    "b33": "b<sub>33</sub>",
    "b12": "|b<sub>12</sub>|",
    "b13": "|b<sub>13</sub>|",
    "b23": "|b<sub>23</sub>|",
    "anis": "Anisotropy index",
    "devx": "devx",
    "devy": "devy",
    "devz": "devz",
    "maxdev": "Max deviation",
}
DEFAULT_AXIS_LABELS = {
    "time": "t/t₀",
    "energy_frac": "Energy fraction",
    "bij": "Anisotropy tensor b<sub>ij</sub>",
    "cross": "Cross-correlations / Anisotropy index",
    "dev": "Absolute deviation",
    "convergence": "Running standard deviation",
    "lumley_x": "ξ = (III<sub>b</sub>/2)<sup>1/3</sup>",
    "lumley_y": "η = (-II<sub>b</sub>/3)<sup>1/2</sup>",
}


def init_session_state():
    """Initialize session state for Real Isotropy page."""
    if "real_iso_legends" not in st.session_state:
        st.session_state.real_iso_legends = DEFAULT_LEGENDS.copy()
    else:
        for key, value in DEFAULT_LEGENDS.items():
            if key not in st.session_state.real_iso_legends:
                st.session_state.real_iso_legends[key] = value

    if "axis_labels_real_iso" not in st.session_state:
        st.session_state.axis_labels_real_iso = DEFAULT_AXIS_LABELS.copy()
    else:
        for key, value in DEFAULT_AXIS_LABELS.items():
            if key not in st.session_state.axis_labels_real_iso:
                st.session_state.axis_labels_real_iso[key] = value

    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}


def _find_eps_file(data_dir: Path) -> Optional[Path]:
    """Find eps_real_validation or turbulence_validation CSV."""
    files = detect_simulation_files(str(data_dir))
    for f in files.get("spectral_turb_stats", []):
        name = Path(f).name
        if name.startswith("eps_real_validation") or name.startswith("turbulence_validation"):
            return Path(f)
    for candidate in ("eps_real_validation.csv", "turbulence_validation.csv"):
        exact_file = data_dir / candidate
        if exact_file.exists():
            return exact_file
    for pattern in ("eps_real_validation*.csv", "turbulence_validation*.csv"):
        matches = glob.glob(str(data_dir / pattern))
        if matches:
            return Path(matches[0])
    return None


def _find_stress_file(data_dir: Path, eps_file: Path) -> Optional[Path]:
    """Find reynolds_stress_validation CSV matching eps file tag."""
    eps_name = eps_file.name
    if "_data" in eps_name:
        tag_match = re.search(r"_data\d+", eps_name)
        if tag_match:
            tag = tag_match.group(0)
            stress_with_tag = data_dir / f"reynolds_stress_validation{tag}.csv"
            if stress_with_tag.exists():
                return stress_with_tag
    exact_stress = data_dir / "reynolds_stress_validation.csv"
    if exact_stress.exists():
        return exact_stress
    matches = glob.glob(str(data_dir / "reynolds_stress_validation*.csv"))
    if matches:
        return Path(matches[0])
    return None


def load_data(data_dir: Path) -> Optional[Tuple[Any, Any, Dict, Dict, float]]:
    """
    Load turbulence data and compute anisotropy.
    Returns (turb, R, b, inv, t0_raw) or None on failure.
    """
    eps_file = _find_eps_file(data_dir)
    if eps_file is None or not eps_file.exists():
        st.error(
            "Validation CSV not found in dataset folder "
            "(eps_real_validation*.csv or turbulence_validation*.csv)"
        )
        st.info("Looking for: eps_real_validation*.csv, turbulence_validation*.csv")
        st.info(f"📂 Current directory: {data_dir}")
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            st.write("Available CSV files in directory:")
            for f in csv_files:
                st.write(f"  - {f.name}")
        return None

    stress_file = _find_stress_file(data_dir, eps_file)
    turb = load_turbulence_data(eps_file)
    R = load_reynolds_stress(stress_file, turb)
    b = anisotropy_tensor(R)
    inv = invariants(b)
    t0_raw = turb["iter"][0] if turb["iter"][0] != 0 else 1.0
    return (turb, R, b, inv, t0_raw)


def render_legend_and_axis_labels():
    """Render legend names and axis labels in sidebar."""
    with st.sidebar.expander("🏷️ Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Curve names")
        for k in st.session_state.real_iso_legends:
            st.session_state.real_iso_legends[k] = st.text_input(
                k, st.session_state.real_iso_legends[k], key=f"realiso_leg_{k}"
            )
        st.markdown("---")
        st.markdown("### Axis labels")
        st.caption("**Which subplot uses each label:**")
        st.caption("• time → X-axis for plots A, C, D, E, F")
        st.caption("• energy_frac → Y-axis for plot A")
        st.caption("• lumley_x → X-axis for plot B")
        st.caption("• lumley_y → Y-axis for plot B")
        st.caption("• bij → Y-axis for plot C")
        st.caption("• cross → Y-axis for plot D")
        st.caption("• dev → Y-axis for plot E")
        st.caption("• convergence → Y-axis for plot F")
        st.markdown("")
        for k in st.session_state.axis_labels_real_iso:
            st.session_state.axis_labels_real_iso[k] = st.text_input(
                k, st.session_state.axis_labels_real_iso[k], key=f"realiso_ax_{k}"
            )
        if st.button("♻️ Reset labels/legends", key="realiso_reset_labels"):
            st.session_state.real_iso_legends = DEFAULT_LEGENDS.copy()
            st.session_state.axis_labels_real_iso = DEFAULT_AXIS_LABELS.copy()
            st.toast("Reset.")
            st.rerun()
