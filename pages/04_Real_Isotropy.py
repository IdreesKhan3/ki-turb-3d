"""
Isotropy Validation (Real Space) Page — Streamlit

High-standard features:
- Reads real-space isotropy files (LBM/NS):
    * eps_real_validation*.csv or turbulence_validation*.csv (required)
    * reynolds_stress_validation*.csv (optional)
- Computes anisotropy tensor b_ij and Pope/Lumley invariants
- Produces 6 interactive subplots:
    (a) Energy fractions vs t/t0 + moving averages + tolerance bands
    (b) Lumley triangle (xi, eta) trajectory
    (c) b11, b22, b33 vs t/t0
    (d) |b12|, |b13|, |b23| + anisotropy index
    (e) energy-fraction deviations from isotropy
    (f) convergence (running std)
- Full user controls (in-memory session state)
- Research-grade export (requires kaleido)

Requires kaleido:
    pip install -U kaleido
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from content.real_isotropy_theory_content import get_real_isotropy_theory_markdown
from pages.RealIsotropy import (
    init_session_state,
    load_data,
    render_legend_and_axis_labels,
    plot_style_sidebar,
    render_tab1,
    render_tab2,
    render_tab3,
    render_summary,
)

st.set_page_config(page_icon="⚫")

CURVES = [
    "Ex", "Ey", "Ez", "b11", "b22", "b33",
    "b12", "b13", "b23", "anis", "devx", "devy", "devz", "maxdev",
]
PLOT_NAMES = [
    "Energy Fractions (A)",
    "Lumley Triangle (B)",
    "Diagonal b_ii (C)",
    "Cross-correlations (D)",
    "Deviations (E)",
    "Convergence (F)",
]


def main():
    inject_theme_css()
    st.title("Isotropy Validation — Real Space")

    data_dir = st.session_state.get("data_directory", None)
    if not data_dir:
        st.warning("Please select a data directory from the Overview page.")
        return
    data_dir = Path(data_dir)

    init_session_state()
    result = load_data(data_dir)
    if result is None:
        return

    turb, R, b, inv, t0_raw = result
    E_x, E_y, E_z = turb["frac_x"], turb["frac_y"], turb["frac_z"]

    render_legend_and_axis_labels()

    st.sidebar.subheader("Analysis Controls")
    normalize_x = st.sidebar.checkbox(
        "Normalize X-axis (t/t₀)", value=True, key="real_iso_norm_x",
        help="Use normalized time (t/t₀) instead of raw iteration numbers",
    )
    x_norm = st.sidebar.number_input(
        "X normalization constant",
        value=float(t0_raw),
        min_value=1.0,
        step=1000.0,
        disabled=not normalize_x,
        key="real_iso_x_norm",
        help="Normalization constant for X-axis (default: first iteration value)",
    )

    if normalize_x:
        time_norm = turb["iter"] / x_norm
    else:
        time_norm = turb["iter"]

    stationary_iter = st.sidebar.number_input("Stationarity iteration", value=50000.0, step=5000.0)
    stationary_t = stationary_iter / (x_norm if normalize_x else t0_raw)

    st.sidebar.markdown("**Tolerance bands**")
    tol_list_a = st.sidebar.multiselect(
        "Subplot A (Energy fractions)", [0.005, 0.01, 0.02],
        default=[0.005, 0.01, 0.02], key="tol_a",
    )
    tol_list_c = st.sidebar.multiselect(
        "Subplot C (Diagonal b_ii)", [0.005, 0.01, 0.02],
        default=[0.005, 0.01, 0.02], key="tol_c",
    )
    tol_list_d = st.sidebar.multiselect(
        "Subplot D (Cross-correlations)", [0.001, 0.005, 0.01],
        default=[0.001, 0.01], key="tol_d",
    )
    tol_list_e = st.sidebar.multiselect(
        "Subplot E (Deviations)", [0.005, 0.01, 0.02],
        default=[0.01, 0.02], key="tol_e",
    )

    min_len = len(turb["frac_x"])
    default_ma_win = max(10, min_len // 10) if min_len > 20 else 0
    ma_win = st.sidebar.slider("Moving average window (0=off)", 0, 500, default_ma_win, 5)

    plot_style_sidebar(data_dir, CURVES, PLOT_NAMES)

    st.markdown("### Real-space isotropy diagnostics")
    tab1, tab2, tab3 = st.tabs(["Energy & Lumley", "Anisotropy Tensor", "Deviations & Convergence"])

    with tab1:
        render_tab1(
            data_dir, time_norm, E_x, E_y, E_z, inv,
            ma_win=ma_win, tol_list_a=tol_list_a, stationary_t=stationary_t,
        )

    with tab2:
        render_tab2(
            data_dir, time_norm, b, inv,
            tol_list_c=tol_list_c, tol_list_d=tol_list_d,
        )

    with tab3:
        render_tab3(
            data_dir, time_norm, E_x, E_y, E_z,
            tol_list_e=tol_list_e, stationary_t=stationary_t,
        )

    st.markdown("### Final isotropy summary")
    render_summary(inv, E_x, E_y, E_z)

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(get_real_isotropy_theory_markdown())


if __name__ == "__main__":
    main()
