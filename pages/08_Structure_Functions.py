"""
Structure Functions Page (Streamlit)
ESS analysis + scaling exponents + full persistent UI controls

Features:
- Reads structure functions from binary (.bin) or text (.txt)
- Groups by simulation prefix
- Time-averages over selected time window
- Plots: S_p(r) vs r, ESS, anomalies (ξₚ − p/3)
- Computes ESS scaling exponents with user-fit range
- Full user controls, research-grade export

Refactored: logic in pages/StructureFunctions/ (plot_style, file_loading, data_helpers, views, ess_inset).

Requires kaleido for static export:
    pip install -U kaleido
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from pages.StructureFunctions import (
    init_session_state,
    load_structure_groups,
    render_legend_and_axis_labels,
    plot_style_sidebar,
    render_sp_tab,
    render_ess_tab,
    render_table_tab,
    render_theory_section,
)
from pages.StructureFunctions.data_helpers import compute_time_avg_structure

st.set_page_config(page_icon="⚫")


def main():
    inject_theme_css()
    st.title("Structure Functions")

    init_session_state()
    result = load_structure_groups()
    if result is None:
        return

    data_dir, sim_groups = result

    st.sidebar.subheader("Time Window")
    file_lengths = {k: len(v["files"]) for k, v in sim_groups.items()}
    min_len = min(file_lengths.values())
    if len(sim_groups) > 1:
        with st.sidebar.expander("File counts", expanded=False):
            for sim_prefix in sorted(file_lengths.keys()):
                st.text(f"{sim_prefix}: {file_lengths[sim_prefix]} files")

    start_idx = st.sidebar.slider("Start file index", 1, min_len, 1, key="struct_start_idx")
    end_idx = st.sidebar.slider("End file index", start_idx, min_len, min_len, key="struct_end_idx")

    st.sidebar.subheader("Orders / Normalization")
    sample_key = sorted(sim_groups.keys())[0]
    sample_files = tuple(sim_groups[sample_key]["files"][start_idx - 1 : end_idx])
    r_s, Sp_m_s, Sp_sd_s, urms_s, ps_list = compute_time_avg_structure(
        sample_files, sim_groups[sample_key]["kind"]
    )
    if ps_list is None:
        st.error("Could not read structure function data from the selected range.")
        return

    max_p = max(ps_list)
    selected_ps = st.sidebar.multiselect(
        "Orders p to plot (S_p and ESS)",
        options=list(range(1, max_p + 1)),
        default=list(range(1, min(7, max_p + 1))),
        key="struct_selected_ps",
    )
    ref_p = st.sidebar.selectbox(
        "ESS reference order (x-axis)",
        options=ps_list,
        index=ps_list.index(3) if 3 in ps_list else 0,
        key="struct_ref_p",
    )
    normalize_by_urms = st.sidebar.checkbox("Normalize S_p by u_rms^p", value=True, key="struct_norm_urms")

    st.sidebar.subheader("Error band / Theory")
    error_display = st.sidebar.radio(
        "Error display",
        ["Shaded band", "Error bars", "Both", "None"],
        index=0,
        help="Choose how to display ±1σ uncertainty (applies to both S_p and ESS plots)",
        key="struct_error_display",
    )
    show_std_band = error_display in ["Shaded band", "Both"]
    show_error_bars = error_display in ["Error bars", "Both"]
    show_sl_theory = st.sidebar.checkbox("Show She-Leveque anomalies", value=True, key="struct_show_sl")
    show_exp_anom = st.sidebar.checkbox("Show experimental anomalies (B93)", value=True, key="struct_show_exp")
    show_inset = st.sidebar.checkbox("Show ESS inset (anomalies)", value=True, key="struct_show_inset")

    st.sidebar.subheader("Fit range for ESS exponents")
    if r_s is not None and np.any(r_s > 0):
        r_pos = r_s[r_s > 0]
        r_min_default = float(np.percentile(r_pos, 10))
        r_max_default = float(np.percentile(r_pos, 60))
    else:
        r_min_default, r_max_default = 1e-3, 1e-1
    fit_rmin = st.sidebar.number_input("Fit r_min", value=r_min_default, min_value=0.0, format="%.6g", key="struct_fit_rmin")
    fit_rmax = st.sidebar.number_input("Fit r_max", value=r_max_default, min_value=fit_rmin + 1e-12, format="%.6g", key="struct_fit_rmax")

    render_legend_and_axis_labels(sim_groups)

    plot_names = ["S_p(r) vs r", "ESS (S_p vs S_3)", "ESS Inset", "Anomalies (ξₚ − p/3)"]
    plot_style_sidebar(data_dir, sim_groups, plot_names)

    tabs = st.tabs(["Sₚ(r) vs r", "ESS (Sₚ vs S₃)", "Scaling Exponents Table"])

    with tabs[0]:
        st.subheader("Time-averaged Structure Functions")
        render_sp_tab(
            data_dir,
            sim_groups,
            start_idx=start_idx,
            end_idx=end_idx,
            selected_ps=selected_ps,
            normalize_by_urms=normalize_by_urms,
            show_std_band=show_std_band,
            show_error_bars=show_error_bars,
        )

    with tabs[1]:
        st.subheader("Extended Self-Similarity (ESS)")
        render_ess_tab(
            data_dir,
            sim_groups,
            start_idx=start_idx,
            end_idx=end_idx,
            selected_ps=selected_ps,
            ref_p=ref_p,
            normalize_by_urms=normalize_by_urms,
            show_std_band=show_std_band,
            show_error_bars=show_error_bars,
            show_sl_theory=show_sl_theory,
            show_exp_anom=show_exp_anom,
            show_inset=show_inset,
            fit_rmin=fit_rmin,
            fit_rmax=fit_rmax,
        )

    with tabs[2]:
        render_table_tab()

    render_theory_section()


if __name__ == "__main__":
    main()
