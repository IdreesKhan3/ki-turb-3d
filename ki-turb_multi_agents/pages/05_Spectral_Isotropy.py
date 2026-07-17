"""
Isotropy Validation (Spectral) Page — Streamlit

Refactored: logic in pages/SpectralIsotropy/ (plot_style, file_loading, views).
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from content.spectral_isotropy_theory_content import get_spectral_isotropy_theory_markdown

from pages.SpectralIsotropy import (
    init_session_state,
    load_ic_groups,
    render_legend_and_axis_labels,
    plot_style_sidebar,
    render_ic_tab,
    render_component_spectra_tab,
    render_summary_tab,
)

st.set_page_config(page_icon="⚫")


def main():
    inject_theme_css()
    st.title("Isotropy Validation — Spectral")

    init_session_state()

    result = load_ic_groups()
    if result is None:
        return

    data_dir, ic_groups = result

    render_legend_and_axis_labels(ic_groups)

    min_len = min(len(files) for files in ic_groups.values()) if ic_groups else 1
    start_idx = st.sidebar.slider("Start file index", 1, min_len, 1, key="speciso_start_idx")
    end_idx = st.sidebar.slider("End file index", start_idx, min_len, min_len, key="speciso_end_idx")

    st.sidebar.subheader("Options")
    show_snapshot_lines = st.sidebar.checkbox(
        "Show per-snapshot IC(k)", value=False, key="speciso_show_snap"
    )
    error_display = st.sidebar.radio(
        "Error display",
        ["Shaded band", "Error bars", "Both", "None"],
        index=3,
        help="Choose how to display ±1σ uncertainty",
        key="speciso_error_display",
    )
    show_std_band = error_display in ["Shaded band", "Both"]
    show_error_bars = error_display in ["Error bars", "Both"]
    show_component_spectra = st.sidebar.checkbox(
        "Show E11/E22/E33 plot", value=True, key="speciso_show_component"
    )
    show_curves = st.sidebar.multiselect(
        "Component Spectra curves",
        options=["E11", "E22", "E33"],
        default=["E11", "E22", "E33"],
        key="speciso_show_curves",
    )
    if not show_curves:
        show_curves = ["E11", "E22", "E33"]

    curves = ["IC", "IC_snap", "E11", "E22", "E33"]
    plot_names = ["IC(k) Time-Avg", "Component Spectra"]
    plot_style_sidebar(data_dir, curves, plot_names, sim_groups=ic_groups)

    tabs = st.tabs(["IC(k) Time-Avg", "Component Spectra", "Summary"])

    with tabs[0]:
        if not render_ic_tab(
            data_dir, ic_groups, start_idx, end_idx,
            show_snapshot_lines, show_std_band, show_error_bars,
        ):
            return

    with tabs[1]:
        render_component_spectra_tab(
            data_dir, ic_groups, start_idx, end_idx, show_component_spectra, show_curves
        )

    with tabs[2]:
        render_summary_tab(ic_groups, start_idx, end_idx)

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(get_spectral_isotropy_theory_markdown())


if __name__ == "__main__":
    main()
