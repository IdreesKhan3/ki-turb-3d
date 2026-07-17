"""Flatness factors page (Streamlit). Logic in pages/Flatness/.

Time-averaged F(r) with optional Gaussian reference (F=3) and time-window
selection. Static export needs kaleido.
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from pages.Flatness import (
    init_session_state,
    load_flatness_groups,
    render_legend_and_axis_labels,
    plot_style_sidebar,
    render_main_plot,
    render_theory_section,
)

st.set_page_config(page_icon="⚫")


def main():
    inject_theme_css()
    st.title("Flatness Factors")

    init_session_state()
    result = load_flatness_groups()
    if result is None:
        return

    data_dir, sim_groups = result

    st.sidebar.subheader("Time Window")
    max_files = min(len(v) for v in sim_groups.values())
    start_idx = st.sidebar.slider("Start file index", 1, max_files, 1, key="flatness_start_idx")
    end_idx = st.sidebar.slider("End file index", start_idx, max_files, max_files, key="flatness_end_idx")

    st.sidebar.subheader("Averaging / Error bars")
    num_errorbars = st.sidebar.slider("Number of error bar points", 10, 80, 20, key="flatness_num_errorbars")
    error_display = st.sidebar.radio(
        "Error display",
        ["Shaded band", "Error bars", "Both", "None"],
        index=0,
        help="Choose how to display ±1σ uncertainty",
        key="flatness_error_display",
    )
    show_std = error_display in ["Shaded band", "Both"]
    show_error_bars = error_display in ["Error bars", "Both"]

    st.sidebar.subheader("Plot Options")
    show_reference = st.sidebar.checkbox("Show Gaussian reference (F=3)", value=True, key="flatness_show_ref")

    render_legend_and_axis_labels(sim_groups)

    plot_names = ["Flatness Factors"]
    plot_style_sidebar(data_dir, sim_groups, plot_names)

    st.header("Time-Averaged Flatness Factors")
    render_main_plot(
        data_dir,
        sim_groups,
        start_idx=start_idx,
        end_idx=end_idx,
        num_errorbars=num_errorbars,
        show_std=show_std,
        show_error_bars=show_error_bars,
        show_reference=show_reference,
    )

    render_theory_section()


if __name__ == "__main__":
    main()
