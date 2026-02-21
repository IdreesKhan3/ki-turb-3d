"""
Energy Spectra Page (Streamlit) — High Standard + Full Styling

Refactored: logic in pages/EnergySpectra/ (plot_style, file_loading, time_averaged, time_evolution).
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from content.energy_spectra_theory_content import get_energy_spectra_theory_markdown

from pages.EnergySpectra import (
    get_plot_style,
    apply_plot_style,
    plot_style_sidebar,
)
from pages.EnergySpectra.file_loading import (
    init_session_state,
    load_files_and_groups,
    render_legend_and_axis_labels,
)
from pages.EnergySpectra.time_averaged import render_time_averaged
from pages.EnergySpectra.time_evolution import render_time_evolution

st.set_page_config(page_icon="⚫")


def main():
    inject_theme_css()
    st.title("Energy Spectra")

    init_session_state()

    result = load_files_and_groups()
    if result is None:
        return

    data_dir, sim_groups, norm_groups = result

    st.sidebar.header("Options")
    view_mode = st.sidebar.radio(
        "View Mode", ["Time-Averaged", "Time Evolution"], index=0, key="energy_view_mode"
    )

    if sim_groups or norm_groups:
        render_legend_and_axis_labels(sim_groups, norm_groups)

    plot_names = ["Raw Energy Spectrum", "Normalized Spectrum", "Time Evolution"]
    plot_style_sidebar(data_dir, sim_groups, norm_groups, plot_names)

    if view_mode == "Time-Averaged":
        render_time_averaged(data_dir, sim_groups, norm_groups)
    else:
        render_time_evolution(data_dir, sim_groups)

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(get_energy_spectra_theory_markdown())


if __name__ == "__main__":
    main()
