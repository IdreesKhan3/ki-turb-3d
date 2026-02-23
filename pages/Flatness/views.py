"""
Flatness — Main plot renderer.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List

from utils.report_builder import capture_button
from utils.export_figs import export_panel

from visualizations.flatness_vis import create_flatness_figure

from .plot_style import get_plot_style, apply_plot_style
from .data_helpers import compute_time_avg_flatness, format_legend_name


def render_main_plot(
    data_dir: Path,
    sim_groups: Dict[str, List[str]],
    start_idx: int,
    end_idx: int,
    num_errorbars: int,
    show_std: bool,
    show_error_bars: bool,
    show_reference: bool,
) -> bool:
    """
    Render time-averaged flatness plot.
    Returns True if any data was plotted, False otherwise.
    """
    plot_name = "Flatness Factors"
    ps = get_plot_style(plot_name)

    datasets = []
    for sim_prefix, files in sorted(sim_groups.items()):
        selected_files = tuple(files[start_idx - 1 : end_idx])
        if not selected_files:
            continue

        r_plot, F_mean, F_std = compute_time_avg_flatness(selected_files, num_errorbars)
        if r_plot is None:
            continue

        datasets.append({
            "sim_prefix": sim_prefix,
            "r": r_plot,
            "F_mean": F_mean,
            "F_std": F_std,
        })

    if not datasets:
        st.info("No valid flatness data could be plotted from selected range.")
        return False

    legend_names = {
        sp: st.session_state.flatness_legend_names.get(sp, format_legend_name(sp))
        for sp, _ in sorted(sim_groups.items())
    }
    axis_labels = st.session_state.axis_labels_flatness

    fig = create_flatness_figure(
        datasets,
        ps,
        show_std=show_std,
        show_error_bars=show_error_bars,
        show_reference=show_reference,
        axis_labels=axis_labels,
        legend_names=legend_names,
        apply_style=False,
    )
    fig = apply_plot_style(fig, ps)

    st.plotly_chart(fig, width="stretch")
    capture_button(fig, title="Flatness Factors", source_page="Flatness")
    st.subheader("Export Figure")
    export_panel(fig, data_dir, base_name="flatness_factors")
    return True


def render_theory_section():
    """Render theory & equations expander."""
    from content.flatness_theory_content import get_flatness_theory_markdown

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(get_flatness_theory_markdown())
