"""
Flatness — Main plot renderer.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List

from utils.report_builder import capture_button
from utils.export_figs import export_panel
from utils.plot_style import apply_axis_limits, apply_figure_size, _get_palette, resolve_line_style

from .plot_style import get_plot_style, apply_plot_style
from .data_helpers import compute_time_avg_flatness, format_legend_name, color_to_rgb_tuple


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
    colors = _get_palette(ps)

    fig = go.Figure()
    plotted_any = False

    for idx, (sim_prefix, files) in enumerate(sorted(sim_groups.items())):
        selected_files = tuple(files[start_idx - 1 : end_idx])
        if not selected_files:
            continue

        r_plot, F_mean, F_std = compute_time_avg_flatness(selected_files, num_errorbars)
        if r_plot is None:
            continue

        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix,
            idx,
            colors,
            ps,
            style_key="per_sim_style_flatness",
            include_marker=True,
            default_marker="square",
        )

        legend_name = st.session_state.flatness_legend_names.get(
            sim_prefix, format_legend_name(sim_prefix)
        )
        plotted_any = True

        mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
        trace_kwargs = dict(
            x=r_plot,
            y=F_mean,
            mode=mode,
            name=legend_name,
            line=dict(color=color, width=lw, dash=dash),
            hovertemplate="r=%{x:.3g}<br>F(r)=%{y:.3g}<extra></extra>",
        )
        if override_on and marker and msize > 0:
            trace_kwargs["marker"] = dict(size=msize, symbol=marker, line=dict(width=1, color=color))
        if show_error_bars and F_std is not None:
            trace_kwargs["error_y"] = dict(
                type="data",
                array=F_std,
                visible=True,
                thickness=1,
                color=color,
            )
        fig.add_trace(go.Scatter(**trace_kwargs))

        if show_std and F_std is not None:
            rgb = color_to_rgb_tuple(color)
            fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps['std_alpha']})"
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([r_plot, r_plot[::-1]]),
                    y=np.concatenate([F_mean - F_std, (F_mean + F_std)[::-1]]),
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if plotted_any and show_reference:
        fig.add_hline(
            y=3,
            line_dash=ps.get("reference_dash", "dot"),
            line_color=ps.get("reference_color", "#000000"),
            line_width=ps.get("reference_width", 1.5),
            annotation_text="Gaussian (F=3)",
            annotation_position="right",
        )

    if plotted_any:
        layout_kwargs = dict(
            xaxis_title=st.session_state.axis_labels_flatness["x"],
            yaxis_title=st.session_state.axis_labels_flatness["y"],
            legend_title="Simulation",
            height=500,
        )
        layout_kwargs = apply_axis_limits(layout_kwargs, ps)
        layout_kwargs = apply_figure_size(layout_kwargs, ps)
        fig.update_layout(**layout_kwargs)
        fig = apply_plot_style(fig, ps)

        st.plotly_chart(fig, width="stretch")
        capture_button(fig, title="Flatness Factors", source_page="Flatness")
        st.subheader("Export Figure")
        export_panel(fig, data_dir, base_name="flatness_factors")
    else:
        st.info("No valid flatness data could be plotted from selected range.")

    return plotted_any


def render_theory_section():
    """Render theory & equations expander."""
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("**Longitudinal flatness factor:**")
        st.latex(
            r"""
        F_L(r) = \frac{\langle [\delta u_L(r)]^4 \rangle}{\langle [\delta u_L(r)]^2 \rangle^2}
        """
        )
        st.markdown(
            r"""
        where $\delta u_L(r) = u_L(\mathbf{x} + r\mathbf{e}_L) - u_L(\mathbf{x})$ is the longitudinal velocity increment.
        """
        )
        st.markdown("**Interpretation:**")
        st.markdown(
            r"""
        - $F_L(r) = 3$: Gaussian increments (no intermittency)
        - $F_L(r) > 3$: Intermittent, fat-tailed PDFs
        - $F_L(r) < 3$: Sub-Gaussian
        """
        )
        st.divider()
        st.markdown("**Reference:** [Pope (2001)](/Citation#pope2001) — Turbulent flows")
