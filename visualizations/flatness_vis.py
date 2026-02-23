"""
Shared flatness visualization — single source of truth for flatness factor F(r) plots.

Used by:
  1. Manual page (07_Flatness.py)
  2. AI agents (plot_flatness tool)

Pure Python plotting logic — no Streamlit dependency.
Supports: multi-sim, per-sim line styles, error bands/bars, Gaussian reference (F=3).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from utils.plot_style import (
    apply_axis_limits,
    apply_figure_size,
    apply_plot_style as apply_plot_style_base,
    resolve_line_style,
    _get_palette,
)


def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()


def _to_rgb(color) -> tuple:
    """Convert color to RGB tuple for rgba fill."""
    if isinstance(color, str) and color.startswith("#"):
        try:
            return hex_to_rgb(color)
        except (ValueError, TypeError):
            return (0, 0, 0)
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    return (0, 0, 0)


def create_flatness_figure(
    datasets,
    ps,
    *,
    show_std=True,
    show_error_bars=True,
    show_reference=True,
    axis_labels=None,
    legend_names=None,
    apply_style=True,
):
    """
    Create flatness factor F(r) vs r figure with multi-sim support.

    datasets: List of dicts with keys: sim_prefix, r (or x), F_mean (or y), F_std (or y_std)
    ps: Plot style dict (from get_plot_style or session)
    axis_labels: {"x": "...", "y": "..."}
    legend_names: {sim_prefix: display_name}
    """
    axis_labels = axis_labels or {"x": "r", "y": "Longitudinal flatness F<sub>L</sub>(r)"}
    legend_names = legend_names or {}
    colors = _get_palette(ps)

    fig = go.Figure()
    for idx, d in enumerate(datasets):
        sim_prefix = d.get("sim_prefix", f"sim_{idx}")
        r_raw = d.get("r") if d.get("r") is not None else d.get("x")
        r = np.asarray(r_raw if r_raw is not None else [], dtype=float)
        F_raw = d.get("F_mean") if d.get("F_mean") is not None else d.get("y")
        F_mean = np.asarray(F_raw if F_raw is not None else [], dtype=float)
        F_std_raw = d.get("F_std") if d.get("F_std") is not None else d.get("y_std")
        F_std = None
        if F_std_raw is not None:
            F_std = np.asarray(F_std_raw, dtype=float)

        if r.size == 0 or F_mean.size == 0:
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
        label = legend_names.get(sim_prefix, _default_labelify(sim_prefix))

        mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
        trace_kwargs = dict(
            x=r,
            y=F_mean,
            mode=mode,
            name=label,
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
            rgb = _to_rgb(color)
            fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps.get('std_alpha', 0.18)})"
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([r, r[::-1]]),
                    y=np.concatenate([F_mean - F_std, (F_mean + F_std)[::-1]]),
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if show_reference and len(fig.data) > 0:
        fig.add_hline(
            y=3,
            line_dash=ps.get("reference_dash", "dot"),
            line_color=ps.get("reference_color", "#000000"),
            line_width=ps.get("reference_width", 1.5),
            annotation_text="Gaussian (F=3)",
            annotation_position="right",
        )

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "r"),
        yaxis_title=axis_labels.get("y", "Longitudinal flatness F<sub>L</sub>(r)"),
        xaxis_type=ps.get("x_axis_type", "log"),
        yaxis_type=ps.get("y_axis_type", "linear"),
        legend_title="Simulation",
        height=500,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)

    if apply_style:
        fig = apply_plot_style_base(fig, ps)
    return fig
