"""
Shared PDF visualization for probability density function plots.

Used by:
  1. Manual page (09_PDFs.py via pages/PDFs/*)
  2. AI agents (plot_pdf tool)

Pure Python plotting logic — no Streamlit dependency.
Supports: velocity components, 1D PDFs (velocity magnitude, vorticity, enstrophy, dissipation),
2D joint PDFs (velocity-dissipation, velocity-enstrophy, dissipation-enstrophy, R-Q).
"""

import plotly.graph_objects as go

from utils.plot_style import (
    apply_plot_style,
    resolve_line_style,
    _get_palette,
)


def _to_list(arr):
    """Convert array-like to list for JSON serialization."""
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _default_label(s: str) -> str:
    return s.replace("_", " ").title()


# =============================================================================
# Velocity components PDF (u, v, w)
# =============================================================================

def create_velocity_components_pdf_figure(
    u_grid,
    pdf_u,
    pdf_v,
    pdf_w,
    style_config,
    *,
    x_label="u",
    y_label="P(u)",
    title="Velocity PDF",
    legend_title=None,
    label_base="",
    legend_names=None,
    apply_style=True,
):
    """
    Create velocity components PDF figure (u, v, w on same axes).

    label_base: Base name for traces (e.g. filename stem).
    legend_names: Override {comp: display_name} or {f"{label_base} - {comp}": display_name}.
    """
    legend_names = legend_names or {}
    component_colors = ["#1f77b4", "#2ca02c", "#d62728"]
    line_width = style_config.get("line_width", 2.4)

    fig = go.Figure()
    for idx, (y_vals, comp) in enumerate([(pdf_u, "u"), (pdf_v, "v"), (pdf_w, "w")]):
        key = f"{label_base} - {comp}" if label_base else comp
        trace_name = legend_names.get(comp) or legend_names.get(key) or key
        fig.add_trace(go.Scatter(
            x=_to_list(u_grid),
            y=_to_list(y_vals),
            mode="lines",
            name=trace_name,
            line=dict(color=component_colors[idx], width=line_width),
            hovertemplate="%{x:.4f}<br>PDF = %{y:.4e}<extra></extra>",
        ))

    layout_kwargs = dict(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=style_config.get("figure_height", 500),
        hovermode="x unified",
        legend=dict(x=1.02, y=1),
        legend_title_text=legend_title if legend_title else None,
    )
    fig.update_layout(**layout_kwargs)

    if apply_style:
        try:
            fig = apply_plot_style(fig, style_config)
        except Exception:
            pass
    return fig


# =============================================================================
# 1D PDFs (velocity magnitude, vorticity, enstrophy, dissipation) — multi-sim
# =============================================================================

def create_1d_pdf_figure(
    trace_data,
    style_config,
    *,
    x_label="|u|",
    y_label="P(|u|)",
    title="Velocity Magnitude PDF",
    legend_title=None,
    simulation_legend_names=None,
    sim_legends=None,
    apply_style=True,
):
    """
    Create 1D PDF figure with multi-simulation support.

    trace_data: List of (sim_prefix, x_vals, pdf_vals)
    simulation_legend_names: Agent override {sim_prefix: display_name}
    sim_legends: Session-persisted {sim_prefix: display_name}
    """
    from utils.plot_style import ensure_per_sim_defaults

    simulation_legend_names = simulation_legend_names or {}
    sim_legends = sim_legends or {}
    sim_groups = {sp: sp for sp, _xv, _pv in trace_data}
    ensure_per_sim_defaults(
        style_config, sim_groups,
        style_key="per_sim_style_comparison",
        include_marker=True,
    )
    colors = _get_palette(style_config)

    fig = go.Figure()
    for idx, (sim_prefix, xv, pv) in enumerate(trace_data):
        trace_label = (
            simulation_legend_names.get(sim_prefix)
            or sim_legends.get(sim_prefix)
            or _default_label(sim_prefix)
        )
        color, width, dash = resolve_line_style(
            sim_prefix, idx, colors, style_config,
            style_key="per_sim_style_comparison",
            include_marker=False,
        )
        fig.add_trace(go.Scatter(
            x=_to_list(xv),
            y=_to_list(pv),
            mode="lines",
            name=trace_label,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate="%{x:.4f}<br>PDF = %{y:.4e}<extra></extra>",
        ))

    layout_kwargs = dict(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=style_config.get("figure_height", 500),
        hovermode="x unified",
        legend=dict(x=1.02, y=1),
        legend_title_text=legend_title if legend_title else None,
    )
    if style_config.get("plot_bgcolor"):
        layout_kwargs["plot_bgcolor"] = style_config["plot_bgcolor"]
    if style_config.get("paper_bgcolor"):
        layout_kwargs["paper_bgcolor"] = style_config["paper_bgcolor"]

    fig.update_layout(**layout_kwargs)

    if apply_style:
        try:
            fig = apply_plot_style(fig, style_config)
        except Exception:
            pass
    return fig


# =============================================================================
# 2D joint PDFs (contour)
# =============================================================================

def create_2d_contour_pdf_figure(
    x_centers,
    y_centers,
    z_data,
    style_config,
    *,
    x_label="|u|",
    y_label="ε",
    z_label="PDF",
    title="Joint PDF",
    trace_name=None,
    legend_title=None,
    hovertemplate="|u| = %{x:.4f}<br>ε = %{y:.4e}<br>PDF = %{z:.2e}<extra></extra>",
    use_log_scale=False,
    apply_style=True,
):
    """
    Create 2D contour PDF figure (joint PDFs, R-Q, etc.).

    use_log_scale: If True, plot log10(PDF) with Jet colorscale (matches manual page).
    """
    import numpy as np
    if use_log_scale:
        z_arr = np.asarray(z_data, dtype=np.float64)
        safe = z_arr.copy()
        safe[safe <= 0] = np.nan
        plot_z = np.log10(safe)
        z_label = "log₁₀(PDF)"
        colorscale = "Jet"
    else:
        plot_z = z_data
        colorscale = "Viridis"
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=_to_list(x_centers),
        y=_to_list(y_centers),
        z=_to_list(plot_z),
        name=trace_name or "",
        colorscale=colorscale,
        colorbar=dict(title=z_label),
        hovertemplate=hovertemplate,
    ))

    layout_kwargs = dict(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=style_config.get("figure_height", 500),
        hovermode="closest",
        legend=dict(x=1.02, y=1),
        legend_title_text=legend_title if legend_title else None,
    )
    fig.update_layout(**layout_kwargs)

    if apply_style:
        try:
            fig = apply_plot_style(fig, style_config)
        except Exception:
            pass
    return fig


# KI_TURB_HIT_PROVENANCE_WRAPPERS_V2
from functools import wraps as _kiturb_wraps
from visualizations.provenance import stamp_plotly_figure as _kiturb_stamp_plotly_figure

def _kiturb_hit_provenance_wrapper(_fn):
    @_kiturb_wraps(_fn)
    def _wrapped(*args, **kwargs):
        _provenance = kwargs.pop("hit_provenance", None)
        _figure = _fn(*args, **kwargs)
        if _figure is None:
            return None
        return _kiturb_stamp_plotly_figure(_figure, _provenance)
    return _wrapped

for _kiturb_name in ('create_velocity_components_pdf_figure', 'create_1d_pdf_figure', 'create_2d_contour_pdf_figure',):
    if _kiturb_name in globals():
        globals()[_kiturb_name] = _kiturb_hit_provenance_wrapper(globals()[_kiturb_name])
