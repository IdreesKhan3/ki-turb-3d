"""
Shared spectrum visualization — single source of truth for energy spectrum plots.

Used by:
  1. Manual page (06_Energy_Spectra.py)
  2. AI agents (plot_spectrum tool)

Pure Python plotting logic — no Streamlit dependency.
Supports: raw spectra, normalized spectra with Pope model, time evolution.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from utils.plot_style import (
    default_plot_style,
    apply_plot_style as apply_plot_style_base,
    apply_axis_limits,
    apply_figure_size,
    resolve_line_style,
    _get_palette,
)


def add_kolmogorov_line(fig, k_vals, E_avg, kmin, kmax, ps,
                        kolm_scale_factor=1.0,
                        label="Kolmogorov k<sup>-5/3</sup>"):
    """
    Add scaled -5/3 reference line on [kmin, kmax].
    """
    mask = (k_vals >= kmin) & (k_vals <= kmax)
    k_fit = k_vals[mask]
    if k_fit.size < 3:
        return fig

    ref = k_fit ** (-5.0 / 3.0)
    mid_idx = np.argmin(np.abs(k_fit - np.median(k_fit)))
    scale = E_avg[mask][mid_idx] / ref[mid_idx]
    ref = ref * scale * kolm_scale_factor

    fig.add_trace(go.Scatter(
        x=k_fit, y=ref,
        mode="lines",
        name=label,
        line=dict(
            color=ps.get("kolmogorov_color", "#666666"),
            width=ps.get("line_width", 2.4),
            dash="dot",
        ),
        hovertemplate="k=%{x:.3g}<br>ref=%{y:.3g}<extra></extra>",
    ))
    return fig


def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()


# =============================================================================
# Raw spectrum (multi-sim, Kolmogorov, error bars/bands)
# =============================================================================

def create_raw_spectrum_figure(
    datasets,
    ps,
    *,
    show_std=True,
    show_error_bars=True,
    show_kolmogorov=True,
    kmin=3.0,
    kmax=20.0,
    kolm_scale_factor=1.0,
    kolm_scale_data=None,
    axis_labels=None,
    legend_names=None,
    apply_style=True,
):
    """
    Create raw energy spectrum figure with multi-sim support.

    datasets: List of dicts with keys: sim_prefix, x (k_vals), y (E_avg), y_std (E_std)
    ps: Plot style dict (from get_plot_style or session)
    kolm_scale_data: Optional {"x": k_vals, "y": E_avg} for scaling Kolmogorov line
    axis_labels: {"x": "...", "y": "..."}
    legend_names: {sim_prefix: display_name}
    """
    axis_labels = axis_labels or {"x": "Wavenumber k", "y": "Energy spectrum E(k)"}
    legend_names = legend_names or {}
    colors = _get_palette(ps)

    fig = go.Figure()
    for idx, d in enumerate(datasets):
        sim_prefix = d.get("sim_prefix", f"sim_{idx}")
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        y_std = d.get("y_std")

        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix, idx, colors, ps,
            style_key="per_sim_style_raw",
            include_marker=True,
            default_marker="circle",
        )
        label = legend_names.get(sim_prefix, _default_labelify(sim_prefix))

        mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
        trace_kwargs = dict(
            x=x, y=y, mode=mode, name=label,
            line=dict(color=color, width=lw, dash=dash),
        )
        if override_on and marker and msize > 0:
            trace_kwargs["marker"] = dict(symbol=marker, size=msize)
        if show_error_bars and y_std is not None:
            trace_kwargs["error_y"] = dict(
                type="data", array=np.asarray(y_std, dtype=float),
                visible=True, thickness=1, color=color,
            )
        fig.add_trace(go.Scatter(**trace_kwargs))

        if show_std and y_std is not None:
            rgb = hex_to_rgb(color)
            fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps.get('std_alpha', 0.18)})"
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y - y_std, (y + y_std)[::-1]]),
                fill="toself", fillcolor=fill_rgba,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

    # Agent-controllable: show_kolmogorov from ps (LLM can set to False)
    show_k = ps.get("show_kolmogorov", show_kolmogorov)
    if show_k and kolm_scale_data:
        k_scale = kolm_scale_data.get("x")
        E_avg_scale = kolm_scale_data.get("y")
        if k_scale is not None and E_avg_scale is not None:
            add_kolmogorov_line(fig, np.asarray(k_scale), np.asarray(E_avg_scale),
                               kmin, kmax, ps, kolm_scale_factor=kolm_scale_factor)

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "Wavenumber k"),
        yaxis_title=axis_labels.get("y", "Energy spectrum E(k)"),
        xaxis_type=ps.get("x_axis_type", "log"),
        yaxis_type=ps.get("y_axis_type", "log"),
        legend_title="Simulation", height=400,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)

    if apply_style:
        fig = apply_plot_style_base(fig, ps)
    return fig


# =============================================================================
# Normalized spectrum (multi-sim, Pope model)
# =============================================================================

def create_normalized_spectrum_figure(
    datasets,
    ps,
    *,
    show_std=True,
    show_error_bars=True,
    pope_scaling_prefix=None,
    axis_labels=None,
    legend_names=None,
    apply_style=True,
):
    """
    Create normalized spectrum figure with Pope model.

    datasets: List of dicts with keys: sim_prefix, x (keta), y (En_avg), y_std, y_pope (Ep_avg)
    pope_scaling_prefix: If set, only show Pope for that sim; else show for all
    """
    axis_labels = axis_labels or {
        "x": "Normalized wavenumber kη",
        "y": "Normalized spectrum E<sub>norm</sub>(kη)",
    }
    legend_names = legend_names or {}
    colors = _get_palette(ps)

    fig = go.Figure()
    for idx, d in enumerate(datasets):
        sim_prefix = d.get("sim_prefix", f"sim_{idx}")
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        y_std = d.get("y_std")
        y_pope = d.get("y_pope")

        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix, idx, colors, ps,
            style_key="per_sim_style_norm",
            include_marker=True,
            default_marker="circle",
        )
        label = legend_names.get(sim_prefix, _default_labelify(sim_prefix))

        mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
        trace_kwargs = dict(
            x=x, y=y, mode=mode, name=label,
            line=dict(color=color, width=lw, dash=dash),
        )
        if override_on and marker and msize > 0:
            trace_kwargs["marker"] = dict(symbol=marker, size=msize)
        if show_error_bars and y_std is not None:
            trace_kwargs["error_y"] = dict(
                type="data", array=np.asarray(y_std, dtype=float),
                visible=True, thickness=1, color=color,
            )
        fig.add_trace(go.Scatter(**trace_kwargs))

        if show_std and y_std is not None:
            rgb = hex_to_rgb(color)
            fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps.get('std_alpha', 0.18)})"
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y - y_std, (y + y_std)[::-1]]),
                fill="toself", fillcolor=fill_rgba,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

        # Agent-controllable: show_pope from ps (LLM can set to False)
        show_pope = ps.get("show_pope", True)
        if show_pope and y_pope is not None and (pope_scaling_prefix is None or sim_prefix == pope_scaling_prefix):
            pope_label = f"{label} Pope Model" if pope_scaling_prefix else f"{label} Pope"
            fig.add_trace(go.Scatter(
                x=x, y=np.asarray(y_pope, dtype=float),
                mode="lines", name=pope_label,
                line=dict(color=ps.get("pope_color", "#000000"),
                         width=ps.get("line_width", 2.4), dash="dash"),
            ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "Normalized wavenumber kη"),
        yaxis_title=axis_labels.get("y", "Normalized spectrum E<sub>norm</sub>(kη)"),
        xaxis_type=ps.get("x_axis_type", "log"),
        yaxis_type=ps.get("y_axis_type", "log"),
        legend_title="Simulation", width=550, height=600,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)

    if apply_style:
        fig = apply_plot_style_base(fig, ps)
    return fig


# =============================================================================
# Time evolution (thin curves + highlighted)
# =============================================================================

def create_time_evolution_figure(
    thin_curves,
    highlight_curve,
    ps,
    *,
    axis_labels=None,
    apply_style=True,
):
    """
    Create time evolution figure.

    thin_curves: List of {"x": k, "y": E}
    highlight_curve: {"x": k, "y": E, "label": "Highlighted iter 123"}
    """
    axis_labels = axis_labels or {"x": "Wavenumber k", "y": "Energy spectrum E(k)"}

    fig = go.Figure()
    for d in thin_curves:
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(width=max(1.0, ps.get("line_width", 2.4) * 0.6)),
            opacity=0.25, showlegend=False,
        ))

    if highlight_curve:
        x = np.asarray(highlight_curve["x"], dtype=float)
        y = np.asarray(highlight_curve["y"], dtype=float)
        label = highlight_curve.get("label", "Highlighted")
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=label,
            line=dict(width=ps.get("line_width", 2.4) * 1.2,
                     color=ps.get("highlight_color", "#E41A1C")),
            opacity=1.0,
        ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "Wavenumber k"),
        yaxis_title=axis_labels.get("y", "Energy spectrum E(k)"),
        xaxis_type="log", yaxis_type="log",
        width=850, height=550,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)

    if apply_style:
        fig = apply_plot_style_base(fig, ps)
    return fig


# =============================================================================
# Simple single-curve (agent fallback)
# =============================================================================

def create_spectrum_figure(k_vals, E_avg, E_std=None, style_config=None,
                          show_kolmogorov=True, kmin=None, kmax=None,
                          kolm_scale_factor=1.0, show_std=True, show_error_bars=True,
                          x_label="Wavenumber k", y_label="Energy E(k)", title="Energy Spectrum"):
    """
    Generate a simple single-curve energy spectrum figure.
    Used by agents when full multi-sim config is not available.
    """
    k_vals = np.asarray(k_vals, dtype=float)
    E_avg = np.asarray(E_avg, dtype=float)

    ps = default_plot_style()
    ps.update({
        "line_width": 2.4,
        "kolmogorov_color": "#666666",
        "x_axis_type": "log",
        "y_axis_type": "log",
    })
    if style_config:
        ps.update(style_config)

    # Agent-controllable: show_kolmogorov from ps
    show_k = ps.get("show_kolmogorov", show_kolmogorov)
    # custom_colors: list ["#hex", ...] or dict {"Energy Spectrum": "#hex"} from agent
    cols = ps.get("custom_colors")
    if isinstance(cols, dict) and cols:
        color = cols.get("Energy Spectrum") or next(iter(cols.values()), None)
    elif isinstance(cols, (list, tuple)) and cols:
        color = cols[0]
    else:
        color = None
    if not color or not isinstance(color, str):
        color = "#1f77b4"

    fig = go.Figure()
    trace_kwargs = dict(
        x=k_vals, y=E_avg, mode="lines", name="Energy Spectrum",
        line=dict(color=color, width=ps["line_width"]),
    )
    if show_error_bars and E_std is not None:
        trace_kwargs["error_y"] = dict(
            type="data", array=np.asarray(E_std, dtype=float),
            visible=True, thickness=1, color=color,
        )
    fig.add_trace(go.Scatter(**trace_kwargs))

    if show_std and E_std is not None:
        rgb = hex_to_rgb(color) if isinstance(color, str) and color.startswith("#") else (31, 119, 180)
        fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps.get('std_alpha', 0.18)})"
        fig.add_trace(go.Scatter(
            x=np.concatenate([k_vals, k_vals[::-1]]),
            y=np.concatenate([E_avg - np.asarray(E_std, dtype=float), (E_avg + np.asarray(E_std, dtype=float))[::-1]]),
            fill="toself", fillcolor=fill_rgba,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    if show_k and len(k_vals) > 2:
        if kmin is not None and kmax is not None:
            add_kolmogorov_line(fig, k_vals, E_avg, kmin, kmax, ps,
                               kolm_scale_factor=kolm_scale_factor)
        else:
            ref_k = k_vals[k_vals > 0]
            if len(ref_k) > 1:
                mid_idx = len(k_vals) // 2
                c = float(E_avg[mid_idx]) * (float(k_vals[mid_idx]) ** (5 / 3))
                ref_E = c * (ref_k ** (-5 / 3))
                fig.add_trace(go.Scatter(
                    x=ref_k, y=ref_E, mode="lines", name="Kolmogorov -5/3",
                    line=dict(color=ps["kolmogorov_color"], width=ps["line_width"], dash="dot"),
                    hoverinfo="skip",
                ))

    layout_kwargs = dict(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        xaxis_type=ps.get("x_axis_type", "log"),
        yaxis_type=ps.get("y_axis_type", "log"),
    )
    fig.update_layout(**layout_kwargs)
    fig = apply_plot_style_base(fig, ps)
    return fig
