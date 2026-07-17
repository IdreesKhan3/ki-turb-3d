"""
Shared spectral isotropy visualization — IC(k) and E11/E22/E33 component spectra.

Used by:
  1. Manual page (05_Spectral_Isotropy.py)
  2. AI agents (plot_spectral_isotropy, plot_component_spectra tools)

Pure Python plotting logic — no Streamlit dependency.
"""

import numpy as np
import plotly.graph_objects as go

from utils.plot_style import (
    default_plot_style,
    apply_plot_style as apply_plot_style_base,
    apply_axis_limits,
    apply_figure_size,
    resolve_line_style,
    _get_palette,
    _scalar_val,
    color_to_rgb,
)


def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()


# =============================================================================
# IC(k) Time-Averaged
# =============================================================================

def create_ic_isotropy_figure(
    sim_items,
    ps,
    *,
    show_std_band=True,
    show_error_bars=True,
    show_snapshot_lines=False,
    axis_labels=None,
    simulation_legend_names=None,
    ic_snap_label="IC(k) snapshots",
    apply_style=True,
):
    """
    Create IC(k) spectral isotropy figure with multi-simulation support.

    sim_items: List of (sim_prefix, data) where data has:
        - k, IC_mean, IC_std (arrays)
        - optional snapshot_curves: list of (k_array, ic_array) for per-snapshot lines
    ps: Plot style dict
    axis_labels: {"x": "k", "y": "IC(k)"}
    simulation_legend_names: {sim_prefix: display_name}
    """
    axis_labels = axis_labels or {"x": "k", "y": "IC(k)"}
    simulation_legend_names = simulation_legend_names or {}
    colors = _get_palette(ps)
    lw = ps.get("line_width", 2.2)

    fig = go.Figure()
    plotted_any = False

    for idx, (sim_prefix, data) in enumerate(sim_items):
        k = np.asarray(data.get("k", []))
        ic_mean = np.asarray(data.get("IC_mean", []))
        ic_std_raw = data.get("IC_std")
        # Keep finite positive IC; ignore under-resolved placeholder zeros already NaN'd upstream.
        valid = (k > 0) & np.isfinite(ic_mean) & (ic_mean > 0)
        if not np.any(valid):
            continue
        k, ic_mean = k[valid], ic_mean[valid]
        ic_std = None
        if ic_std_raw is not None:
            arr = np.asarray(ic_std_raw, dtype=float)
            if arr.shape == valid.shape:
                ic_std = arr[valid]

        legend_name = simulation_legend_names.get(sim_prefix, _default_labelify(sim_prefix))
        # Allow page to pass pre-computed style (e.g. from per-curve overrides)
        style_override = data.get("_style")
        if style_override:
            color = _scalar_val(style_override.get("color")) or colors[idx % len(colors)]
            lw_use = _scalar_val(style_override.get("width")) or lw
            dash = _scalar_val(style_override.get("dash"), "solid") or "solid"
            marker = style_override.get("marker") or "circle"
            msize = style_override.get("msize") or ps.get("marker_size", 6)
            override_on = style_override.get("override_on", False)
        else:
            _c, _w, _d, marker, msize, override_on = resolve_line_style(
                sim_prefix, idx, colors, ps,
                style_key="per_sim_style_ic",
                include_marker=True,
                default_marker="circle",
            )
            color, lw_use, dash = _c, _w, _scalar_val(_d, "solid") or "solid"

        # Per-snapshot lines (optional)
        if show_snapshot_lines:
            snapshot_curves = data.get("snapshot_curves", [])
            for i, (k0, ic0) in enumerate(snapshot_curves):
                k0 = np.asarray(k0, dtype=float)
                ic0 = np.asarray(ic0, dtype=float)
                if len(k0) == 0 or len(ic0) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=k0, y=ic0, mode="lines",
                    name=ic_snap_label,
                    line=dict(color=color, width=lw_use * 0.6, dash="dot"),
                    showlegend=(sim_prefix == sim_items[0][0] and i == 0),
                ))

        # Main time-averaged curve
        mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
        dash_scalar = _scalar_val(dash, "solid") or "solid"
        trace_kwargs = dict(x=k, y=ic_mean, mode=mode, name=legend_name, line=dict(color=color, width=lw_use, dash=dash_scalar))
        if override_on and marker and msize > 0:
            trace_kwargs["marker"] = dict(symbol=marker, size=msize)
        if show_error_bars and ic_std is not None and np.any(np.isfinite(ic_std)):
            trace_kwargs["error_y"] = dict(type="data", array=ic_std, visible=True, thickness=1, color=color)
        fig.add_trace(go.Scatter(**trace_kwargs))
        if show_std_band and ic_std is not None and np.any(np.isfinite(ic_std)):
            rgb = color_to_rgb(color)
            fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps.get('std_alpha', 0.18)})"
            fig.add_trace(go.Scatter(
                x=np.concatenate([k, k[::-1]]),
                y=np.concatenate([ic_mean - ic_std, (ic_mean + ic_std)[::-1]]),
                fill="toself", fillcolor=fill_rgba, line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
        plotted_any = True

    if not plotted_any:
        return None

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Isotropic (IC=1)")

    layout_kwargs = dict(
        xaxis_type=ps.get("x_axis_type", "log"),
        yaxis_type=ps.get("y_axis_type", "linear"),
        xaxis_title=axis_labels.get("x", "k"),
        yaxis_title=axis_labels.get("y", "IC(k)"),
        margin=dict(
            l=ps.get("margin_left", 60),
            r=ps.get("margin_right", 20),
            t=ps.get("margin_top", 40),
            b=ps.get("margin_bottom", 50),
        ),
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)
    if ps.get("show_plot_title") and ps.get("plot_title"):
        fig.update_layout(title=ps["plot_title"])
    if apply_style:
        fig = apply_plot_style_base(fig, ps)
    return fig


# =============================================================================
# Component Spectra E11, E22, E33
# =============================================================================

def create_component_spectra_figure(
    sim_items,
    ps,
    *,
    axis_labels=None,
    simulation_legend_names=None,
    curve_legend_names=None,
    show_curves=None,
    apply_style=True,
):
    """
    Create E11/E22/E33 component spectra figure with multi-simulation support.

    sim_items: List of (sim_prefix, data) where data has k, E11_mean, E22_mean, E33_mean
    ps: Plot style dict
    axis_labels: {"x": "k", "y": "E_ii(k)"}
    simulation_legend_names: {sim_prefix: display_name}
    curve_legend_names: {"E11": "E11(k)", "E22": "E22(k)", "E33": "E33(k)"}
    show_curves: List of curve names to show, e.g. ["E11", "E22", "E33"]. None or empty = all.
    """
    axis_labels = axis_labels or {"x": "k", "y": "E<sub>ii</sub>(k)"}
    simulation_legend_names = simulation_legend_names or {}
    curve_legend_names = curve_legend_names or {}
    default_curve_labels = {"E11": "E<sub>11</sub>(k)", "E22": "E<sub>22</sub>(k)", "E33": "E<sub>33</sub>(k)"}
    for k, v in default_curve_labels.items():
        curve_legend_names.setdefault(k, v)

    curves_to_plot = ["E11", "E22", "E33"]
    if show_curves:
        curves_to_plot = [c for c in ["E11", "E22", "E33"] if c in show_curves]

    colors = _get_palette(ps)
    lw = ps.get("line_width", 2.2)

    fig = go.Figure()
    plotted_any = False

    for idx, (sim_prefix, data) in enumerate(sim_items):
        if data.get("E11_mean") is None:
            continue
        k = np.asarray(data.get("k", []))
        valid = k > 0
        if not np.any(valid):
            continue
        k = k[valid]
        legend_name = simulation_legend_names.get(sim_prefix, _default_labelify(sim_prefix))

        # Allow page to pass pre-computed style per curve
        style_override = data.get("_style")
        if style_override:
            color = _scalar_val(style_override.get("color")) or colors[idx % len(colors)]
            lw_use = _scalar_val(style_override.get("width")) or lw
            dash = _scalar_val(style_override.get("dash"), "solid") or "solid"
        else:
            _c, _w, _d, _, _, _ = resolve_line_style(
                sim_prefix, idx, colors, ps,
                style_key="per_sim_style_eii",
                include_marker=True,
                default_marker="circle",
            )
            color, lw_use, dash = _c, _w, _scalar_val(_d, "solid") or "solid"

        for i, curve in enumerate(curves_to_plot):
            arr = data.get(f"{curve}_mean")
            if arr is None:
                continue
            arr = np.asarray(arr)
            if len(arr) != len(valid):
                continue
            arr = arr[valid]
            curve_label = curve_legend_names.get(curve, default_curve_labels[curve])
            trace_name = f"{legend_name} - {curve_label}"
            # Per-curve style override (from page's _resolve_curve_style)
            curve_styles = data.get("_curve_styles", {})
            cs = curve_styles.get(curve)
            if cs:
                c = _scalar_val(cs.get("color")) or color
                w = _scalar_val(cs.get("width")) or lw_use
                d = _scalar_val(cs.get("dash"), "solid") or dash
            else:
                c, w, d = color, lw_use, dash
            d_scalar = _scalar_val(d, "solid") or "solid"
            fig.add_trace(go.Scatter(
                x=k, y=arr, mode="lines",
                name=trace_name,
                line=dict(color=c, width=w, dash=d_scalar),
            ))
            plotted_any = True

    if not plotted_any:
        return None

    layout_kwargs = dict(
        xaxis_type=ps.get("x_axis_type", "log"),
        yaxis_type=ps.get("y_axis_type", "log"),
        xaxis_title=axis_labels.get("x", "k"),
        yaxis_title=axis_labels.get("y", "E<sub>ii</sub>(k)"),
        margin=dict(
            l=ps.get("margin_left", 60),
            r=ps.get("margin_right", 20),
            t=ps.get("margin_top", 40),
            b=ps.get("margin_bottom", 50),
        ),
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)
    if ps.get("show_plot_title") and ps.get("plot_title"):
        fig.update_layout(title=ps["plot_title"])
    if apply_style:
        fig = apply_plot_style_base(fig, ps)
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

for _kiturb_name in ('create_ic_isotropy_figure', 'create_component_spectra_figure',):
    if _kiturb_name in globals():
        globals()[_kiturb_name] = _kiturb_hit_provenance_wrapper(globals()[_kiturb_name])
