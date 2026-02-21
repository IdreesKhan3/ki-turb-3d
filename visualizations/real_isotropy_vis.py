"""
Shared real-space isotropy visualization — energy fractions and Lumley triangle.

Used by:
  1. Manual page (04_Real_Isotropy.py)
  2. AI agents (plot_real_isotropy, plot_lumley_triangle tools)

Pure Python plotting logic — no Streamlit dependency.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from utils.plot_style import (
    default_plot_style,
    apply_plot_style as apply_plot_style_base,
    apply_axis_limits,
    apply_figure_size,
    _get_palette,
    resolve_curve_style,
)

# Plot keys for per-curve style (match page's _normalize_plot_name)
_PLOT_KEY_ENERGY_FRACTIONS = "Energy_Fractions_A"
# Energy Fractions: style keys Ex/Ey/Ez (page sidebar) vs legend keys frac_x/frac_y/frac_z
_ENERGY_LEGEND_KEY = {"Ex": "frac_x", "Ey": "frac_y", "Ez": "frac_z"}
_PLOT_KEY_DIAGONAL_BII = "Diagonal_b_ii_C"
_PLOT_KEY_CROSS = "Cross_correlations_D"
_PLOT_KEY_DEVIATIONS = "Deviations_E"


# =============================================================================
# Energy Fractions (frac_x, frac_y, frac_z vs t/t0)
# =============================================================================

def create_energy_fractions_figure(
    iter_norm,
    frac_x,
    frac_y,
    frac_z,
    ps,
    *,
    axis_labels=None,
    legend_names=None,
    apply_style=True,
    ma_win=None,
    add_raw_suffix=False,
    tol_list=None,
    stationary_t=None,
    raw_data_opacity=None,
):
    """
    Create energy fractions figure: frac_x, frac_y, frac_z vs t/t0.

    iter_norm: Normalized iteration/time (x-axis)
    frac_x, frac_y, frac_z: Energy fractions (y-axis)
    ps: Plot style dict
    axis_labels: {"x": "t/t0", "y": "Energy fraction"}
    legend_names: {"frac_x": "E_x", "frac_y": "E_y", "frac_z": "E_z"}
    ma_win: Moving average window (int > 1 to enable). When set, adds MA traces.
    add_raw_suffix: If True, append " (raw)" to base curve legend names (for page parity).
    tol_list: List of tolerance values (e.g. [0.005, 0.01, 0.02]) for ±tol bands around 1/3.
    stationary_t: Normalized time (t/t0) for statistical stationarity vertical line.
    raw_data_opacity: Opacity for raw data traces (0–1). Overrides ps. Default from ps or 0.5.
    """
    axis_labels = axis_labels or {"x": "t/t0", "y": "Energy fraction"}
    legend_names = legend_names or {"frac_x": "frac_x", "frac_y": "frac_y", "frac_z": "frac_z"}

    iter_norm = np.asarray(iter_norm, dtype=float)
    frac_x = np.asarray(frac_x, dtype=float)
    frac_y = np.asarray(frac_y, dtype=float)
    frac_z = np.asarray(frac_z, dtype=float)

    raw_lw = ps.get("line_width", 2.2) * (0.8 if add_raw_suffix else 1.0)
    palette_colors = _get_palette(ps)
    # Use Ex, Ey, Ez to match page's per_curve_style keys (sidebar stores under these)
    curves = [(frac_x, "Ex"), (frac_y, "Ey"), (frac_z, "Ez")]
    markers = ["circle", "square", "triangle-up"] if add_raw_suffix else [None] * 3

    raw_opacity = raw_data_opacity if raw_data_opacity is not None else ps.get("raw_data_opacity", 0.5)
    fig = go.Figure()
    raw_suffix = " (raw)" if add_raw_suffix else ""
    for i, (arr, key) in enumerate(curves):
        c, lw, dash, mk, ms = resolve_curve_style(key, i, palette_colors, ps, _PLOT_KEY_ENERGY_FRACTIONS)
        line_color = c
        if add_raw_suffix:
            rgb = hex_to_rgb(c)
            line_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
        legend_label = legend_names.get(key) or legend_names.get(_ENERGY_LEGEND_KEY.get(key, key), key)
        trace_kw = dict(x=iter_norm, y=arr, mode="lines", name=legend_label + raw_suffix, line=dict(color=line_color, width=lw if not add_raw_suffix else raw_lw, dash=dash))
        if markers[i]:
            trace_kw["mode"] = "lines+markers"
            trace_kw["marker"] = dict(
                symbol=markers[i],
                size=max(2, ms * 0.4),
                color=c,
                opacity=raw_opacity,
                line=dict(width=0),
            )
        fig.add_trace(go.Scatter(**trace_kw))

    # Moving average traces (when ma_win > 1)
    if ma_win and ma_win > 1 and len(frac_x) > ma_win:
        def _ma(x):
            k = np.ones(ma_win) / ma_win
            return np.convolve(x, k, mode="valid")

        t_ma = iter_norm[ma_win // 2 : ma_win // 2 + len(_ma(frac_x))]
        for i, (arr, key) in enumerate(curves):
            c, lw, dash, _, _ = resolve_curve_style(key, i, palette_colors, ps, _PLOT_KEY_ENERGY_FRACTIONS)
            ma_lw = lw * 1.1
            legend_label = legend_names.get(key) or legend_names.get(_ENERGY_LEGEND_KEY.get(key, key), key)
            fig.add_trace(go.Scatter(
                x=t_ma, y=_ma(arr), mode="lines",
                name=legend_label + f" (MA-{ma_win})",
                line=dict(color=c, width=ma_lw, dash=dash),
            ))

    iso_color = ps.get("isotropic_1_3_color", "gray")
    if add_raw_suffix:
        fig.add_hline(y=1.0 / 3, line_dash="dash", line_color=iso_color, line_width=1.5,
                      opacity=0.8, annotation_text="", showlegend=False)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=iso_color, width=1.5, dash="dash"),
            name="Isotropic (1/3)",
            showlegend=True,
        ))
    else:
        fig.add_hline(y=1.0 / 3, line_dash="dash", line_color=iso_color, annotation_text="Isotropic (1/3)")

    # Tolerance bands (when tol_list provided)
    if tol_list:
        tol_colors = ["lightcoral", "lightpink", "mistyrose"]
        for i, tol in enumerate(tol_list):
            tol = float(tol)
            color = tol_colors[i % len(tol_colors)]
            fig.add_hrect(y0=1 / 3 - tol, y1=1 / 3 + tol, fillcolor=color, opacity=0.3,
                         line_width=0, layer="below")
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color, opacity=0.3),
                name=f"±{tol:.1%} tolerance",
                showlegend=True,
            ))

    # Statistical stationarity line (when stationary_t provided)
    if stationary_t is not None:
        stat_color = ps.get("stationary_line_color", "#800080")
        fig.add_vline(x=float(stationary_t), line_dash="dash", line_color=stat_color, line_width=1.5,
                     opacity=0.8, annotation_text="", showlegend=False)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=stat_color, width=1.5, dash="dash"),
            name="Statistical stationarity",
            showlegend=True,
        ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "t/t0"),
        yaxis_title=axis_labels.get("y", "Energy fraction"),
        margin=dict(l=60, r=40, t=50, b=50),
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
# Diagonal b_ii (b11, b22, b33 vs t/t0)
# =============================================================================

def create_diagonal_bii_figure(
    iter_norm,
    b11,
    b22,
    b33,
    ps,
    *,
    axis_labels=None,
    legend_names=None,
    apply_style=True,
    tol_list=None,
):
    """
    Create diagonal b_ii figure: b11, b22, b33 vs t/t0.

    iter_norm: Normalized iteration/time (x-axis)
    b11, b22, b33: Diagonal anisotropy tensor components (y-axis)
    ps: Plot style dict
    axis_labels: {"x": "t/t₀", "y": "Anisotropy tensor b<sub>ij</sub>"} (matches page subplot C)
    legend_names: {"b11": "...", "b22": "...", "b33": "..."}
    tol_list: List of tolerance values (e.g. [0.005, 0.01, 0.02]) for ±tol bands around 0.
    """
    # b_ii can be negative; log scale causes artifacts — always use linear
    ps = dict(ps)
    ps["x_axis_type"] = "linear"
    ps["y_axis_type"] = "linear"
    axis_labels = axis_labels or {"x": "t/t₀", "y": "Anisotropy tensor b<sub>ij</sub>"}
    legend_names = legend_names or {"b11": "b11", "b22": "b22", "b33": "b33"}

    iter_norm = np.asarray(iter_norm, dtype=float)
    b11 = np.asarray(b11, dtype=float)
    b22 = np.asarray(b22, dtype=float)
    b33 = np.asarray(b33, dtype=float)

    colors = _get_palette(ps)
    curves = [(b11, "b11"), (b22, "b22"), (b33, "b33")]

    fig = go.Figure()
    for i, (arr, key) in enumerate(curves):
        c, lw, dash, _, _ = resolve_curve_style(key, i, colors, ps, _PLOT_KEY_DIAGONAL_BII)
        fig.add_trace(go.Scatter(
            x=iter_norm, y=arr, mode="lines",
            name=legend_names.get(key, key),
            line=dict(color=c, width=lw, dash=dash),
        ))

    iso_color = ps.get("isotropic_0_color", "#000000")
    fig.add_hline(y=0, line_dash="dash", line_color=iso_color, line_width=1.5,
                  annotation_text="", showlegend=False)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=iso_color, width=1.5, dash="dash"),
        name="Isotropic value (0)",
        showlegend=True,
    ))

    if tol_list:
        tol_colors = ["lightcoral", "lightpink", "mistyrose"]
        for i, tol in enumerate(tol_list):
            tol = float(tol)
            color = tol_colors[i % len(tol_colors)]
            fig.add_hrect(y0=-tol, y1=tol, fillcolor=color, opacity=0.3,
                         line_width=0, layer="below")
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color, opacity=0.3),
                name=f"±{tol:.1%} tolerance",
                showlegend=True,
            ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "t/t₀"),
        yaxis_title=axis_labels.get("y", "Anisotropy tensor b<sub>ij</sub>"),
        margin=dict(l=60, r=40, t=50, b=50),
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
# Cross-correlations (|b12|, |b13|, |b23|, anisotropy index vs t/t0)
# =============================================================================

def create_cross_correlations_figure(
    iter_norm,
    b12,
    b13,
    b23,
    anis_index,
    ps,
    *,
    axis_labels=None,
    legend_names=None,
    tol_list=None,
    apply_style=True,
):
    """
    Create cross-correlations figure: |b12|, |b13|, |b23|, anisotropy index vs t/t0.

    iter_norm: Normalized iteration/time (x-axis)
    b12, b13, b23: Off-diagonal anisotropy tensor components (y-axis, plotted as |b_ij|)
    anis_index: Anisotropy index (y-axis, black curve)
    ps: Plot style dict
    axis_labels: {"x": "t/t₀", "y": "Cross-correlations / Anisotropy index"}
    legend_names: {"b12": "...", "b13": "...", "b23": "...", "anis": "..."}
    tol_list: List of tolerance values (e.g. [0.001, 0.005, 0.01]) for horizontal lines.
    """
    # Cross-correlations often use log scale; allow from ps
    ps = dict(ps)
    axis_labels = axis_labels or {"x": "t/t₀", "y": "Cross-correlations / Anisotropy index"}
    legend_names = legend_names or {"b12": "|b12|", "b13": "|b13|", "b23": "|b23|", "anis": "Anisotropy index"}

    iter_norm = np.asarray(iter_norm, dtype=float)
    b12 = np.asarray(np.abs(b12), dtype=float)
    b13 = np.asarray(np.abs(b13), dtype=float)
    b23 = np.asarray(np.abs(b23), dtype=float)
    anis_index = np.asarray(anis_index, dtype=float)

    colors = _get_palette(ps)
    curves = [(b12, "b12"), (b13, "b13"), (b23, "b23")]

    fig = go.Figure()
    for i, (arr, key) in enumerate(curves):
        c, lw, dash, _, _ = resolve_curve_style(key, i, colors, ps, _PLOT_KEY_CROSS)
        fig.add_trace(go.Scatter(
            x=iter_norm, y=arr, mode="lines",
            name=legend_names.get(key, key),
            line=dict(color=c, width=lw, dash=dash),
        ))

    c_anis, lw_anis, dash_anis, _, _ = resolve_curve_style("anis", 3, colors, ps, _PLOT_KEY_CROSS)
    anis_color = ps.get("anis_index_color") or c_anis
    fig.add_trace(go.Scatter(
        x=iter_norm, y=anis_index, mode="lines",
        name=legend_names.get("anis", "Anisotropy index"),
        line=dict(color=anis_color, width=lw_anis, dash=dash_anis),
    ))

    # Tolerance lines (horizontal lines at tol values)
    if tol_list:
        tol_colors = ["lightcoral", "lightpink", "mistyrose"]
        for i, tol in enumerate(tol_list):
            tol = float(tol)
            color = tol_colors[i % len(tol_colors)]
            fig.add_hline(y=tol, line_dash="dot", line_color=color, line_width=1.5,
                         annotation_text="", showlegend=False)
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                name=f"{tol:.1%} tolerance",
                showlegend=True,
            ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "t/t₀"),
        yaxis_title=axis_labels.get("y", "Cross-correlations / Anisotropy index"),
        margin=dict(l=60, r=40, t=50, b=50),
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
# Deviations (|E_x−1/3|, |E_y−1/3|, |E_z−1/3|, max dev vs t/t0)
# =============================================================================

def create_deviations_figure(
    iter_norm,
    devx,
    devy,
    devz,
    maxdev,
    ps,
    *,
    axis_labels=None,
    legend_names=None,
    tol_list=None,
    stationary_t=None,
    apply_style=True,
):
    """
    Create deviations figure: |E_x−1/3|, |E_y−1/3|, |E_z−1/3|, max dev vs t/t0.

    iter_norm: Normalized iteration/time (x-axis)
    devx, devy, devz: Absolute deviations from 1/3 (y-axis)
    maxdev: Maximum of devx, devy, devz (y-axis, black curve)
    ps: Plot style dict
    axis_labels: {"x": "t/t₀", "y": "Absolute deviation"}
    legend_names: {"devx": "...", "devy": "...", "devz": "...", "maxdev": "..."}
    tol_list: List of tolerance values (e.g. [0.005, 0.01, 0.02]) for horizontal lines.
    stationary_t: Normalized time (t/t0) for statistical stationarity vertical line.
    """
    ps = dict(ps)
    axis_labels = axis_labels or {"x": "t/t₀", "y": "Absolute deviation"}
    legend_names = legend_names or {"devx": "devx", "devy": "devy", "devz": "devz", "maxdev": "Max deviation"}

    iter_norm = np.asarray(iter_norm, dtype=float)
    devx = np.asarray(devx, dtype=float)
    devy = np.asarray(devy, dtype=float)
    devz = np.asarray(devz, dtype=float)
    maxdev = np.asarray(maxdev, dtype=float)

    colors = _get_palette(ps)
    curves = [(devx, "devx"), (devy, "devy"), (devz, "devz")]

    fig = go.Figure()
    for i, (arr, key) in enumerate(curves):
        c, lw, dash, _, _ = resolve_curve_style(key, i, colors, ps, _PLOT_KEY_DEVIATIONS)
        fig.add_trace(go.Scatter(
            x=iter_norm, y=arr, mode="lines",
            name=legend_names.get(key, key),
            line=dict(color=c, width=lw, dash=dash),
        ))

    c_max, lw_max, dash_max, _, _ = resolve_curve_style("maxdev", 3, colors, ps, _PLOT_KEY_DEVIATIONS)
    maxdev_color = ps.get("maxdev_color") or c_max
    fig.add_trace(go.Scatter(
        x=iter_norm, y=maxdev, mode="lines",
        name=legend_names.get("maxdev", "Max deviation"),
        line=dict(color=maxdev_color, width=lw_max, dash=dash_max),
    ))

    # Tolerance lines (horizontal lines at tol values)
    if tol_list:
        tol_colors = ["lightcoral", "lightpink", "mistyrose"]
        for i, tol in enumerate(tol_list):
            tol = float(tol)
            color = tol_colors[i % len(tol_colors)]
            fig.add_hline(y=tol, line_dash="dot", line_color=color, line_width=1.5,
                         annotation_text="", showlegend=False)
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                name=f"{tol:.1%} tolerance",
                showlegend=True,
            ))

    # Statistical stationarity vertical line
    if stationary_t is not None:
        stat_color = ps.get("stationary_line_color", "#800080")
        fig.add_vline(x=float(stationary_t), line_dash="dash", line_color=stat_color, line_width=1.5,
                     opacity=0.8, annotation_text="", showlegend=False)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=stat_color, width=1.5, dash="dash"),
            name="Statistical stationarity",
            showlegend=True,
        ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "t/t₀"),
        yaxis_title=axis_labels.get("y", "Absolute deviation"),
        margin=dict(l=60, r=40, t=50, b=50),
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
# Convergence (running std of E_x, E_y, E_z vs t/t0)
# =============================================================================

def create_convergence_figure(
    iter_norm,
    frac_x,
    frac_y,
    frac_z,
    ps,
    *,
    axis_labels=None,
    conv_windows=None,
    apply_style=True,
):
    """
    Create convergence figure: running std of (E_x, E_y, E_z) vs t/t0.

    iter_norm: Normalized iteration/time (x-axis)
    frac_x, frac_y, frac_z: Energy fractions (E_x, E_y, E_z)
    ps: Plot style dict
    axis_labels: {"x": "t/t₀", "y": "Running standard deviation"}
    conv_windows: List of window sizes for running std (e.g. [max(10, n//10), max(20, n//5)]).
                  If None, derived from data length.
    """
    ps = dict(ps)
    axis_labels = axis_labels or {"x": "t/t₀", "y": "Running standard deviation"}

    iter_norm = np.asarray(iter_norm, dtype=float)
    frac_x = np.asarray(frac_x, dtype=float)
    frac_y = np.asarray(frac_y, dtype=float)
    frac_z = np.asarray(frac_z, dtype=float)

    min_len = len(frac_x)
    fig = go.Figure()

    if min_len > 20:
        if conv_windows is None:
            conv_windows = [max(10, min_len // 10), max(20, min_len // 5)]
        colors_conv = ps.get("convergence_colors", ["#1f77b4", "#ff7f0e"])
        lw = ps.get("line_width", 1.5)

        for idx, window in enumerate(conv_windows):
            if window < min_len:
                running_stds = []
                for i in range(window, min_len + 1):
                    std_x = np.std(frac_x[i - window : i])
                    std_y = np.std(frac_y[i - window : i])
                    std_z = np.std(frac_z[i - window : i])
                    avg_std = (std_x + std_y + std_z) / 3
                    running_stds.append(avg_std)
                running_stds = np.asarray(running_stds, dtype=float)
                conv_time = iter_norm[window - 1 : window - 1 + len(running_stds)]
                color = colors_conv[idx % len(colors_conv)]
                fig.add_trace(go.Scatter(
                    x=conv_time, y=running_stds, mode="lines",
                    name=f"Running std (window={window})",
                    line=dict(color=color, width=lw),
                ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "t/t₀"),
        yaxis_title=axis_labels.get("y", "Running standard deviation"),
        margin=dict(l=60, r=40, t=50, b=50),
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
# Lumley Triangle (ξ, η trajectory)
# =============================================================================

def create_lumley_triangle_figure(
    xi,
    eta,
    ps,
    *,
    axis_labels=None,
    apply_style=True,
):
    """
    Create Lumley triangle figure: (ξ, η) trajectory with realizability boundaries.

    xi, eta: Lumley invariants arrays
    ps: Plot style dict
    axis_labels: {"x": "ξ", "y": "η"}
    """
    axis_labels = axis_labels or {"x": "ξ", "y": "η"}
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)

    fig = go.Figure()

    # Realizability boundaries
    xi_vals = np.linspace(-1 / 6, 1 / 3, 300)
    eta_two_comp = np.sqrt(1 / 27 + 2 * xi_vals**3)
    eta_axi_exp = -xi_vals[xi_vals <= 0]
    eta_axi_con = xi_vals[xi_vals >= 0]
    boundary_color = "#d4d4d4" if "dark" in str(ps.get("template", "")).lower() else "black"

    fig.add_trace(go.Scatter(
        x=xi_vals[xi_vals <= 0], y=eta_axi_exp, mode="lines",
        line=dict(color=boundary_color, width=1.5), name="Axisymmetric expansion", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=xi_vals[xi_vals >= 0], y=eta_axi_con, mode="lines",
        line=dict(color=boundary_color, width=1.5), name="Axisymmetric contraction", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=xi_vals, y=eta_two_comp, mode="lines",
        line=dict(color="red", width=1.5), name="Two-component limit", showlegend=True,
    ))
    # Lower boundary (invisible) - tonexty fills from trace above down to this
    eta_lower = np.where(xi_vals < 0, -xi_vals, xi_vals)
    fig.add_trace(go.Scatter(
        x=xi_vals, y=eta_lower, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fill_color = "rgba(62, 62, 66, 0.3)" if "dark" in str(ps.get("template", "")).lower() else "rgba(211, 211, 211, 0.3)"
    fig.add_trace(go.Scatter(
        x=xi_vals, y=eta_two_comp, mode="lines", fill="tonexty", fillcolor=fill_color,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Trajectory: wire-like segments with Viridis colormap
    n = len(xi)
    for i in range(1, n):
        t_val = i / max(n, 1)
        r, g, b_val = 0.267, 0.005 + 0.866 * t_val, 0.329 + 0.671 * (1 - t_val)
        color_rgb = f"rgba({int(r*255)}, {int(min(1, g)*255)}, {int(min(1, b_val)*255)}, 0.8)"
        fig.add_trace(go.Scatter(
            x=xi[i - 1 : i + 1].tolist(), y=eta[i - 1 : i + 1].tolist(), mode="lines",
            line=dict(color=color_rgb, width=1.5), showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=xi.tolist(), y=eta.tolist(), mode="markers",
        marker=dict(size=3, color=np.linspace(0, 1, len(xi)), colorscale="Viridis", line=dict(width=0.5, color="black"), opacity=0.9),
        name="Trajectory", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[xi[0]], y=[eta[0]], mode="markers",
        marker=dict(size=12, color="red", symbol="circle", line=dict(width=2, color="black")),
        name="Start", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[xi[-1]], y=[eta[-1]], mode="markers",
        marker=dict(size=12, color="green", symbol="circle", line=dict(width=2, color="black")),
        name="End", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=12, color="yellow", symbol="star", line=dict(width=1.5, color="black")),
        name="Isotropic", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[-1 / 6], y=[1 / 6], mode="markers",
        marker=dict(size=10, color="magenta", symbol="circle", line=dict(width=1.5, color="black")),
        name="2-component axisym", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[1 / 3], y=[1 / 3], mode="markers",
        marker=dict(size=10, color="blue", symbol="circle", line=dict(width=1.5, color="black")),
        name="1-component", showlegend=True,
    ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "ξ"),
        yaxis_title=axis_labels.get("y", "η"),
        margin=dict(l=60, r=40, t=50, b=50),
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)
    if ps.get("show_plot_title") and ps.get("plot_title"):
        fig.update_layout(title=ps["plot_title"])
    if apply_style:
        fig = apply_plot_style_base(fig, ps)
    return fig
