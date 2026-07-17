"""
Real Isotropy — Tab renderers (Energy & Lumley, Anisotropy Tensor, Deviations & Convergence, Summary).
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from plotly.colors import hex_to_rgb
from utils.report_builder import capture_button
from utils.export_figs import export_panel
from utils.plot_style import apply_axis_limits, apply_figure_size, _get_palette
from visualizations.real_isotropy_vis import (
    create_energy_fractions_figure,
    create_lumley_triangle_figure,
    create_diagonal_bii_figure,
    create_cross_correlations_figure,
    create_deviations_figure,
    create_convergence_figure,
)

from .plot_style import get_plot_style, apply_plot_style, resolve_curve_style


def _apply_energy_fractions_colors(fig, ps: Dict[str, Any], colors_orig: Dict[str, str]):
    """Re-apply colors after plot style for Energy Fractions (MA vs raw)."""
    if ps.get("enable_per_curve_style", False):
        return
    ma_line_width = ps.get("line_width", 2.2) * 1.1
    raw_opacity = ps.get("raw_data_opacity", 0.5)
    raw_marker_size = max(2, ps.get("marker_size", 6) * 0.4)
    for trace in fig.data:
        if trace.name and "(MA-" in trace.name:
            if "Ex" in trace.name or "E<sub>x</sub>" in trace.name:
                trace.line.color = colors_orig["primary"]
            elif "Ey" in trace.name or "E<sub>y</sub>" in trace.name:
                trace.line.color = colors_orig["secondary"]
            elif "Ez" in trace.name or "E<sub>z</sub>" in trace.name:
                trace.line.color = colors_orig["tertiary"]
            trace.line.width = ma_line_width
            trace.opacity = 1.0
        elif trace.name and "(raw)" in trace.name:
            if "Ex" in trace.name or "E<sub>x</sub>" in trace.name:
                rgb = hex_to_rgb(colors_orig["primary"])
                trace.line.color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
                trace.marker.color = colors_orig["primary"]
            elif "Ey" in trace.name or "E<sub>y</sub>" in trace.name:
                rgb = hex_to_rgb(colors_orig["secondary"])
                trace.line.color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
                trace.marker.color = colors_orig["secondary"]
            elif "Ez" in trace.name or "E<sub>z</sub>" in trace.name:
                rgb = hex_to_rgb(colors_orig["tertiary"])
                trace.line.color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
                trace.marker.color = colors_orig["tertiary"]
            if hasattr(trace, "marker") and trace.marker:
                trace.marker.size = raw_marker_size
                trace.marker.opacity = raw_opacity
            trace.opacity = 1.0


def render_tab1(
    data_dir: Path,
    time_norm,
    E_x,
    E_y,
    E_z,
    inv: Dict[str, Any],
    ma_win: int,
    tol_list_a: List[float],
    stationary_t: float,
):
    """Render Energy Fractions (A) + Lumley Triangle (B)."""
    labels = st.session_state.axis_labels_real_iso
    legends = st.session_state.real_iso_legends

    # (a) Energy Fractions
    plot_name_a = "Energy Fractions (A)"
    ps_a = get_plot_style(plot_name_a)
    cols_a = _get_palette(ps_a)
    colors_orig = {
        "primary": cols_a[0 % len(cols_a)],
        "secondary": cols_a[1 % len(cols_a)],
        "tertiary": cols_a[2 % len(cols_a)],
    }
    legend_names_a = {
        "frac_x": legends["Ex"],
        "frac_y": legends["Ey"],
        "frac_z": legends["Ez"],
    }
    axis_labels_a = {"x": labels["time"], "y": labels["energy_frac"]}
    fig_a = create_energy_fractions_figure(
        time_norm,
        E_x,
        E_y,
        E_z,
        ps_a,
        axis_labels=axis_labels_a,
        legend_names=legend_names_a,
        apply_style=False,
        ma_win=ma_win if ma_win and ma_win > 1 else None,
        add_raw_suffix=True,
        tol_list=tol_list_a,
        stationary_t=stationary_t,
    )
    layout_kwargs_a = dict(
        xaxis_title=labels["time"],
        yaxis_title=labels["energy_frac"],
        height=420,
    )
    layout_kwargs_a = apply_axis_limits(layout_kwargs_a, ps_a)
    layout_kwargs_a = apply_figure_size(layout_kwargs_a, ps_a)
    fig_a.update_layout(**layout_kwargs_a)
    fig_a = apply_plot_style(fig_a, ps_a)
    _apply_energy_fractions_colors(fig_a, ps_a, colors_orig)
    st.plotly_chart(fig_a, width="stretch")
    capture_button(fig_a, title="Real-Space Isotropy Analysis (Part A)", source_page="Real Isotropy")
    export_panel(fig_a, data_dir, "real_iso_energy_fractions")

    # (b) Lumley Triangle
    plot_name_b = "Lumley Triangle (B)"
    ps_b = get_plot_style(plot_name_b)
    xi, eta = inv["xi"], inv["eta"]
    axis_labels_b = {"x": labels["lumley_x"], "y": labels["lumley_y"]}
    fig_b = create_lumley_triangle_figure(xi, eta, ps_b, axis_labels=axis_labels_b, apply_style=True)
    layout_kwargs_b = dict(height=420, showlegend=True)
    layout_kwargs_b = apply_axis_limits(layout_kwargs_b, ps_b)
    layout_kwargs_b = apply_figure_size(layout_kwargs_b, ps_b)
    fig_b.update_layout(**layout_kwargs_b)
    fig_b = apply_plot_style(fig_b, ps_b)
    st.plotly_chart(fig_b, width="stretch")
    capture_button(fig_b, title="Real-Space Isotropy Analysis (Part B)", source_page="Real Isotropy")
    export_panel(fig_b, data_dir, "real_iso_lumley_triangle")


def render_tab2(
    data_dir: Path,
    time_norm,
    b: Dict[str, Any],
    inv: Dict[str, Any],
    tol_list_c: List[float],
    tol_list_d: List[float],
):
    """Render Diagonal b_ii (C) + Cross-correlations (D)."""
    labels = st.session_state.axis_labels_real_iso
    legends = st.session_state.real_iso_legends

    # (c) Diagonal b_ii
    plot_name_c = "Diagonal b_ii (C)"
    ps_c = get_plot_style(plot_name_c)
    legend_names_c = {"b11": legends["b11"], "b22": legends["b22"], "b33": legends["b33"]}
    axis_labels_c = {"x": labels["time"], "y": labels["bij"]}
    fig_c = create_diagonal_bii_figure(
        time_norm,
        b["b11"],
        b["b22"],
        b["b33"],
        ps_c,
        axis_labels=axis_labels_c,
        legend_names=legend_names_c,
        apply_style=False,
        tol_list=tol_list_c,
    )
    layout_kwargs_c = dict(
        xaxis_title=labels["time"],
        yaxis_title=labels["bij"],
        height=360,
    )
    layout_kwargs_c = apply_axis_limits(layout_kwargs_c, ps_c)
    layout_kwargs_c = apply_figure_size(layout_kwargs_c, ps_c)
    fig_c.update_layout(**layout_kwargs_c)
    fig_c = apply_plot_style(fig_c, ps_c)
    colors_c = _get_palette(ps_c)
    for i, curve in enumerate(["b11", "b22", "b33"]):
        if i < len(fig_c.data):
            c, lw, dash, mk, ms = resolve_curve_style(curve, i, colors_c, ps_c, plot_name_c)
            fig_c.data[i].line.color = c
            fig_c.data[i].line.width = lw
            fig_c.data[i].line.dash = dash
    st.plotly_chart(fig_c, width="stretch")
    export_panel(fig_c, data_dir, "real_iso_bii_diag")

    # (d) Cross-correlations
    plot_name_d = "Cross-correlations (D)"
    ps_d = get_plot_style(plot_name_d)
    legend_names_d = {
        "b12": legends["b12"],
        "b13": legends["b13"],
        "b23": legends["b23"],
        "anis": legends["anis"],
    }
    axis_labels_d = {"x": labels["time"], "y": labels["cross"]}
    fig_d = create_cross_correlations_figure(
        time_norm,
        b["b12"],
        b["b13"],
        b["b23"],
        inv["anis_index"],
        ps_d,
        axis_labels=axis_labels_d,
        legend_names=legend_names_d,
        tol_list=tol_list_d,
        apply_style=False,
    )
    layout_kwargs_d = dict(
        xaxis_title=labels["time"],
        yaxis_title=labels["cross"],
        height=360,
    )
    layout_kwargs_d = apply_axis_limits(layout_kwargs_d, ps_d)
    layout_kwargs_d = apply_figure_size(layout_kwargs_d, ps_d)
    fig_d.update_layout(**layout_kwargs_d)
    fig_d = apply_plot_style(fig_d, ps_d)
    colors_d = _get_palette(ps_d)
    for i, curve in enumerate(["b12", "b13", "b23", "anis"]):
        if i < len(fig_d.data):
            c, lw, dash, mk, ms = resolve_curve_style(curve, i, colors_d, ps_d, plot_name_d)
            fig_d.data[i].line.color = c
            fig_d.data[i].line.width = lw
            fig_d.data[i].line.dash = dash
    st.plotly_chart(fig_d, width="stretch")
    export_panel(fig_d, data_dir, "real_iso_cross_corr")


def render_tab3(
    data_dir: Path,
    time_norm,
    E_x,
    E_y,
    E_z,
    tol_list_e: List[float],
    stationary_t: float,
):
    """Render Deviations (E) + Convergence (F)."""
    labels = st.session_state.axis_labels_real_iso
    devx = np.abs(E_x - 1 / 3)
    devy = np.abs(E_y - 1 / 3)
    devz = np.abs(E_z - 1 / 3)
    maxdev = np.maximum(np.maximum(devx, devy), devz)
    legend_names_e = {"devx": "devx", "devy": "devy", "devz": "devz", "maxdev": "Max deviation"}

    # (e) Deviations
    plot_name_e = "Deviations (E)"
    ps_e = get_plot_style(plot_name_e)
    axis_labels_e = {"x": labels["time"], "y": labels["dev"]}
    fig_e = create_deviations_figure(
        time_norm,
        devx,
        devy,
        devz,
        maxdev,
        ps_e,
        axis_labels=axis_labels_e,
        legend_names=legend_names_e,
        tol_list=tol_list_e,
        stationary_t=stationary_t,
        apply_style=False,
    )
    layout_kwargs_e = dict(
        xaxis_title=labels["time"],
        yaxis_title=labels["dev"],
        height=360,
    )
    layout_kwargs_e = apply_axis_limits(layout_kwargs_e, ps_e)
    layout_kwargs_e = apply_figure_size(layout_kwargs_e, ps_e)
    fig_e.update_layout(**layout_kwargs_e)
    fig_e = apply_plot_style(fig_e, ps_e)
    colors_e = _get_palette(ps_e)
    for i, curve in enumerate(["devx", "devy", "devz", "maxdev"]):
        if i < len(fig_e.data):
            c, lw, dash, mk, ms = resolve_curve_style(curve, i, colors_e, ps_e, plot_name_e)
            fig_e.data[i].line.color = c
            fig_e.data[i].line.width = lw
            fig_e.data[i].line.dash = dash
    st.plotly_chart(fig_e, width="stretch")
    export_panel(fig_e, data_dir, "real_iso_deviation")

    # (f) Convergence
    plot_name_f = "Convergence (F)"
    ps_f = get_plot_style(plot_name_f)
    min_len = len(E_x)
    conv_windows = [max(10, min_len // 10), max(20, min_len // 5)] if min_len > 20 else None
    conv_label = labels.get("convergence", "Running standard deviation")
    axis_labels_f = {"x": labels["time"], "y": conv_label}
    fig_f = create_convergence_figure(
        time_norm,
        E_x,
        E_y,
        E_z,
        ps_f,
        axis_labels=axis_labels_f,
        conv_windows=conv_windows,
        apply_style=False,
    )
    layout_kwargs_f = dict(
        xaxis_title=labels["time"],
        yaxis_title=conv_label,
        height=360,
    )
    layout_kwargs_f = apply_axis_limits(layout_kwargs_f, ps_f)
    layout_kwargs_f = apply_figure_size(layout_kwargs_f, ps_f)
    fig_f.update_layout(**layout_kwargs_f)
    fig_f = apply_plot_style(fig_f, ps_f)
    st.plotly_chart(fig_f, width="stretch")
    export_panel(fig_f, data_dir, "real_iso_convergence")


def render_summary(inv: Dict[str, Any], E_x, E_y, E_z):
    """Render final isotropy summary table and download."""
    df_sum = pd.DataFrame(
        [
            {
                "Final Ex": float(E_x[-1]),
                "Final Ey": float(E_y[-1]),
                "Final Ez": float(E_z[-1]),
                "Final anisotropy index": float(inv["anis_index"][-1]),
                "Mean anisotropy index": float(np.mean(inv["anis_index"])),
            }
        ]
    )
    st.dataframe(df_sum, width="stretch")
    st.download_button(
        "Download summary CSV",
        df_sum.to_csv(index=False).encode("utf-8"),
        file_name="real_isotropy_summary.csv",
        mime="text/csv",
        key="realiso_download_summary",
    )
