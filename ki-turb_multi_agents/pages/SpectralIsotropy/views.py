"""
Spectral Isotropy — Tab renderers (IC(k), Component Spectra, Summary).
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from utils.report_builder import capture_button
from utils.export_figs import export_panel
from utils.plot_style import apply_axis_limits, apply_figure_size, resolve_line_style
from visualizations.spectral_isotropy_vis import (
    create_ic_isotropy_figure,
    create_component_spectra_figure,
)

from utils.plot_style import _get_palette
from .plot_style import get_plot_style, apply_plot_style, resolve_curve_style
from .file_loading import _default_labelify
from core_physics.spectral_isotropy import snapshot_ic_curve

from .data_helpers import read_isotropy_coeff_cached, avg_isotropy_coeff_from_files


def render_ic_tab(
    data_dir: Path,
    ic_groups: Dict[str, List[str]],
    start_idx: int,
    end_idx: int,
    show_snapshot_lines: bool,
    show_std_band: bool,
    show_error_bars: bool,
) -> bool:
    """Render IC(k) Time-Avg tab. Returns False if no valid data."""
    st.subheader("Time-averaged Spectral Isotropy Ratio")

    plot_name_ic = "IC(k) Time-Avg"
    ps_ic = get_plot_style(plot_name_ic)
    colors_ic = _get_palette(ps_ic)

    sim_items_ic = []
    for idx, (sim_prefix, files) in enumerate(sorted(ic_groups.items())):
        selected_files = tuple(files[start_idx - 1 : end_idx])
        if not selected_files:
            continue
        avg = avg_isotropy_coeff_from_files(selected_files)
        if avg is None:
            continue
        d = dict(avg)
        if show_snapshot_lines:
            snapshot_curves = []
            for f in selected_files:
                rd = read_isotropy_coeff_cached(str(f))
                if rd.size == 0:
                    continue
                snapshot_curves.append(snapshot_ic_curve(rd, kind="standard"))
            d["snapshot_curves"] = snapshot_curves

        c_ic, lw_ic, dash_ic, mk_ic, ms_ic = resolve_curve_style(
            "IC", idx, colors_ic, ps_ic, plot_name_ic
        )
        color_sim, lw_sim, dash_sim, marker_sim, msize_sim, override_on_sim = resolve_line_style(
            sim_prefix, idx, colors_ic, ps_ic, style_key="per_sim_style_ic",
            include_marker=True, default_marker="circle"
        )
        if ps_ic.get("enable_per_curve_style", False):
            d["_style"] = {
                "color": c_ic, "width": lw_ic, "dash": dash_ic, "marker": mk_ic, "msize": ms_ic,
                "override_on": (mk_ic != "circle" or ms_ic > 0),
            }
        else:
            d["_style"] = {
                "color": color_sim, "width": lw_sim, "dash": dash_sim,
                "marker": marker_sim, "msize": msize_sim, "override_on": override_on_sim,
            }
        sim_items_ic.append((sim_prefix, d))

    axis_labels_ic = {
        "x": st.session_state.axis_labels_spec_iso["k"],
        "y": st.session_state.axis_labels_spec_iso["ic"],
    }
    sim_legend_map = {
        k: st.session_state.spec_iso_sim_legend_names.get(k, _default_labelify(k))
        for k, _ in sim_items_ic
    }
    fig_ic = create_ic_isotropy_figure(
        sim_items_ic,
        ps_ic,
        show_std_band=show_std_band,
        show_error_bars=show_error_bars,
        show_snapshot_lines=show_snapshot_lines,
        axis_labels=axis_labels_ic,
        simulation_legend_names=sim_legend_map,
        ic_snap_label=st.session_state.spec_iso_legends["IC_snap"],
        apply_style=True,
    )
    if fig_ic is None:
        st.error("No valid data in selected isotropy files.")
        return False

    layout_kwargs_ic = dict(height=500)
    layout_kwargs_ic = apply_axis_limits(layout_kwargs_ic, ps_ic)
    layout_kwargs_ic = apply_figure_size(layout_kwargs_ic, ps_ic)
    fig_ic.update_layout(**layout_kwargs_ic)
    fig_ic = apply_plot_style(fig_ic, ps_ic)
    st.plotly_chart(fig_ic, width="stretch")
    capture_button(fig_ic, title="Spectral Isotropy (IC)", source_page="Spectral Isotropy")
    export_panel(fig_ic, data_dir, "spectral_isotropy_IC")
    return True


def render_component_spectra_tab(
    data_dir: Path,
    ic_groups: Dict[str, List[str]],
    start_idx: int,
    end_idx: int,
    show_component_spectra: bool,
    show_curves: Optional[List[str]] = None,
) -> bool:
    """Render Component Spectra tab."""
    st.subheader("Component Spectra (time-avg)")

    if not show_component_spectra:
        st.info("Component spectra not available (disabled).")
        return True

    plot_name_eii = "Component Spectra"
    ps_eii = get_plot_style(plot_name_eii)
    colors_eii = _get_palette(ps_eii)

    sim_items_eii = []
    for idx, (sim_prefix, files) in enumerate(sorted(ic_groups.items())):
        selected_files = tuple(files[start_idx - 1 : end_idx])
        if not selected_files:
            continue
        avg = avg_isotropy_coeff_from_files(selected_files)
        if avg is None or avg.get("E11_mean") is None:
            continue
        d = dict(avg)
        color_base, lw_base, dash_base, _, _, override_on_base = resolve_line_style(
            sim_prefix, idx, colors_eii, ps_eii, style_key="per_sim_style_eii",
            include_marker=True, default_marker="circle"
        )
        if ps_eii.get("enable_per_curve_style", False):
            d["_curve_styles"] = {}
            for i, curve in enumerate(["E11", "E22", "E33"]):
                c, lw, dash, _, _ = resolve_curve_style(curve, i, colors_eii, ps_eii, plot_name_eii)
                d["_curve_styles"][curve] = {"color": c, "width": lw, "dash": dash}
        elif override_on_base:
            d["_style"] = {"color": color_base, "width": lw_base, "dash": dash_base}
        sim_items_eii.append((sim_prefix, d))

    axis_labels_eii = {
        "x": st.session_state.axis_labels_spec_iso["k"],
        "y": st.session_state.axis_labels_spec_iso["ek"],
    }
    sim_legend_map = {
        k: st.session_state.spec_iso_sim_legend_names.get(k, _default_labelify(k))
        for k, _ in sim_items_eii
    }
    fig_eii = create_component_spectra_figure(
        sim_items_eii,
        ps_eii,
        axis_labels=axis_labels_eii,
        simulation_legend_names=sim_legend_map,
        curve_legend_names=st.session_state.spec_iso_legends,
        show_curves=show_curves,
        apply_style=True,
    )
    if fig_eii is None:
        st.info("Component spectra not available (missing columns in data).")
        return True

    layout_kwargs_eii = dict(width=700, height=600)
    layout_kwargs_eii = apply_axis_limits(layout_kwargs_eii, ps_eii)
    fig_eii.update_layout(**layout_kwargs_eii)
    fig_eii = apply_plot_style(fig_eii, ps_eii)
    st.plotly_chart(fig_eii, width="content")
    capture_button(fig_eii, title="Spectral Isotropy (E_ii)", source_page="Spectral Isotropy")
    export_panel(fig_eii, data_dir, "spectral_isotropy_Eii")
    return True


def render_summary_tab(
    ic_groups: Dict[str, List[str]],
    start_idx: int,
    end_idx: int,
) -> bool:
    """Render Summary tab."""
    st.subheader("Summary")
    summary_rows = []
    for sim_prefix, files in sorted(ic_groups.items()):
        selected_files = tuple(files[start_idx - 1 : end_idx])
        if not selected_files:
            continue
        avg = avg_isotropy_coeff_from_files(selected_files)
        if avg is None:
            continue
        IC_mean = avg["IC_mean"]
        IC_std = avg["IC_std"]
        legend_name = st.session_state.spec_iso_sim_legend_names.get(
            sim_prefix, _default_labelify(sim_prefix)
        )
        summary_rows.append({
            "Simulation": legend_name,
            "Snapshots used": len(selected_files),
            "Mean IC": float(np.nanmean(IC_mean)),
            "Std(IC)": float(np.nanmean(IC_std)),
            "Min IC": float(np.nanmin(IC_mean)),
            "Max IC": float(np.nanmax(IC_mean)),
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, width="stretch")
        st.download_button(
            "Download summary CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="spectral_isotropy_summary.csv",
            mime="text/csv",
            key="speciso_download_summary",
        )
    else:
        st.info("No data available for summary.")
    return True
