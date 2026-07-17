"""
Energy Spectra — Time-averaged raw and normalized spectrum rendering.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List

from utils.report_builder import capture_button
from utils.export_figs import export_panel
from visualizations.spectra_vis import (
    create_raw_spectrum_figure,
    create_normalized_spectrum_figure,
)

from .plot_style import get_plot_style, apply_plot_style
from .data_helpers import compute_time_avg, compute_time_avg_norm
from .file_loading import _default_labelify


def render_time_averaged(
    data_dir: Path,
    sim_groups: Dict[str, List[str]],
    norm_groups: Dict[str, List[str]],
) -> bool:
    """
    Render time-averaged raw and normalized spectra.
    Returns True if rendered, False if early exit (e.g. no sim_groups).
    """
    st.header("Time-Averaged Energy Spectra")

    if not sim_groups:
        st.warning("No spectrum*.dat groups found.")
        return False

    sim_options = sorted(sim_groups.keys())
    legend_names = st.session_state.spectrum_legend_names
    display_to_prefix = {
        legend_names.get(prefix, _default_labelify(prefix)): prefix
        for prefix in sim_options
    }
    display_names = list(display_to_prefix.keys())

    kolm_scaling_name = st.sidebar.selectbox(
        "Scale Kolmogorov Line to Spectra:",
        ["None"] + display_names,
        index=1 if display_names else 0,
        help="Select the simulation curve to scale the k^(-5/3) line against.",
        key="energy_kolm_scaling",
    )
    kolm_scaling_prefix = display_to_prefix[kolm_scaling_name] if kolm_scaling_name != "None" else None

    total_files = min(len(g) for g in sim_groups.values())
    if total_files < 1:
        st.warning("No spectrum files available to average.")
        return False

    if total_files == 1:
        start_idx, end_idx = 1, 1
        st.sidebar.caption("Averaging over 1 file (only snapshot available)")
    else:
        # Default start skips early transients when possible, but must stay < total_files
        # so the end-index slider has a valid (min < max) range.
        default_start = min(20, total_files - 1)
        start_idx = st.sidebar.slider(
            "Start file index",
            1,
            total_files,
            default_start,
            key="energy_start_idx",
        )
        if start_idx >= total_files:
            end_idx = total_files
            st.sidebar.caption(f"End file index: {end_idx}")
        else:
            end_idx = st.sidebar.slider(
                "End file index",
                start_idx,
                total_files,
                total_files,
                key="energy_end_idx",
            )

    show_kolm = st.sidebar.checkbox("Show Kolmogorov -5/3 line", value=True, key="energy_show_kolm")
    so = st.session_state.setdefault(
        "spectra_options",
        {
            "show_std": True,
            "show_error_bars": True,
            "pope_scaling_prefix": None,
            "kmin": 3.0,
            "kmax": 20.0,
            "kolm_scale_factor": 1.0,
        },
    )
    err_opts = ["Shaded band", "Error bars", "Both", "None"]
    err_idx = (
        2
        if (so.get("show_std", True) and so.get("show_error_bars", True))
        else (0 if so.get("show_std", True) else (1 if so.get("show_error_bars", True) else 3))
    )
    error_display = st.sidebar.radio(
        "Error display",
        err_opts,
        index=min(err_idx, 3),
        help="Choose how to display ±1σ uncertainty",
        key="energy_error_display",
    )
    show_std = error_display in ["Shaded band", "Both"]
    show_error_bars = error_display in ["Error bars", "Both"]
    so["show_std"], so["show_error_bars"] = show_std, show_error_bars
    show_normalized = st.sidebar.checkbox(
        "Show normalized (collapsed) panel with Pope", value=True, key="energy_show_norm"
    )

    pope_scaling_prefix = None
    if show_normalized and norm_groups:
        norm_options = sorted(norm_groups.keys())
        norm_legend_names = st.session_state.norm_legend_names
        norm_display_to_prefix = {
            norm_legend_names.get(prefix, _default_labelify(prefix)): prefix
            for prefix in norm_options
        }
        norm_display_names = list(norm_display_to_prefix.keys())
        prefix_to_display = {v: k for k, v in norm_display_to_prefix.items()}
        saved_pope = so.get("pope_scaling_prefix")
        pope_idx = 0
        if saved_pope and saved_pope in prefix_to_display:
            dn = prefix_to_display[saved_pope]
            if dn in norm_display_names:
                pope_idx = norm_display_names.index(dn) + 1
        pope_opts = ["None (Default: Use all plotted simulations)"] + norm_display_names
        pope_scaling_name = st.sidebar.selectbox(
            "Plot Pope Model from Spectra:",
            pope_opts,
            index=min(pope_idx, len(pope_opts) - 1),
            help="If set, only the Pope model corresponding to this selected simulation will be plotted.",
            key="energy_pope_scaling",
        )
        if pope_scaling_name != "None (Default: Use all plotted simulations)":
            pope_scaling_prefix = norm_display_to_prefix[pope_scaling_name]
        so["pope_scaling_prefix"] = pope_scaling_prefix

    kmin = st.sidebar.number_input(
        "Inertial range k_min", min_value=1.0, value=float(so.get("kmin", 3.0)), key="energy_kmin"
    )
    kmax = st.sidebar.number_input(
        "Inertial range k_max", min_value=kmin + 1e-6, value=float(so.get("kmax", 20.0)), key="energy_kmax"
    )
    kolm_scale_factor = st.sidebar.slider(
        "Kolmogorov Line Scale Factor",
        0.1,
        5.0,
        float(so.get("kolm_scale_factor", 1.0)),
        step=0.05,
        help="Manually scale the k^(-5/3) line up (> 1.0) or down (< 1.0) for better visual fit.",
        key="energy_kolm_factor",
    )
    so["kmin"], so["kmax"], so["kolm_scale_factor"] = kmin, kmax, kolm_scale_factor

    k_scale, E_avg_scale = None, None
    if kolm_scaling_prefix and show_kolm:
        selected_files_scale = tuple(sim_groups[kolm_scaling_prefix][start_idx - 1 : end_idx])
        if selected_files_scale:
            k_scale, E_avg_scale, _ = compute_time_avg(selected_files_scale)

    ps_raw = get_plot_style("Raw Energy Spectrum")
    datasets_raw = []
    for sim_prefix, files in sorted(sim_groups.items()):
        selected_files = tuple(files[start_idx - 1 : end_idx])
        if not selected_files:
            continue
        result = compute_time_avg(selected_files)
        if result[0] is None:
            continue
        k_vals, E_avg, E_std = result
        datasets_raw.append(
            {"sim_prefix": sim_prefix, "x": k_vals, "y": E_avg, "y_std": E_std}
        )

    if not datasets_raw:
        st.info("No valid spectra could be plotted from selected range.")
        return False

    kolm_scale_data = (
        {"x": k_scale, "y": E_avg_scale}
        if (show_kolm and k_scale is not None and E_avg_scale is not None)
        else None
    )

    fig_raw = create_raw_spectrum_figure(
        datasets_raw,
        ps_raw,
        show_std=show_std,
        show_error_bars=show_error_bars,
        show_kolmogorov=show_kolm and kolm_scale_data is not None,
        kmin=kmin,
        kmax=kmax,
        kolm_scale_factor=kolm_scale_factor,
        kolm_scale_data=kolm_scale_data,
        axis_labels=st.session_state.axis_labels_raw,
        legend_names=st.session_state.spectrum_legend_names,
        apply_style=False,
    )
    fig_raw = apply_plot_style(fig_raw, ps_raw)

    fig_norm = None
    if show_normalized and norm_groups:
        ps_norm = get_plot_style("Normalized Spectrum")
        datasets_norm = []
        for norm_prefix, files in sorted(norm_groups.items()):
            selected_files = tuple(files[start_idx - 1 : end_idx])
            if not selected_files:
                continue
            result = compute_time_avg_norm(selected_files)
            if result[0] is None:
                continue
            keta, En_avg, En_std, Ep_avg = result
            datasets_norm.append(
                {
                    "sim_prefix": norm_prefix,
                    "x": keta,
                    "y": En_avg,
                    "y_std": En_std,
                    "y_pope": Ep_avg,
                }
            )
        fig_norm = create_normalized_spectrum_figure(
            datasets_norm,
            ps_norm,
            show_std=show_std,
            show_error_bars=show_error_bars,
            pope_scaling_prefix=pope_scaling_prefix,
            axis_labels=st.session_state.axis_labels_norm,
            legend_names=st.session_state.norm_legend_names,
            apply_style=False,
        )
        fig_norm = apply_plot_style(fig_norm, ps_norm)

    if fig_norm is not None:
        st.markdown("### Raw Energy Spectrum")
        st.plotly_chart(fig_raw, width="stretch")
        capture_button(fig_raw, title="Energy Spectra (Raw)", source_page="Energy Spectra")
        st.markdown("### Normalized (Collapsed) Spectrum")
        st.plotly_chart(fig_norm, width="stretch")
        capture_button(fig_norm, title="Energy Spectra (Normalized)", source_page="Energy Spectra")
    else:
        st.plotly_chart(fig_raw, width="stretch")
        capture_button(fig_raw, title="Energy Spectra", source_page="Energy Spectra")

    st.subheader("Export")
    export_panel(fig_raw, data_dir, "energy_spectra_raw")
    if fig_norm is not None:
        export_panel(fig_norm, data_dir, "energy_spectra_normalized")

    first_sim = list(sorted(sim_groups.keys()))[0]
    files0 = tuple(sim_groups[first_sim][start_idx - 1 : end_idx])
    result0 = compute_time_avg(files0)
    if result0[0] is not None:
        k0, E0, S0 = result0
        df_out = pd.DataFrame({"k": k0, "E_avg": E0, "E_std": S0})
        st.download_button(
            "Download averaged CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="energy_spectrum_avg.csv",
            mime="text/csv",
            key="energy_download_csv",
        )

    return True
