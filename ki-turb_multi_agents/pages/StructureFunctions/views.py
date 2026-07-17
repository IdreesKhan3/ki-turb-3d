"""
Structure Functions — Tab renderers (S_p, ESS, Table, Theory).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any

from utils.report_builder import capture_button
from utils.export_figs import export_panel
from utils.plot_style import apply_axis_limits, apply_figure_size, _get_palette, resolve_line_style, default_plot_style
from core_physics import zeta_p_she_leveque, TABLE_P, EXP_ZETA

from .plot_style import get_plot_style, apply_plot_style
from .data_helpers import compute_time_avg_structure, color_to_rgb_tuple
from .ess_inset import add_ess_inset


def render_sp_tab(
    data_dir: Path,
    sim_groups: Dict[str, Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    selected_ps: List[int],
    normalize_by_urms: bool,
    show_std_band: bool,
    show_error_bars: bool,
) -> bool:
    """Render S_p(r) vs r tab. Returns True if any data was plotted."""
    plot_name_sp = "S_p(r) vs r"
    ps_sp = get_plot_style(plot_name_sp)
    colors_sp = _get_palette(ps_sp)
    labels = st.session_state.axis_labels_structure
    legends = st.session_state.structure_legend_names

    fig_sp = go.Figure()
    plotted_any = False

    for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
        kind = sim_groups[sim_prefix]["kind"]
        files = sim_groups[sim_prefix]["files"][start_idx - 1 : end_idx]
        if not files:
            st.warning(f"No files found for {sim_prefix} in selected time range.")
            continue
        r, Sp_mean, Sp_std, urms, ps_here = compute_time_avg_structure(tuple(files), kind)
        if r is None:
            st.warning(f"Could not read structure function data for {sim_prefix}. Check file format.")
            continue
        if not Sp_mean:
            st.warning(f"No structure function data found for {sim_prefix}.")
            continue

        legend_base = legends.get(sim_prefix, sim_prefix.replace("_", " ").title())
        color_base, lw_base, dash_base, marker_base, msize_base, override_on = resolve_line_style(
            sim_prefix, idx, colors_sp, ps_sp,
            style_key="per_sim_style_structure",
            include_marker=True,
            default_marker="circle",
        )
        plotted_any = True

        for p in selected_ps:
            if p not in Sp_mean:
                continue
            y = Sp_mean[p].copy()
            ystd = Sp_std.get(p)
            if ystd is not None:
                ystd = ystd.copy()

            if normalize_by_urms and np.isfinite(urms) and float(urms) > 0.0:
                y = y / (urms ** p)
                if ystd is not None:
                    ystd = ystd / (urms ** p)

            line_color = color_base
            if override_on and marker_base and msize_base > 0:
                mode = "lines+markers"
                marker_dict = dict(symbol=marker_base, size=msize_base)
            else:
                mode = "lines"
                marker_dict = None

            trace_kwargs = dict(
                x=r, y=y,
                mode=mode,
                name=f"{legend_base}  (p={p})",
                line=dict(color=line_color, width=lw_base, dash=dash_base),
                hovertemplate="r=%{x:.3g}<br>S_p=%{y:.3g}<extra></extra>",
            )
            if marker_dict:
                trace_kwargs["marker"] = marker_dict
            if show_error_bars and ystd is not None:
                trace_kwargs["error_y"] = dict(type="data", array=ystd, visible=True, thickness=1, color=line_color)
            fig_sp.add_trace(go.Scatter(**trace_kwargs))

            if show_std_band and ystd is not None:
                rgb = color_to_rgb_tuple(line_color)
                fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps_sp['std_alpha']})"
                fig_sp.add_trace(
                    go.Scatter(
                        x=np.concatenate([r, r[::-1]]),
                        y=np.concatenate([y - ystd, (y + ystd)[::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    if not plotted_any:
        st.info("No valid structure function data in selected range.")
        return False

    layout_kwargs = dict(
        xaxis_title=labels.get("x_r", "Separation distance r"),
        yaxis_title=labels.get("y_sp", "Structure functions S<sub>p</sub>(r)"),
        legend_title="Simulation / Order",
        height=500,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps_sp)
    layout_kwargs = apply_figure_size(layout_kwargs, ps_sp)
    fig_sp.update_layout(**layout_kwargs)
    fig_sp = apply_plot_style(fig_sp, ps_sp)
    st.plotly_chart(fig_sp, width="stretch")
    capture_button(fig_sp, title="Structure Functions S_p(r)", source_page="Structure Functions")
    export_panel(fig_sp, data_dir, base_name="structure_functions_sp")
    return True


def render_ess_tab(
    data_dir: Path,
    sim_groups: Dict[str, Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    selected_ps: List[int],
    ref_p: int,
    normalize_by_urms: bool,
    show_std_band: bool,
    show_error_bars: bool,
    show_sl_theory: bool,
    show_exp_anom: bool,
    show_inset: bool,
    fit_rmin: float,
    fit_rmax: float,
) -> bool:
    """Render ESS tab (ESS plot + anomalies). Stores xi_all, anom_all, xi_err_all in session state."""
    plot_name_ess = "ESS (S_p vs S_3)"
    ps_ess = get_plot_style(plot_name_ess)
    colors_ess = _get_palette(ps_ess)
    labels = st.session_state.axis_labels_structure
    legends = st.session_state.structure_legend_names

    fig_ess = go.Figure()
    plotted_any = False
    xi_all = {}
    xi_err_all = {}
    anom_all = {}

    for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
        kind = sim_groups[sim_prefix]["kind"]
        files = sim_groups[sim_prefix]["files"][start_idx - 1 : end_idx]
        if not files:
            continue
        r, Sp_mean, Sp_std, urms, ps_here = compute_time_avg_structure(tuple(files), kind)
        if r is None:
            continue
        if ref_p not in Sp_mean:
            st.warning(f"Reference order p={ref_p} not available for {sim_prefix}. Available: {sorted(Sp_mean.keys()) if Sp_mean else 'none'}")
            continue

        legend_base = legends.get(sim_prefix, sim_prefix.replace("_", " ").title())
        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix, idx, colors_ess, ps_ess,
            style_key="per_sim_style_structure",
            include_marker=True,
            default_marker="circle",
        )
        plotted_any = True

        def _norm(p, arr):
            if normalize_by_urms and np.isfinite(urms) and float(urms) > 0.0:
                return arr / (urms ** p)
            return arr

        x = _norm(ref_p, Sp_mean[ref_p])
        xi_all[sim_prefix] = {}
        xi_err_all[sim_prefix] = {}
        anom_all[sim_prefix] = {}

        for p in selected_ps:
            if p not in Sp_mean:
                continue
            y = _norm(p, Sp_mean[p])
            y_std = _norm(p, Sp_std[p]) if p in Sp_std else None
            x_std = _norm(ref_p, Sp_std[ref_p]) if ref_p in Sp_std else None

            rmask = (
                (r >= fit_rmin) & (r <= fit_rmax)
                & np.isfinite(x) & (x > 0)
                & np.isfinite(y) & (y > 0)
            )

            trace_kwargs = dict(
                x=x, y=y,
                mode="lines+markers",
                name=f"{legend_base} (p={p})",
                line=dict(color=color, width=lw, dash=dash),
                marker=dict(symbol=marker, size=msize),
                hovertemplate=f"S_{ref_p}=%{{x:.3g}}<br>S_{p}=%{{y:.3g}}<extra></extra>",
            )
            if show_error_bars:
                error_dict = {}
                if x_std is not None:
                    error_dict["error_x"] = dict(type="data", array=x_std, visible=True, thickness=1, color=color)
                if y_std is not None:
                    error_dict["error_y"] = dict(type="data", array=y_std, visible=True, thickness=1, color=color)
                if error_dict:
                    trace_kwargs.update(error_dict)
            fig_ess.add_trace(go.Scatter(**trace_kwargs))

            if show_std_band and y_std is not None:
                rgb = color_to_rgb_tuple(color)
                fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps_ess['std_alpha']})"
                fig_ess.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([y - y_std, (y + y_std)[::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            if np.count_nonzero(rmask) >= 3:
                logx = np.log(x[rmask])
                logy = np.log(y[rmask])
                valid = np.isfinite(logx) & np.isfinite(logy)
                if np.count_nonzero(valid) >= 3:
                    slope, intercept = np.polyfit(logx[valid], logy[valid], 1)
                    yfit = slope * logx[valid] + intercept
                    resid = logy[valid] - yfit
                    dof = max(len(resid) - 2, 1)
                    stderr = np.sqrt(np.sum(resid**2) / dof) / np.sqrt(len(resid))
                    xi_all[sim_prefix][p] = float(slope)
                    xi_err_all[sim_prefix][p] = float(stderr)
                    anom_all[sim_prefix][p] = float(slope - p / 3)

    if not plotted_any:
        st.info("No valid ESS data to plot.")
        return False

    layout_kwargs = dict(
        xaxis_title=labels.get("x_ess", "S<sub>3</sub>(r)"),
        yaxis_title=labels.get("y_ess", "S<sub>p</sub>(r)"),
        legend_title="Simulation / Order",
        height=500,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps_ess)
    layout_kwargs = apply_figure_size(layout_kwargs, ps_ess)
    fig_ess.update_layout(**layout_kwargs)
    fig_ess = apply_plot_style(fig_ess, ps_ess)

    if show_inset:
        plot_name_inset = "ESS Inset"
        ps_inset = get_plot_style(plot_name_inset)
        if not st.session_state.get("plot_styles", {}).get(plot_name_inset):
            for key, value in ps_ess.items():
                if key not in ps_inset or ps_inset[key] == default_plot_style().get(key):
                    ps_inset[key] = value
        fig_ess = add_ess_inset(
            fig=fig_ess,
            xi_all=xi_all,
            anom_all=anom_all,
            xi_err_all=xi_err_all,
            sim_groups=sim_groups,
            legend_names=legends,
            colors_palette=colors_ess,
            plot_style=ps_inset,
            show_sl_theory=show_sl_theory,
            show_exp_anom=show_exp_anom,
            inset_x_label=labels.get("x_inset", "p"),
            inset_y_label=labels.get("y_inset", "ξ<sub>p</sub> - p/3"),
            inset_legend_sl=labels.get("inset_legend_sl", "SL94"),
            inset_legend_b93=labels.get("inset_legend_b93", "B93"),
        )

    st.plotly_chart(fig_ess, width="stretch")
    capture_button(fig_ess, title="Structure Functions ESS", source_page="Structure Functions")
    export_panel(fig_ess, data_dir, base_name="structure_functions_ess")

    st.markdown("#### Anomalies (ξₚ − p/3)")
    plot_name_anom = "Anomalies (ξₚ − p/3)"
    ps_anom = get_plot_style(plot_name_anom)
    colors_anom = _get_palette(ps_anom)
    fig_anom = go.Figure()

    for idx, sim_prefix in enumerate(sorted(xi_all.keys())):
        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix, idx, colors_anom, ps_anom,
            style_key="per_sim_style_structure",
            include_marker=True,
            default_marker="circle",
        )
        ps_show = sorted(xi_all[sim_prefix].keys())
        yvals = [anom_all[sim_prefix][p] for p in ps_show]
        yerr = [xi_err_all[sim_prefix].get(p, 0.0) for p in ps_show]
        fig_anom.add_trace(
            go.Scatter(
                x=ps_show,
                y=yvals,
                mode="lines+markers",
                name=legends.get(sim_prefix, sim_prefix.replace("_", " ").title()),
                line=dict(color=color, width=max(1.0, lw * 0.7)),
                marker=dict(symbol=marker, size=max(4, int(msize * 0.7))),
                error_y=dict(type="data", array=yerr, visible=True, thickness=1),
            )
        )

    if show_sl_theory:
        ps_theory = list(range(1, max(selected_ps) + 1))
        theory_anom = [zeta_p_she_leveque(p) - p / 3 for p in ps_theory]
        sl_color = ps_anom.get("she_leveque_color", "#000000")
        fig_anom.add_trace(
            go.Scatter(
                x=ps_theory,
                y=theory_anom,
                mode="lines+markers",
                name="She–Leveque 1994",
                line=dict(color=sl_color, dash="dash", width=1.5),
                marker=dict(symbol="diamond", size=5),
            )
        )

    if show_exp_anom:
        exp_anom = [EXP_ZETA[i] - TABLE_P[i] / 3 for i in range(len(TABLE_P))]
        exp_color = ps_anom.get("experimental_b93_color", "#00BFC4")
        fig_anom.add_trace(
            go.Scatter(
                x=TABLE_P,
                y=exp_anom,
                mode="lines+markers",
                name="Experiment (B93)",
                line=dict(color=exp_color, width=1.5),
                marker=dict(symbol="x", size=6),
            )
        )

    fig_anom.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)
    layout_kwargs_anom = dict(
        xaxis_title=labels.get("x_anom", "p"),
        yaxis_title=labels.get("y_anom", "ξ<sub>p</sub> - p/3"),
        height=360,
        legend_title="",
    )
    layout_kwargs_anom = apply_axis_limits(layout_kwargs_anom, ps_anom)
    layout_kwargs_anom = apply_figure_size(layout_kwargs_anom, ps_anom)
    fig_anom.update_layout(**layout_kwargs_anom)
    fig_anom = apply_plot_style(fig_anom, ps_anom)
    st.plotly_chart(fig_anom, width="stretch")
    export_panel(fig_anom, data_dir, base_name="structure_functions_anomalies")

    st.session_state["_xi_all"] = xi_all
    st.session_state["_anom_all"] = anom_all
    st.session_state["_xi_err_all"] = xi_err_all
    return True


def render_table_tab():
    """Render scaling exponents table tab (reads from session state)."""
    st.subheader("Computed ESS Scaling Exponents")
    xi_all = st.session_state.get("_xi_all", {})
    xi_err_all = st.session_state.get("_xi_err_all", {})
    legends = st.session_state.structure_legend_names

    if not xi_all:
        st.info("Run ESS tab first to populate exponents.")
        return

    all_simulations = sorted(xi_all.keys())
    if len(all_simulations) > 1:
        selected_sims = st.multiselect(
            "Select simulations to display:",
            options=all_simulations,
            default=all_simulations,
            key="table_sim_selector",
        )
    else:
        selected_sims = all_simulations

    if not selected_sims:
        st.info("Please select at least one simulation to display.")
        return

    rows = []
    for sim_prefix in selected_sims:
        if sim_prefix not in xi_all:
            continue
        for p, xi in xi_all[sim_prefix].items():
            rows.append({
                "simulation": legends.get(sim_prefix, sim_prefix),
                "p": p,
                "xi_p": f"{xi:.6f}",
                "stderr": f"{xi_err_all.get(sim_prefix, {}).get(p, np.nan):.6f}",
                "xi_p - p/3": f"{xi - p/3:.6f}",
                "She–Leveque ζ_p": f"{zeta_p_she_leveque(p):.6f}",
                "xi_p - ζ_p": f"{xi - zeta_p_she_leveque(p):.6f}",
            })

    if rows:
        df = pd.DataFrame(rows).sort_values(["simulation", "p"])
        st.dataframe(df, width="stretch", hide_index=True, height=min(400, 50 + len(df) * 35))
        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="ess_scaling_exponents.csv",
                mime="text/csv",
                key="download_ess_table",
            )
        with col2:
            st.caption(f"Showing {len(selected_sims)} simulation(s) with {len(df)} total rows")
    else:
        st.warning("No data available for selected simulations.")


def render_theory_section():
    """Render theory & equations expander."""
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("**Structure functions:**")
        st.latex(r"""
        S_p(r) = \langle |\delta u_L(r)|^p \rangle
        """)
        st.markdown(r"""
        where $\delta u_L(r) = u_L(\mathbf{x} + r\mathbf{e}_L) - u_L(\mathbf{x})$ is the longitudinal velocity increment.
        """)
        st.markdown("**Extended Self-Similarity (ESS):** ([Benzi et al., 1993](/Citation#benzi1993))")
        st.latex(r"""
        S_p(r) \propto S_3(r)^{\xi_p}
        """)
        st.markdown(r"""
        The scaling exponent $\xi_p$ is obtained from the slope of $\log S_p$ vs $\log S_3$.
        """)
        st.markdown("**She–Leveque 1994 scaling (theoretical):** ([She & Leveque, 1994](/Citation#she1994))")
        st.latex(r"""
        \zeta_p = \frac{p}{9} + 2\left(1 - \left(\frac{2}{3}\right)^{p/3}\right)
        """)
        st.markdown(r"""
        Anomalies are plotted as $\xi_p - p/3$ to compare with theoretical predictions.
        """)
