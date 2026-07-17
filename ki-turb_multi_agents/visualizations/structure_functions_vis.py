"""
Shared structure functions visualization for S_p(r), ESS, anomalies.

Used by:
  1. Manual page (08_Structure_Functions.py) — via views
  2. AI agents (plot_structure_functions tool)

Pure Python plotting logic — no Streamlit dependency.
Supports: S_p(r) vs r, ESS (S_p vs S_ref), anomalies (ξₚ − p/3), ESS inset.
"""

import numpy as np
import plotly.graph_objects as go

from utils.plot_style import (
    apply_axis_limits,
    apply_figure_size,
    apply_plot_style as apply_plot_style_base,
    resolve_line_style,
    _get_palette,
    color_to_rgb,
)


def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()


def _compute_ess_fit(
    sim_items: list,
    ref_p: int,
    selected_ps: list,
    fit_rmin: float,
    fit_rmax: float,
    normalize_by_urms: bool,
) -> tuple:
    """
    Compute ESS scaling exponents from cached structure function data.
    Returns (xi_all, anom_all, xi_err_all) dicts: {sim_prefix: {p: value}}.
    """
    xi_all = {}
    anom_all = {}
    xi_err_all = {}

    for sim_prefix, d in sim_items:
        r = np.asarray(d["r"], dtype=float)
        Sp_mean = {int(p): np.asarray(arr, dtype=float) for p, arr in d["Sp_mean"].items()}
        urms = float(d.get("urms", 0.0))

        if ref_p not in Sp_mean:
            continue

        def _norm(p, arr):
            if normalize_by_urms and np.isfinite(urms) and urms > 0:
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
            rmask = (
                (r >= fit_rmin) & (r <= fit_rmax)
                & np.isfinite(x) & (x > 0)
                & np.isfinite(y) & (y > 0)
            )
            if np.count_nonzero(rmask) < 3:
                continue
            logx = np.log(x[rmask])
            logy = np.log(y[rmask])
            valid = np.isfinite(logx) & np.isfinite(logy)
            if np.count_nonzero(valid) < 3:
                continue
            slope, intercept = np.polyfit(logx[valid], logy[valid], 1)
            yfit = slope * logx[valid] + intercept
            resid = logy[valid] - yfit
            dof = max(len(resid) - 2, 1)
            stderr = np.sqrt(np.sum(resid**2) / dof) / np.sqrt(len(resid))
            xi_all[sim_prefix][p] = float(slope)
            xi_err_all[sim_prefix][p] = float(stderr)
            anom_all[sim_prefix][p] = float(slope - p / 3)

    return xi_all, anom_all, xi_err_all


def create_sp_figure(
    datasets: list,
    ps: dict,
    *,
    selected_ps: list = None,
    normalize_by_urms: bool = True,
    show_std: bool = True,
    show_error_bars: bool = False,
    axis_labels: dict = None,
    legend_names: dict = None,
    apply_style: bool = True,
):
    """
    Create S_p(r) vs r figure.

    datasets: List of {sim_prefix, r, Sp_mean, Sp_std, urms, ps}
    selected_ps: Orders p to plot (default: all in first dataset)
    """
    axis_labels = axis_labels or {"x": "Separation distance r", "y": "Structure functions S<sub>p</sub>(r)"}
    legend_names = legend_names or {}
    colors = _get_palette(ps)

    if not datasets:
        return None

    if selected_ps is None:
        selected_ps = sorted(datasets[0].get("ps", [1, 2, 3, 4, 5, 6]))

    fig = go.Figure()
    for idx, d in enumerate(datasets):
        sim_prefix = d.get("sim_prefix", f"sim_{idx}")
        r = np.asarray(d["r"], dtype=float)
        Sp_mean = d["Sp_mean"]
        Sp_std = d.get("Sp_std", {})
        urms = float(d.get("urms", 0.0))

        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix, idx, colors, ps,
            style_key="per_sim_style_structure",
            include_marker=True,
            default_marker="circle",
        )
        label_base = legend_names.get(sim_prefix, _default_labelify(sim_prefix))
        mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
        marker_dict = dict(symbol=marker, size=msize) if (override_on and marker and msize > 0) else None

        for p in selected_ps:
            if p not in Sp_mean:
                continue
            y = np.asarray(Sp_mean[p], dtype=float).copy()
            ystd = np.asarray(Sp_std.get(p), dtype=float) if p in Sp_std else None
            if normalize_by_urms and np.isfinite(urms) and urms > 0:
                y = y / (urms ** p)
                if ystd is not None:
                    ystd = ystd / (urms ** p)

            trace_kwargs = dict(
                x=r, y=y, mode=mode,
                name=f"{label_base}  (p={p})",
                line=dict(color=color, width=lw, dash=dash),
                hovertemplate="r=%{x:.3g}<br>S_p=%{y:.3g}<extra></extra>",
            )
            if marker_dict:
                trace_kwargs["marker"] = marker_dict
            if show_error_bars and ystd is not None:
                trace_kwargs["error_y"] = dict(type="data", array=ystd, visible=True, thickness=1, color=color)
            fig.add_trace(go.Scatter(**trace_kwargs))

            if show_std and ystd is not None:
                rgb = color_to_rgb(color)
                fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps.get('std_alpha', 0.18)})"
                fig.add_trace(go.Scatter(
                    x=np.concatenate([r, r[::-1]]),
                    y=np.concatenate([y - ystd, (y + ystd)[::-1]]),
                    fill="toself", fillcolor=fill_rgba,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "Separation distance r"),
        yaxis_title=axis_labels.get("y", "Structure functions S<sub>p</sub>(r)"),
        legend_title="Simulation / Order",
        height=500,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)
    if apply_style:
        fig = apply_plot_style_base(fig, ps)
    return fig


def create_ess_figure(
    datasets: list,
    ps: dict,
    *,
    ref_p: int = 3,
    selected_ps: list = None,
    normalize_by_urms: bool = True,
    show_std: bool = True,
    show_error_bars: bool = False,
    fit_rmin: float = None,
    fit_rmax: float = None,
    show_inset: bool = True,
    show_sl_theory: bool = True,
    show_exp_anom: bool = True,
    axis_labels: dict = None,
    legend_names: dict = None,
    apply_style: bool = True,
):
    """
    Create ESS (S_p vs S_ref) figure, optionally with anomalies inset.
    """
    from core_physics import zeta_p_she_leveque, TABLE_P, EXP_ZETA
    from pages.StructureFunctions.ess_inset import add_ess_inset

    axis_labels = axis_labels or {"x_ess": "S<sub>3</sub>(r)", "y_ess": "S<sub>p</sub>(r)"}
    legend_names = legend_names or {}
    colors = _get_palette(ps)

    if not datasets:
        return None

    sim_items = [(d.get("sim_prefix", f"sim_{i}"), d) for i, d in enumerate(datasets)]
    if selected_ps is None:
        selected_ps = sorted(datasets[0].get("ps", [1, 2, 3, 4, 5, 6]))
    if ref_p not in selected_ps and ref_p in (datasets[0].get("ps") or []):
        pass
    elif ref_p not in (datasets[0].get("ps") or []):
        ref_p = (datasets[0].get("ps") or [3])[0]

    r = np.asarray(datasets[0]["r"], dtype=float)
    if fit_rmin is None:
        r_pos = r[r > 0]
        fit_rmin = float(np.percentile(r_pos, 10)) if len(r_pos) > 0 else 1e-3
    if fit_rmax is None:
        r_pos = r[r > 0]
        fit_rmax = float(np.percentile(r_pos, 60)) if len(r_pos) > 0 else 1e-1

    xi_all, anom_all, xi_err_all = _compute_ess_fit(
        sim_items, ref_p, selected_ps, fit_rmin, fit_rmax, normalize_by_urms
    )

    fig = go.Figure()
    for idx, (sim_prefix, d) in enumerate(sim_items):
        Sp_mean = {int(p): np.asarray(arr, dtype=float) for p, arr in d["Sp_mean"].items()}
        Sp_std = {int(p): np.asarray(arr, dtype=float) for p, arr in d.get("Sp_std", {}).items()}
        urms = float(d.get("urms", 0.0))

        if ref_p not in Sp_mean:
            continue

        def _norm(p, arr):
            if normalize_by_urms and np.isfinite(urms) and urms > 0:
                return arr / (urms ** p)
            return arr

        color, lw, dash, marker, msize, _ = resolve_line_style(
            sim_prefix, idx, colors, ps,
            style_key="per_sim_style_structure",
            include_marker=True,
            default_marker="circle",
        )
        label_base = legend_names.get(sim_prefix, _default_labelify(sim_prefix))

        for p in selected_ps:
            if p not in Sp_mean:
                continue
            x = _norm(ref_p, Sp_mean[ref_p])
            y = _norm(p, Sp_mean[p])
            x_std = _norm(ref_p, Sp_std[ref_p]) if ref_p in Sp_std else None
            y_std = _norm(p, Sp_std[p]) if p in Sp_std else None

            trace_kwargs = dict(
                x=x, y=y, mode="lines+markers",
                name=f"{label_base} (p={p})",
                line=dict(color=color, width=lw, dash=dash),
                marker=dict(symbol=marker, size=msize),
                hovertemplate=f"S_{ref_p}=%{{x:.3g}}<br>S_{p}=%{{y:.3g}}<extra></extra>",
            )
            if show_error_bars:
                if x_std is not None:
                    trace_kwargs["error_x"] = dict(type="data", array=x_std, visible=True, thickness=1, color=color)
                if y_std is not None:
                    trace_kwargs["error_y"] = dict(type="data", array=y_std, visible=True, thickness=1, color=color)
            fig.add_trace(go.Scatter(**trace_kwargs))

            if show_std and y_std is not None:
                rgb = color_to_rgb(color)
                fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps.get('std_alpha', 0.18)})"
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([y - y_std, (y + y_std)[::-1]]),
                    fill="toself", fillcolor=fill_rgba,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x_ess", "S<sub>3</sub>(r)"),
        yaxis_title=axis_labels.get("y_ess", "S<sub>p</sub>(r)"),
        legend_title="Simulation / Order",
        height=500,
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)
    if apply_style:
        fig = apply_plot_style_base(fig, ps)

    if show_inset and xi_all:
        sim_groups = {k: {} for k in xi_all.keys()}
        ps_inset = dict(ps)
        fig = add_ess_inset(
            fig=fig,
            xi_all=xi_all,
            anom_all=anom_all,
            xi_err_all=xi_err_all,
            sim_groups=sim_groups,
            legend_names=legend_names,
            colors_palette=colors,
            plot_style=ps_inset,
            show_sl_theory=show_sl_theory,
            show_exp_anom=show_exp_anom,
            inset_x_label=axis_labels.get("x_inset", "p"),
            inset_y_label=axis_labels.get("y_inset", "ξ<sub>p</sub> - p/3"),
            inset_legend_sl=axis_labels.get("inset_legend_sl", "SL94"),
            inset_legend_b93=axis_labels.get("inset_legend_b93", "B93"),
        )

    return fig


def create_anomalies_figure(
    datasets: list,
    ps: dict,
    *,
    ref_p: int = 3,
    selected_ps: list = None,
    normalize_by_urms: bool = True,
    fit_rmin: float = None,
    fit_rmax: float = None,
    show_sl_theory: bool = True,
    show_exp_anom: bool = True,
    axis_labels: dict = None,
    legend_names: dict = None,
    apply_style: bool = True,
):
    """Create anomalies (ξₚ − p/3) figure vs order p. Linear axes, data-derived y-range."""
    from core_physics import zeta_p_she_leveque, TABLE_P, EXP_ZETA

    axis_labels = axis_labels or {"x": "p", "y": "ξ<sub>p</sub> - p/3"}
    legend_names = legend_names or {}
    colors = _get_palette(ps)

    if not datasets:
        return None

    sim_items = [(d.get("sim_prefix", f"sim_{i}"), d) for i, d in enumerate(datasets)]
    if selected_ps is None:
        selected_ps = sorted(datasets[0].get("ps", [1, 2, 3, 4, 5, 6]))
    if ref_p not in (datasets[0].get("ps") or []):
        ref_p = (datasets[0].get("ps") or [3])[0]

    r = np.asarray(datasets[0]["r"], dtype=float)
    if fit_rmin is None:
        r_pos = r[r > 0]
        fit_rmin = float(np.percentile(r_pos, 10)) if len(r_pos) > 0 else 1e-3
    if fit_rmax is None:
        r_pos = r[r > 0]
        fit_rmax = float(np.percentile(r_pos, 60)) if len(r_pos) > 0 else 1e-1

    xi_all, anom_all, xi_err_all = _compute_ess_fit(
        sim_items, ref_p, selected_ps, fit_rmin, fit_rmax, normalize_by_urms
    )

    if not xi_all:
        return None

    fig = go.Figure()
    for idx, sim_prefix in enumerate(sorted(xi_all.keys())):
        if not xi_all[sim_prefix]:
            continue
        color, lw, dash, marker, msize, _ = resolve_line_style(
            sim_prefix, idx, colors, ps,
            style_key="per_sim_style_structure",
            include_marker=True,
            default_marker="circle",
        )
        ps_show = sorted(xi_all[sim_prefix].keys())
        yvals = [anom_all[sim_prefix][p] for p in ps_show]
        yerr = [xi_err_all.get(sim_prefix, {}).get(p, 0.0) for p in ps_show]
        fig.add_trace(go.Scatter(
            x=ps_show, y=yvals, mode="lines+markers",
            name=legend_names.get(sim_prefix, _default_labelify(sim_prefix)),
            line=dict(color=color, width=max(1.0, lw * 0.7)),
            marker=dict(symbol=marker, size=max(4, int(msize * 0.7))),
            error_y=dict(type="data", array=yerr, visible=True, thickness=1),
        ))

    if show_sl_theory:
        ps_theory = list(range(1, max(selected_ps) + 1))
        theory_anom = [zeta_p_she_leveque(p) - p / 3 for p in ps_theory]
        sl_color = ps.get("she_leveque_color", "#000000")
        fig.add_trace(go.Scatter(
            x=ps_theory, y=theory_anom, mode="lines+markers",
            name="She–Leveque 1994",
            line=dict(color=sl_color, dash="dash", width=1.5),
            marker=dict(symbol="diamond", size=5),
        ))

    if show_exp_anom:
        exp_anom = [EXP_ZETA[i] - TABLE_P[i] / 3 for i in range(len(TABLE_P))]
        exp_color = ps.get("experimental_b93_color", "#00BFC4")
        fig.add_trace(go.Scatter(
            x=TABLE_P, y=exp_anom, mode="lines+markers",
            name="Experiment (B93)",
            line=dict(color=exp_color, width=1.5),
            marker=dict(symbol="x", size=6),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)

    # Data-derived y-range to avoid auto-scale extremes
    all_anom_vals = []
    for k in anom_all:
        if anom_all[k]:
            all_anom_vals.extend(anom_all[k].values())
    if all_anom_vals:
        y_range = max(all_anom_vals) - min(all_anom_vals)
        pad = 0.15 * y_range if y_range > 0 else 0.2
        data_y_min = min(all_anom_vals) - pad
        data_y_max = max(all_anom_vals) + pad
    else:
        data_y_min, data_y_max = -0.2, 0.2

    layout_kwargs = dict(
        xaxis_title=axis_labels.get("x", "p"),
        yaxis_title=axis_labels.get("y", "ξ<sub>p</sub> - p/3"),
        height=360,
        legend_title="",
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    # Use data-derived range unless user has set manual y-axis limits via style config.
    if "yaxis_range" not in layout_kwargs:
        layout_kwargs["yaxis_range"] = [data_y_min, data_y_max]
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    fig.update_layout(**layout_kwargs)
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

for _kiturb_name in ('create_sp_figure', 'create_ess_figure', 'create_anomalies_figure',):
    if _kiturb_name in globals():
        globals()[_kiturb_name] = _kiturb_hit_provenance_wrapper(globals()[_kiturb_name])
