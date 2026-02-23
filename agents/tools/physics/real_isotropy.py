"""
Real-space isotropy agent tools: energy fractions, Lumley triangle, diagonal b_ii, etc.

Per-subplot style isolation (matches page "Select plot to configure"):
  Each tool uses its OWN session style config — never inherits from other subplots.
  A: real_isotropy_style_config   B: lumley_style_config   C: diagonal_bii_style_config
  When adding D, E, F: use dedicated cross_corr_style_config, deviations_style_config, convergence_style_config.
  style_updates and axis_labels apply ONLY to the subplot that tool produces.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core_physics import (
    load_turbulence_data,
    load_reynolds_stress,
    compute_reynolds_from_fractions,
    anisotropy_tensor,
    invariants,
)

from .._shared import get_from_cache, resolve_data_dir_and_find_files, save_to_cache
from ._meta import get_artifact_source_meta
from pages.AutonomousLab.session_sync import update_data_directory_in_context

from content.real_isotropy_theory_content import get_real_isotropy_theory_markdown


def _write_back_data_dir(session_context: Dict[str, Any], csv_path: Path) -> None:
    """Write data directory to session_context so manual pages sync after agent run."""
    update_data_directory_in_context(session_context, csv_path.parent)


CACHE_KEY_REAL_ISOTROPY = "current_real_isotropy_data"
PATTERN_EPS_REAL = "eps_real_validation*.csv"
PATTERN_TURB_VALIDATION = "turbulence_validation*.csv"


def _resolve_eps_csv(
    csv_path: str,
    data_dir: str,
    project_root: Path,
    session_context: Dict[str, Any],
) -> Optional[Path]:
    """Resolve path to eps_real_validation*.csv or turbulence_validation*.csv (LBM/NS)."""
    if csv_path:
        p = Path(csv_path)
        if not p.is_absolute():
            p = (project_root / csv_path).resolve()
        if p.exists() and p.is_file():
            return p
    for pattern in (PATTERN_EPS_REAL, PATTERN_TURB_VALIDATION):
        files = resolve_data_dir_and_find_files(
            data_dir or csv_path or "",
            pattern,
            project_root,
            session_context,
            max_files=1,
        )
        if files:
            return Path(files[0])
    return None


def _compute_time_axis_and_stationary(
    turb: dict,
    normalize_x: bool = True,
    x_norm: Optional[float] = None,
    stationary_iter: Optional[float] = None,
    stationary_t: Optional[float] = None,
) -> tuple:
    """
    Compute iter_norm and stationary_line_x from page-style Analysis Controls.
    Matches page logic: normalize_x, x_norm, stationary_iter.
    """
    t0_raw = float(turb["iter"][0]) if turb["iter"][0] != 0 else 1.0
    x_norm_val = float(x_norm) if x_norm is not None else t0_raw
    if normalize_x:
        iter_norm = np.asarray(turb["iter"], dtype=float) / x_norm_val
    else:
        iter_norm = np.asarray(turb["iter"], dtype=float)
    stationary_line_x = None
    if stationary_iter is not None:
        if normalize_x:
            stationary_line_x = float(stationary_iter) / x_norm_val
        else:
            stationary_line_x = float(stationary_iter)
    elif stationary_t is not None:
        stationary_line_x = float(stationary_t)
    return iter_norm, stationary_line_x


def _resolve_stress_csv(eps_path: Path) -> Optional[Path]:
    """Resolve reynolds_stress_validation*.csv from eps file path (matches page logic)."""
    import re
    import glob
    data_dir = eps_path.parent
    eps_name = eps_path.name
    stress_file = None
    if "_data" in eps_name:
        tag_match = re.search(r"_data\d+", eps_name)
        if tag_match:
            tag = tag_match.group(0)
            candidate = data_dir / f"reynolds_stress_validation{tag}.csv"
            if candidate.exists():
                stress_file = candidate
    if stress_file is None:
        candidate = data_dir / "reynolds_stress_validation.csv"
        if candidate.exists():
            stress_file = candidate
    if stress_file is None:
        matches = glob.glob(str(data_dir / "reynolds_stress_validation*.csv"))
        if matches:
            stress_file = Path(matches[0])
    return stress_file


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for real-space isotropy."""
    return [
        {
            "name": "compute_isotropy",
            "description": "Compute real-space isotropy score from eps_real_validation.csv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                },
            },
        },
        {
            "name": "plot_real_isotropy",
            "description": "Plot real-space isotropy: energy fractions (frac_x, frac_y, frac_z). Use when user asks for 'real isotropy', 'isotropy form', 'plot isotropy' from eps_real_validation data. Same style API as plot_spectrum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation*.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API: font_family, font_size, title_size, legend_size, tick_font_size, axis_title_size, font_color. Backgrounds: plot_bgcolor, paper_bgcolor. Ticks: tick_len, tick_w, ticks_outside, tick_color. Axis: x_axis_type, y_axis_type, x_tick_format, y_tick_format. Borders: show_axis_lines, axis_line_width, axis_line_color, mirror_axes. Grid: show_grid, grid_on_x, grid_on_y, grid_w, grid_dash, grid_color, grid_opacity. Minor grid: show_minor_grid, minor_grid_*. Curves: line_width, marker_size. Colors: palette, custom_colors. Theme: template. Legend: show_legend. Title: show_plot_title, plot_title. Limits: enable_x_limits, x_min, x_max, enable_y_limits, y_min, y_max. Size: enable_custom_size, figure_width, figure_height. Margins: margin_left, margin_top, margin_right, margin_bottom. Per-curve overrides: enable_per_curve_style=true, per_curve_style_Energy_Fractions_A: {Ex: {enabled:true, color:\"#ff0000\", width:2, dash:\"dash\"}, Ey: {...}, Ez: {...}}.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"t/t0\", \"y\": \"Energy fraction\"}. Partial OK."},
                    "legend_names": {"type": "object", "description": "Override curve names: {\"frac_x\": \"E<sub>x</sub>/E<sub>tot</sub>\", \"frac_y\": \"...\", \"frac_z\": \"...\"}. Matches page Legend & Axis Labels. Partial OK."},
                    "ma_win": {"type": "integer", "description": "Moving average window for energy fractions (0 or 1 = off). When > 1, adds smoothed MA traces over raw data."},
                    "tol_list": {"type": "array", "items": {"type": "number"}, "description": "Tolerance bands around isotropic 1/3, e.g. [0.005, 0.01, 0.02] for ±0.5%, ±1%, ±2%."},
                    "normalize_x": {"type": "boolean", "description": "Normalize X-axis (t/t₀). Default true. When false, use raw iteration numbers."},
                    "x_norm": {"type": "number", "description": "X normalization constant (default: first iteration). Used when normalize_x=true."},
                    "stationary_iter": {"type": "number", "description": "Stationarity iteration (raw). Vertical line at this iteration. Matches page 'Stationarity iteration'."},
                    "stationary_t": {"type": "number", "description": "Normalized time (t/t0) for statistical stationarity vertical line. Alternative to stationary_iter."},
                    "raw_data_opacity": {"type": "number", "description": "Opacity for raw data traces (0–1). Default 0.5."},
                },
            },
        },
        {
            "name": "plot_lumley_triangle",
            "description": "Plot Lumley triangle (ξ, η) trajectory from eps_real_validation data. Shows realizability boundaries, DNS trajectory with time-coloring, start/end points. Use when user asks for 'Lumley triangle', 'lumely triangle', 'subplot B', 'Lumley subplot', 'xi eta', 'realizability'. NOT for energy fractions—use plot_real_isotropy for that. Same style API as plot_spectrum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation*.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API (same as plot_spectrum): font_family, font_size, title_size, legend_size, tick_font_size, axis_title_size, font_color. Backgrounds: plot_bgcolor, paper_bgcolor. Ticks: tick_len, tick_w, ticks_outside, tick_color. Axis: x_axis_type, y_axis_type, x_tick_format, y_tick_format. Borders: show_axis_lines, axis_line_width, axis_line_color, mirror_axes. Grid: show_grid, grid_on_x, grid_on_y, grid_w, grid_dash, grid_color, grid_opacity. Curves: line_width, marker_size. Colors: palette, custom_colors. Theme: template. Legend: show_legend. Title: show_plot_title, plot_title. Limits: enable_x_limits, x_min, x_max, enable_y_limits, y_min, y_max. Size: enable_custom_size, figure_width, figure_height. Margins: margin_left, margin_top, margin_right, margin_bottom.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"ξ\", \"y\": \"η\"}. Partial OK."},
                },
            },
        },
        {
            "name": "plot_diagonal_bii",
            "description": "Plot diagonal anisotropy tensor b11, b22, b33 vs t/t0 from eps_real_validation data. Use when user asks for 'diagonal b_ii', 'subplot C', 'b11 b22 b33', 'anisotropy diagonal'. Uses LINEAR axes (b_ii can be negative). For non-default colors: pass palette='Dark2' or palette='Set1' (top-level) OR style_updates.palette or style_updates.custom_colors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation*.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                    "palette": {"type": "string", "description": "Color palette name for curves (e.g. 'Dark2','Set1','Bold'). Use when user asks for 'different colors' or 'non-default colors'."},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API. For custom colors: palette or custom_colors. Per-curve: enable_per_curve_style, per_curve_style_Diagonal_b_ii_C: {b11: {enabled:true, color, width, dash}, b22: {...}, b33: {...}}. Axis scale always linear for b_ii.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"t/t₀\", \"y\": \"Anisotropy tensor b<sub>ij</sub>\"}. Partial OK. Default matches page subplot C."},
                    "legend_names": {"type": "object", "description": "Override curve names: {\"b11\": \"b<sub>11</sub>\", \"b22\": \"...\", \"b33\": \"...\"}. Matches page Legend & Axis Labels. Partial OK."},
                    "tol_list": {"type": "array", "items": {"type": "number"}, "description": "Tolerance bands around isotropic 0, e.g. [0.005, 0.01, 0.02] for ±0.5%, ±1%, ±2%."},
                    "normalize_x": {"type": "boolean", "description": "Normalize X-axis (t/t₀). Default true."},
                    "x_norm": {"type": "number", "description": "X normalization constant (default: first iteration). Matches page Analysis Controls."},
                },
            },
        },
        {
            "name": "plot_cross_correlations",
            "description": "Plot cross-correlations |b12|, |b13|, |b23| and anisotropy index vs t/t0 from eps_real_validation data. Use when user asks for 'subplot D', 'cross-correlations', 'b12 b13 b23', 'anisotropy index'. tol_list [0.001, 0.005, 0.01] for tolerance lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation*.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                    "palette": {"type": "string", "description": "Color palette name for curves (e.g. 'Dark2','Set1','Bold')."},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API. Per-curve: enable_per_curve_style, per_curve_style_Cross_correlations_D: {b12, b13, b23, anis: {enabled, color, width, dash}}.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"t/t₀\", \"y\": \"Cross-correlations / Anisotropy index\"}. Partial OK."},
                    "legend_names": {"type": "object", "description": "Override curve names: {\"b12\": \"|b<sub>12</sub>|\", \"b13\": \"...\", \"b23\": \"...\", \"anis\": \"Anisotropy index\"}. Matches page Legend & Axis Labels. Partial OK."},
                    "tol_list": {"type": "array", "items": {"type": "number"}, "description": "Tolerance lines at y values, e.g. [0.001, 0.005, 0.01]."},
                    "normalize_x": {"type": "boolean", "description": "Normalize X-axis (t/t₀). Default true."},
                    "x_norm": {"type": "number", "description": "X normalization constant (default: first iteration). Matches page Analysis Controls."},
                },
            },
        },
        {
            "name": "plot_deviations",
            "description": "Plot energy-fraction deviations |E_x−1/3|, |E_y−1/3|, |E_z−1/3|, max dev vs t/t0 from eps_real_validation data. Use when user asks for 'subplot E', 'deviations', 'energy fraction deviations'. tol_list [0.005, 0.01, 0.02], stationary_t for statistical stationarity vertical line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation*.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                    "palette": {"type": "string", "description": "Color palette name for curves (e.g. 'Dark2','Set1','Bold')."},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API. Per-curve: enable_per_curve_style, per_curve_style_Deviations_E: {devx, devy, devz, maxdev: {enabled, color, width, dash}}.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"t/t₀\", \"y\": \"Absolute deviation\"}. Partial OK."},
                    "legend_names": {"type": "object", "description": "Override curve names: {\"devx\": \"...\", \"devy\": \"...\", \"devz\": \"...\", \"maxdev\": \"Max deviation\"}. Matches page Legend & Axis Labels. Partial OK."},
                    "tol_list": {"type": "array", "items": {"type": "number"}, "description": "Tolerance lines at y values, e.g. [0.005, 0.01, 0.02]."},
                    "normalize_x": {"type": "boolean", "description": "Normalize X-axis (t/t₀). Default true."},
                    "x_norm": {"type": "number", "description": "X normalization constant (default: first iteration). Matches page Analysis Controls."},
                    "stationary_iter": {"type": "number", "description": "Stationarity iteration (raw). Vertical line at this iteration. Matches page 'Stationarity iteration'."},
                    "stationary_t": {"type": "number", "description": "Normalized time (t/t0) for statistical stationarity vertical line. Alternative to stationary_iter."},
                },
            },
        },
        {
            "name": "get_real_isotropy_summary",
            "description": "Show real isotropy summary table (Final Ex, Final Ey, Final Ez, Final anisotropy index, Mean anisotropy index). Use when user asks for 'summary', 'table', or 'statistics' of real isotropy. Data from eps_real_validation*.csv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation*.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                },
            },
        },
        {
            "name": "get_real_isotropy_theory",
            "description": "Return Theory & Equations for real-space isotropy. Use subplot to show only equations for that subplot: A=Energy Fractions, B=Lumley, C=Diagonal b_ii, D=Cross-correlations, E=Deviations, F=Convergence. Omit subplot for full theory. Use when user asks for 'theory', 'equations', or 'formulas' for real isotropy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subplot": {"type": "string", "description": "Filter equations to subplot: A (energy fractions), B (Lumley), C (diagonal b_ii), D (cross-correlations), E (deviations), F (convergence). Omit for full theory."},
                },
            },
        },
        {
            "name": "plot_convergence",
            "description": "Plot running std of E_x, E_y, E_z vs t/t0 from eps_real_validation data. Use when user asks for 'subplot F', 'convergence', 'running std'. conv_windows derived from data length by default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to eps_real_validation*.csv (or directory containing it)"},
                    "data_dir": {"type": "string", "description": "Directory path; finds eps_real_validation*.csv if csv_path not given"},
                    "palette": {"type": "string", "description": "Color palette name (e.g. 'Dark2','Set1')."},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API: font_family, font_size, line_width, plot_bgcolor, grid, limits, figure_width, figure_height.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"t/t₀\", \"y\": \"Running standard deviation\"}. Partial OK."},
                    "normalize_x": {"type": "boolean", "description": "Normalize X-axis (t/t₀). Default true."},
                    "x_norm": {"type": "number", "description": "X normalization constant (default: first iteration). Matches page Analysis Controls."},
                    "conv_windows": {"type": "array", "items": {"type": "integer"}, "description": "Window sizes for running std, e.g. [10, 20]. Default: derived from data length."},
                },
            },
        },
    ]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute real-space isotropy tool."""
    session_context = session_context or {}

    if name == "compute_isotropy":
        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return json.dumps({"status": "error", "message": "eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."})
        turb = load_turbulence_data(p)
        R = compute_reynolds_from_fractions(turb)
        b = anisotropy_tensor(R)
        inv = invariants(b)
        score = float(np.clip(1.0 - np.mean(inv["anis_index"]), 0, 1))
        _write_back_data_dir(session_context, p)
        return json.dumps({"isotropy": score})

    if name == "plot_real_isotropy":
        from utils.plot_style import default_plot_style
        from visualizations.real_isotropy_vis import create_energy_fractions_figure

        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return "Error: eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."
        turb = load_turbulence_data(p)
        R = compute_reynolds_from_fractions(turb)
        b = anisotropy_tensor(R)
        inv = invariants(b)
        normalize_x = args.get("normalize_x", True)
        x_norm = args.get("x_norm")
        stationary_iter = args.get("stationary_iter")
        stationary_t = args.get("stationary_t")
        iter_norm, stationary_line_x = _compute_time_axis_and_stationary(
            turb, normalize_x=normalize_x, x_norm=x_norm,
            stationary_iter=stationary_iter, stationary_t=stationary_t,
        )
        frac_x, frac_y, frac_z = turb["frac_x"], turb["frac_y"], turb["frac_z"]
        _write_back_data_dir(session_context, p)
        save_to_cache(session_context, CACHE_KEY_REAL_ISOTROPY, {"iter_norm": iter_norm.tolist(), "frac_x": frac_x.tolist(), "frac_y": frac_y.tolist(), "frac_z": frac_z.tolist()})

        style_updates = args.get("style_updates") or {}
        # Dedicated style for subplot A only — matches page "Select plot to configure"
        style_config = session_context.get("real_isotropy_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({"x_axis_type": "linear", "y_axis_type": "linear", "line_width": 2.2})
            session_context["real_isotropy_style_config"] = style_config
        if style_updates:
            style_config.update(style_updates)
            if "custom_colors" in style_updates:
                style_config["palette"] = "Custom"
            if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                style_config["enable_custom_size"] = True
        axis_labels_real = session_context.get("axis_labels_real_iso") or {"x": "t/t0", "y": "Energy fraction"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_real = dict(axis_labels_real)
            axis_labels_real.update(agent_axis)
            session_context["axis_labels_real_iso"] = axis_labels_real
        axis_labels = agent_axis or axis_labels_real

        ma_win = args.get("ma_win")
        if ma_win is not None and (not isinstance(ma_win, int) or ma_win < 0):
            ma_win = None
        tol_list = args.get("tol_list")
        if tol_list is not None and not isinstance(tol_list, list):
            tol_list = None
        raw_data_opacity = args.get("raw_data_opacity")
        if raw_data_opacity is not None:
            style_config["raw_data_opacity"] = float(raw_data_opacity)
        default_legends = {"frac_x": "E<sub>x</sub>/E<sub>tot</sub>", "frac_y": "E<sub>y</sub>/E<sub>tot</sub>", "frac_z": "E<sub>z</sub>/E<sub>tot</sub>"}
        legend_names_real = session_context.get("real_iso_legend_names") or default_legends
        agent_legends = args.get("legend_names")
        if agent_legends and isinstance(agent_legends, dict):
            legend_names_real = dict(legend_names_real)
            legend_names_real.update(agent_legends)
            session_context["real_iso_legend_names"] = legend_names_real
        legend_names = legend_names_real
        fig = create_energy_fractions_figure(
            iter_norm, frac_x, frac_y, frac_z, style_config,
            axis_labels=axis_labels,
            legend_names=legend_names,
            apply_style=True,
            ma_win=ma_win,
            tol_list=tol_list,
            stationary_t=stationary_line_x,
        )
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Real isotropy figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "plot_lumley_triangle":
        from utils.plot_style import default_plot_style
        from visualizations.real_isotropy_vis import create_lumley_triangle_figure

        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return "Error: eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."
        turb = load_turbulence_data(p)
        stress_file = _resolve_stress_csv(p)
        R = load_reynolds_stress(stress_file, turb) if stress_file else compute_reynolds_from_fractions(turb)
        b = anisotropy_tensor(R)
        inv = invariants(b)
        xi, eta = np.asarray(inv["xi"]), np.asarray(inv["eta"])

        style_updates = args.get("style_updates") or {}
        # Dedicated style for subplot B only — matches page "Select plot to configure"
        style_config = session_context.get("lumley_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({"x_axis_type": "linear", "y_axis_type": "linear", "line_width": 1.5})
            session_context["lumley_style_config"] = style_config
        if style_updates:
            style_config.update(style_updates)
            if "custom_colors" in style_updates:
                style_config["palette"] = "Custom"
            if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                style_config["enable_custom_size"] = True
        axis_labels_lumley = session_context.get("axis_labels_lumley") or {"x": "ξ", "y": "η"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_lumley = dict(axis_labels_lumley)
            axis_labels_lumley.update(agent_axis)
            session_context["axis_labels_lumley"] = axis_labels_lumley
        axis_labels = agent_axis or axis_labels_lumley

        fig = create_lumley_triangle_figure(xi, eta, style_config, axis_labels=axis_labels, apply_style=True)
        _write_back_data_dir(session_context, p)
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Lumley triangle figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "plot_diagonal_bii":
        from utils.plot_style import default_plot_style
        from visualizations.real_isotropy_vis import create_diagonal_bii_figure

        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return "Error: eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."
        turb = load_turbulence_data(p)
        stress_file = _resolve_stress_csv(p)
        R = load_reynolds_stress(stress_file, turb) if stress_file else compute_reynolds_from_fractions(turb)
        b = anisotropy_tensor(R)
        normalize_x = args.get("normalize_x", True)
        x_norm = args.get("x_norm")
        iter_norm, _ = _compute_time_axis_and_stationary(turb, normalize_x=normalize_x, x_norm=x_norm)

        style_updates = args.get("style_updates") or {}
        palette_arg = args.get("palette")
        if palette_arg:
            style_updates = dict(style_updates)
            style_updates["palette"] = palette_arg
        # Use DEDICATED style for subplot C — never inherit from Energy Fractions (A) or Lumley (B)
        style_config = session_context.get("diagonal_bii_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({"x_axis_type": "linear", "y_axis_type": "linear", "line_width": 1.6})
            session_context["diagonal_bii_style_config"] = style_config
        if style_updates:
            style_config.update(style_updates)
            if "custom_colors" in style_updates:
                style_config["palette"] = "Custom"
            if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                style_config["enable_custom_size"] = True
        # Diagonal b_ii MUST use linear scale (b_ii can be negative; log scale causes artifacts)
        style_config["x_axis_type"] = "linear"
        style_config["y_axis_type"] = "linear"
        # Match page defaults: x="t/t₀", y="Anisotropy tensor b<sub>ij</sub>" (page uses axis_labels_real_iso["bij"])
        axis_labels_bii = session_context.get("axis_labels_diagonal_bii") or {"x": "t/t₀", "y": "Anisotropy tensor b<sub>ij</sub>"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_bii = dict(axis_labels_bii)
            axis_labels_bii.update(agent_axis)
            session_context["axis_labels_diagonal_bii"] = axis_labels_bii
        axis_labels = agent_axis or axis_labels_bii

        tol_list = args.get("tol_list")
        if tol_list is not None and not isinstance(tol_list, list):
            tol_list = None

        default_legends = {"b11": "b<sub>11</sub>", "b22": "b<sub>22</sub>", "b33": "b<sub>33</sub>"}
        legend_names_bii = session_context.get("diagonal_bii_legend_names") or default_legends
        agent_legends = args.get("legend_names")
        if agent_legends and isinstance(agent_legends, dict):
            legend_names_bii = dict(legend_names_bii)
            legend_names_bii.update(agent_legends)
            session_context["diagonal_bii_legend_names"] = legend_names_bii
        legend_names = legend_names_bii
        fig = create_diagonal_bii_figure(
            iter_norm, b["b11"], b["b22"], b["b33"], style_config,
            axis_labels=axis_labels,
            legend_names=legend_names,
            apply_style=True,
            tol_list=tol_list,
        )
        _write_back_data_dir(session_context, p)
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Diagonal b_ii figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "plot_cross_correlations":
        from utils.plot_style import default_plot_style
        from visualizations.real_isotropy_vis import create_cross_correlations_figure

        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return "Error: eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."
        turb = load_turbulence_data(p)
        stress_file = _resolve_stress_csv(p)
        R = load_reynolds_stress(stress_file, turb) if stress_file else compute_reynolds_from_fractions(turb)
        b = anisotropy_tensor(R)
        inv = invariants(b)
        normalize_x = args.get("normalize_x", True)
        x_norm = args.get("x_norm")
        iter_norm, _ = _compute_time_axis_and_stationary(turb, normalize_x=normalize_x, x_norm=x_norm)

        style_updates = args.get("style_updates") or {}
        palette_arg = args.get("palette")
        if palette_arg:
            style_updates = dict(style_updates)
            style_updates["palette"] = palette_arg
        style_config = session_context.get("cross_corr_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({"x_axis_type": "linear", "y_axis_type": "log", "line_width": 2.2})
            session_context["cross_corr_style_config"] = style_config
        if style_updates:
            style_config.update(style_updates)
            if "custom_colors" in style_updates:
                style_config["palette"] = "Custom"
            if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                style_config["enable_custom_size"] = True

        axis_labels_cross = session_context.get("axis_labels_cross_corr") or {"x": "t/t₀", "y": "Cross-correlations / Anisotropy index"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_cross = dict(axis_labels_cross)
            axis_labels_cross.update(agent_axis)
            session_context["axis_labels_cross_corr"] = axis_labels_cross
        axis_labels = agent_axis or axis_labels_cross

        tol_list = args.get("tol_list")
        if tol_list is not None and not isinstance(tol_list, list):
            tol_list = None
        if tol_list is None:
            tol_list = [0.001, 0.01]

        default_legends = {"b12": "|b<sub>12</sub>|", "b13": "|b<sub>13</sub>|", "b23": "|b<sub>23</sub>|", "anis": "Anisotropy index"}
        legend_names_cross = session_context.get("cross_corr_legend_names") or default_legends
        agent_legends = args.get("legend_names")
        if agent_legends and isinstance(agent_legends, dict):
            legend_names_cross = dict(legend_names_cross)
            legend_names_cross.update(agent_legends)
            session_context["cross_corr_legend_names"] = legend_names_cross
        legend_names = legend_names_cross
        fig = create_cross_correlations_figure(
            iter_norm,
            b["b12"], b["b13"], b["b23"],
            inv["anis_index"],
            style_config,
            axis_labels=axis_labels,
            legend_names=legend_names,
            tol_list=tol_list,
            apply_style=True,
        )
        _write_back_data_dir(session_context, p)
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Cross-correlations figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "plot_deviations":
        from utils.plot_style import default_plot_style
        from visualizations.real_isotropy_vis import create_deviations_figure

        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return "Error: eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."
        turb = load_turbulence_data(p)
        normalize_x = args.get("normalize_x", True)
        x_norm = args.get("x_norm")
        stationary_iter = args.get("stationary_iter")
        stationary_t = args.get("stationary_t")
        iter_norm, stationary_line_x = _compute_time_axis_and_stationary(
            turb, normalize_x=normalize_x, x_norm=x_norm,
            stationary_iter=stationary_iter, stationary_t=stationary_t,
        )

        frac_x = np.asarray(turb["frac_x"], dtype=float)
        frac_y = np.asarray(turb["frac_y"], dtype=float)
        frac_z = np.asarray(turb["frac_z"], dtype=float)
        devx = np.abs(frac_x - 1.0 / 3)
        devy = np.abs(frac_y - 1.0 / 3)
        devz = np.abs(frac_z - 1.0 / 3)
        maxdev = np.maximum(np.maximum(devx, devy), devz)

        style_updates = args.get("style_updates") or {}
        palette_arg = args.get("palette")
        if palette_arg:
            style_updates = dict(style_updates)
            style_updates["palette"] = palette_arg
        style_config = session_context.get("deviations_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({"x_axis_type": "linear", "y_axis_type": "log", "line_width": 2.2})
            session_context["deviations_style_config"] = style_config
        if style_updates:
            style_config.update(style_updates)
            if "custom_colors" in style_updates:
                style_config["palette"] = "Custom"
            if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                style_config["enable_custom_size"] = True

        axis_labels_dev = session_context.get("axis_labels_deviations") or {"x": "t/t₀", "y": "Absolute deviation"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_dev = dict(axis_labels_dev)
            axis_labels_dev.update(agent_axis)
            session_context["axis_labels_deviations"] = axis_labels_dev
        axis_labels = agent_axis or axis_labels_dev

        tol_list = args.get("tol_list")
        if tol_list is not None and not isinstance(tol_list, list):
            tol_list = None
        if tol_list is None:
            tol_list = [0.01, 0.02]

        default_legends = {"devx": "devx", "devy": "devy", "devz": "devz", "maxdev": "Max deviation"}
        legend_names_dev = session_context.get("deviations_legend_names") or default_legends
        agent_legends = args.get("legend_names")
        if agent_legends and isinstance(agent_legends, dict):
            legend_names_dev = dict(legend_names_dev)
            legend_names_dev.update(agent_legends)
            session_context["deviations_legend_names"] = legend_names_dev
        legend_names = legend_names_dev
        fig = create_deviations_figure(
            iter_norm, devx, devy, devz, maxdev, style_config,
            axis_labels=axis_labels,
            legend_names=legend_names,
            tol_list=tol_list,
            stationary_t=stationary_line_x,
            apply_style=True,
        )
        _write_back_data_dir(session_context, p)
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Deviations figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "get_real_isotropy_summary":
        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return "Error: eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."
        turb = load_turbulence_data(p)
        stress_file = _resolve_stress_csv(p)
        R = load_reynolds_stress(stress_file, turb) if stress_file else compute_reynolds_from_fractions(turb)
        b = anisotropy_tensor(R)
        inv = invariants(b)
        frac_x = np.asarray(turb["frac_x"], dtype=float)
        frac_y = np.asarray(turb["frac_y"], dtype=float)
        frac_z = np.asarray(turb["frac_z"], dtype=float)
        summary_row = {
            "Final Ex": float(frac_x[-1]),
            "Final Ey": float(frac_y[-1]),
            "Final Ez": float(frac_z[-1]),
            "Final anisotropy index": float(inv["anis_index"][-1]),
            "Mean anisotropy index": float(np.mean(inv["anis_index"])),
        }
        headers = list(summary_row.keys())
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        lines.append("| " + " | ".join(str(summary_row[h]) for h in headers) + " |")
        table_md = "\n".join(lines)
        _write_back_data_dir(session_context, p)
        # Store for add_report_section when table_data not provided
        session_context["last_real_isotropy_summary_rows"] = [summary_row]
        return {
            "status": "success",
            "message": f"Real isotropy summary:\n\n{table_md}",
            "artifact_type": "markdown_table",
            "artifact_content": table_md,
            "summary_rows": [summary_row],
        }

    if name == "get_real_isotropy_theory":
        subplot = args.get("subplot")
        content = get_real_isotropy_theory_markdown(subplot)
        title = f"Theory & Equations (Subplot {subplot})" if subplot else "Theory & Equations"
        return {
            "status": "success",
            "message": f"Theory & Equations for real-space isotropy{subplot and f' (Subplot {subplot})' or ''}.",
            "artifact_type": "markdown",
            "artifact_content": content,
            "artifact_title": title,
        }

    if name == "plot_convergence":
        from utils.plot_style import default_plot_style
        from visualizations.real_isotropy_vis import create_convergence_figure

        csv_path = args.get("csv_path", "")
        data_dir = args.get("data_dir", "")
        p = _resolve_eps_csv(csv_path, data_dir, project_root, session_context)
        if p is None or not p.exists():
            return "Error: eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir or csv_path."
        turb = load_turbulence_data(p)
        normalize_x = args.get("normalize_x", True)
        x_norm = args.get("x_norm")
        iter_norm, _ = _compute_time_axis_and_stationary(turb, normalize_x=normalize_x, x_norm=x_norm)

        frac_x = np.asarray(turb["frac_x"], dtype=float)
        frac_y = np.asarray(turb["frac_y"], dtype=float)
        frac_z = np.asarray(turb["frac_z"], dtype=float)

        style_updates = args.get("style_updates") or {}
        palette_arg = args.get("palette")
        if palette_arg:
            style_updates = dict(style_updates)
            style_updates["palette"] = palette_arg
        style_config = session_context.get("convergence_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({"x_axis_type": "linear", "y_axis_type": "log", "line_width": 1.5})
            session_context["convergence_style_config"] = style_config
        if style_updates:
            style_config.update(style_updates)
            if "custom_colors" in style_updates:
                style_config["palette"] = "Custom"
            if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                style_config["enable_custom_size"] = True

        axis_labels_conv = session_context.get("axis_labels_convergence") or {"x": "t/t₀", "y": "Running standard deviation"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_conv = dict(axis_labels_conv)
            axis_labels_conv.update(agent_axis)
            session_context["axis_labels_convergence"] = axis_labels_conv
        axis_labels = agent_axis or axis_labels_conv

        conv_windows = args.get("conv_windows")
        if conv_windows is not None and not isinstance(conv_windows, list):
            conv_windows = None

        fig = create_convergence_figure(
            iter_norm, frac_x, frac_y, frac_z, style_config,
            axis_labels=axis_labels,
            conv_windows=conv_windows,
            apply_style=True,
        )
        _write_back_data_dir(session_context, p)
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Convergence figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    return f"Error: Unknown real isotropy tool '{name}'"
