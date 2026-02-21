"""
Spectral isotropy agent tools: IC(k) from isotropy_coeff_*.dat.
"""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core_physics import read_isotropy_coeff_file, avg_isotropy_coeff

from .._shared import (
    get_from_cache,
    resolve_data_dirs_and_group_isotropy,
    save_to_cache,
)
from ._meta import get_artifact_source_meta


CACHE_KEY_SPECTRAL_ISOTROPY = "current_spectral_isotropy_data"
PATTERN_ISOTROPY_COEFF = "isotropy_coeff_*.dat"


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for spectral isotropy."""
    return [
        {
            "name": "compute_spectral_isotropy",
            "description": "Compute spectral isotropy IC(k) from isotropy_coeff_*.dat. Data stored in cache for plot_spectral_isotropy. Supports multi-simulation comparison and time-window selection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Single directory path (optional: uses SESSION DATA PATH if not set)"},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple directories for multi-simulation comparison (overrides data_dir when set)"},
                    "start_idx": {"type": "integer", "description": "1-based start file index for time window (default 1)"},
                    "end_idx": {"type": "integer", "description": "1-based end file index for time window (default: last file)"},
                    "max_files": {"type": "integer", "description": "Max files per simulation group (default 1000)"},
                },
            },
        },
        {
            "name": "plot_spectral_isotropy",
            "description": "Plot spectral isotropy IC(k) vs k. Call compute_spectral_isotropy first. Supports multi-simulation, per-snapshot lines, and full style API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "Cache key (default: current_spectral_isotropy_data)"},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API (matches IC(k) Time-Avg sidebar): Fonts: font_family, font_size, title_size, legend_size, tick_font_size, axis_title_size, font_color. Backgrounds: plot_bgcolor, paper_bgcolor. Ticks: tick_len, tick_w, ticks_outside, tick_color. Axis: x_axis_type, y_axis_type, x_tick_format, y_tick_format. Borders: show_axis_lines, axis_line_width, axis_line_color, mirror_axes. Grid: show_grid, grid_on_x, grid_on_y, grid_w, grid_dash, grid_color, grid_opacity. Minor grid: show_minor_grid, minor_grid_w, minor_grid_dash, minor_grid_color, minor_grid_opacity. Curves: line_width, marker_size. Colors: palette, custom_colors. Theme: template. Legend: show_legend. Title: show_plot_title, plot_title. Limits: enable_x_limits, x_min, x_max, enable_y_limits, y_min, y_max. Size: enable_custom_size, figure_width, figure_height. Margins: margin_left, margin_top, margin_right, margin_bottom.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"k\", \"y\": \"IC(k)\"}. Partial OK."},
                    "error_display": {"type": "string", "description": "How to show ±1σ uncertainty: Shaded band | Error bars | Both | None. Default: Shaded band."},
                    "show_snapshot_lines": {"type": "boolean", "description": "Show per-snapshot IC(k) lines (convergence visualization). Default: false."},
                    "simulation_legend_names": {"type": "object", "description": "Override legend names per simulation: {\"sim_prefix\": \"Display Name\"}. Partial OK."},
                    "y_limits": {"type": "object", "description": "Y-axis limits: {\"min\": N, \"max\": M}. Overrides style_updates when user asks for y-axis range."},
                },
            },
        },
        {
            "name": "plot_component_spectra",
            "description": "Plot component spectra E11(k), E22(k), E33(k) vs k. Call compute_spectral_isotropy first. Uses same cache as plot_spectral_isotropy. Supports multi-simulation and full style API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "Cache key (default: current_spectral_isotropy_data)"},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API (matches Component Spectra sidebar): x_axis_type, y_axis_type (default log-log), font_family, font_size, line_width, palette, etc.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"k\", \"y\": \"E_ii(k)\"}. Partial OK."},
                    "simulation_legend_names": {"type": "object", "description": "Override legend names per simulation: {\"sim_prefix\": \"Display Name\"}."},
                    "curve_legend_names": {"type": "object", "description": "Override curve names: {\"E11\": \"E11(k)\", \"E22\": \"E22(k)\", \"E33\": \"E33(k)\"}."},
                    "show_curves": {"type": "array", "items": {"type": "string"}, "description": "Which curves to show: [\"E11\", \"E22\", \"E33\"]. Default: all. Use subset to hide curves, e.g. [\"E11\", \"E22\"] to hide E33."},
                    "y_limits": {"type": "object", "description": "Y-axis limits: {\"min\": N, \"max\": M}."},
                    "x_limits": {"type": "object", "description": "X-axis limits: {\"min\": N, \"max\": M}."},
                },
            },
        },
        {
            "name": "get_spectral_isotropy_summary",
            "description": "Show spectral isotropy summary table (Simulation, Snapshots used, Mean IC, Std(IC), Min IC, Max IC). Call compute_spectral_isotropy first. Use when user asks for 'summary', 'table', or 'statistics' of spectral isotropy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "Cache key (default: current_spectral_isotropy_data)"},
                    "simulation_legend_names": {"type": "object", "description": "Override legend names: {\"sim_prefix\": \"Display Name\"}."},
                },
            },
        },
        {
            "name": "get_spectral_isotropy_theory",
            "description": "Show Spectral Isotropy page Theory & Equations: one-dimensional energy spectra E11/E22/E33, derivative-based IC(k), isotropic turbulence. Use when user asks for 'spectral isotropy theory', 'spectral isotropy equations', 'theory for spectral isotropy', 'equations for spectral isotropy', 'IC(k) theory'.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "export_isotropy_data",
            "description": "Export cached isotropy data to CSV. Call after compute_spectral_isotropy. format: ic | component | summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "current_spectral_isotropy_data (default) | current_real_isotropy_data"},
                    "filename": {"type": "string", "description": "Output filename (default: isotropy_export.csv)"},
                    "format": {"type": "string", "description": "Export format: ic (IC(k) data) | component (E11/E22/E33) | summary (Mean IC, Std, Min, Max per simulation). Default: ic."},
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
    """Execute spectral isotropy tool."""
    session_context = session_context or {}

    if name == "compute_spectral_isotropy":
        data_dir = args.get("data_dir", "")
        data_dirs = args.get("data_directories") or []
        start_idx = max(1, int(args.get("start_idx", 1)))
        end_idx = args.get("end_idx")  # None = use last
        max_files = int(args.get("max_files", 1000))

        ic_groups = resolve_data_dirs_and_group_isotropy(
            data_dirs=data_dirs if data_dirs else None,
            data_dir=data_dir,
            pattern=PATTERN_ISOTROPY_COEFF,
            project_root=project_root,
            session_context=session_context,
            max_files_per_group=max_files,
        )
        if not ic_groups:
            return json.dumps({
                "status": "error",
                "message": "No isotropy_coeff_*.dat found. Try data_dir='examples/LES/64' or set Data directory in sidebar. Use find_file(pattern='isotropy_coeff*.dat', directory='.') to locate.",
            })

        # Apply time window (1-based indices)
        results_by_sim: Dict[str, Dict[str, Any]] = {}
        for sim_prefix, files in ic_groups.items():
            n = len(files)
            end = int(end_idx) if end_idx is not None else n
            end = min(max(end, start_idx), n)
            start = min(start_idx, end)
            selected = files[start - 1 : end]
            if not selected:
                continue
            data_list = []
            for f in selected:
                d = read_isotropy_coeff_file(Path(f) if not isinstance(f, Path) else f)
                if d.size > 0:
                    data_list.append(d)
            result = avg_isotropy_coeff(data_list)
            if result is not None:
                result = dict(result)
                result["files"] = [str(f) for f in selected]  # for per-snapshot plotting
                results_by_sim[sim_prefix] = result

        if not results_by_sim:
            return json.dumps({"status": "error", "message": "Could not compute spectral isotropy from files."})

        # Store: single sim = flat dict (backward compat), multi = {"simulations": {...}}
        if len(results_by_sim) == 1:
            cache_data = list(results_by_sim.values())[0]
        else:
            cache_data = {"simulations": results_by_sim}

        save_to_cache(session_context, CACHE_KEY_SPECTRAL_ISOTROPY, cache_data)
        n_sims = len(results_by_sim)
        return json.dumps({
            "status": "success",
            "message": f"Computed spectral isotropy from {n_sims} simulation(s). Data ready for plot_spectral_isotropy.",
        })

    if name == "plot_spectral_isotropy":
        from utils.plot_style import default_plot_style
        from visualizations.spectral_isotropy_vis import create_ic_isotropy_figure

        ref = args.get("data_reference") or CACHE_KEY_SPECTRAL_ISOTROPY
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No spectral isotropy data. Run compute_spectral_isotropy first."

        # Normalize: single sim = flat dict, multi = {"simulations": {sim: data}}
        if "simulations" in cached:
            sim_items = list(sorted(cached["simulations"].items()))
        else:
            sim_items = [("default", dict(cached))]

        def _default_labelify(s: str) -> str:
            return s.replace("_", " ").title()

        simulation_legend_names = args.get("simulation_legend_names") or {}
        if isinstance(simulation_legend_names, dict):
            session_context.setdefault("spec_iso_sim_legend_names", {}).update(simulation_legend_names)
        sim_legends = session_context.get("spec_iso_sim_legend_names") or {}

        # Same agentic schema: plot_styles_refs + style_updates + axis_labels
        style_updates = args.get("style_updates") or {}
        plot_styles_refs = session_context.get("isotropy_plot_styles") or {}
        if not plot_styles_refs:
            style_config = session_context.get("isotropy_style_config")
            if style_config is None:
                style_config = default_plot_style()
                style_config.update({
                    "x_axis_type": "log", "y_axis_type": "linear", "line_width": 2.2,
                    "margin_left": 60, "margin_right": 20, "margin_top": 40, "margin_bottom": 50,
                    "enable_y_limits": True, "y_min": 0.5, "y_max": 1.5,
                })
                session_context.setdefault("plot_styles", {})["IC(k) Time-Avg"] = style_config
                session_context["isotropy_style_config"] = style_config
            plot_styles_refs = {"IC(k) Time-Avg": style_config}
        if style_updates:
            for _name, ref in plot_styles_refs.items():
                if isinstance(ref, dict):
                    ref.update(style_updates)
                    if "custom_colors" in style_updates:
                        ref["palette"] = "Custom"
                    if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                        ref["enable_custom_size"] = True
        y_limits = args.get("y_limits") or style_updates.get("y_limits")
        if y_limits and isinstance(y_limits, dict):
            ymin, ymax = y_limits.get("min"), y_limits.get("max")
            if ymin is not None and ymax is not None:
                for _name, ref in plot_styles_refs.items():
                    if isinstance(ref, dict):
                        ref["enable_y_limits"] = True
                        ref["y_min"] = float(ymin)
                        ref["y_max"] = float(ymax)
        style_config = plot_styles_refs.get("IC(k) Time-Avg") or (list(plot_styles_refs.values())[0] if plot_styles_refs else default_plot_style())
        axis_labels_spec = session_context.get("axis_labels_spec_iso") or {"x": "k", "y": "IC(k)"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_spec = dict(axis_labels_spec)
            axis_labels_spec.update(agent_axis)
            session_context["axis_labels_spec_iso"] = axis_labels_spec
        axis_labels = agent_axis or axis_labels_spec

        error_display = (args.get("error_display") or "Shaded band").strip()
        if error_display not in ("Shaded band", "Error bars", "Both", "None"):
            error_display = "Shaded band"
        show_std_band = error_display in ("Shaded band", "Both")
        show_error_bars = error_display in ("Error bars", "Both")
        show_snapshot_lines = bool(args.get("show_snapshot_lines", False))

        ic_snap_label = session_context.get("spec_iso_legends", {}).get("IC_snap", "IC(k) snapshots")

        # Build sim_items with snapshot_curves when needed
        sim_items_for_vis = []
        for sim_prefix, data in sim_items:
            d = dict(data)
            if show_snapshot_lines:
                snapshot_curves = []
                for fpath in d.get("files", []):
                    try:
                        rd = read_isotropy_coeff_file(Path(fpath))
                        if rd.size == 0:
                            continue
                        k0 = rd[:, 0]
                        IC0 = rd[:, 6] if rd.shape[1] >= 7 else np.divide(rd[:, 1], rd[:, 2], out=np.zeros_like(rd[:, 1]), where=rd[:, 2] != 0)
                        snapshot_curves.append((k0, IC0))
                    except Exception:
                        continue
                d["snapshot_curves"] = snapshot_curves
            sim_items_for_vis.append((sim_prefix, d))

        sim_legend_map = {k: simulation_legend_names.get(k) or sim_legends.get(k) or _default_labelify(k) for k, _ in sim_items}
        fig = create_ic_isotropy_figure(
            sim_items_for_vis, style_config,
            show_std_band=show_std_band,
            show_error_bars=show_error_bars,
            show_snapshot_lines=show_snapshot_lines,
            axis_labels=axis_labels,
            simulation_legend_names=sim_legend_map,
            ic_snap_label=ic_snap_label,
            apply_style=True,
        )
        if fig is None:
            return "Error: No valid IC(k) data to plot."
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Spectral isotropy figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "plot_component_spectra":
        from utils.plot_style import default_plot_style
        from visualizations.spectral_isotropy_vis import create_component_spectra_figure

        ref = args.get("data_reference") or CACHE_KEY_SPECTRAL_ISOTROPY
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No spectral isotropy data. Run compute_spectral_isotropy first."

        if "simulations" in cached:
            sim_items = list(sorted(cached["simulations"].items()))
        else:
            sim_items = [("default", dict(cached))]

        def _default_labelify(s: str) -> str:
            return s.replace("_", " ").title()

        simulation_legend_names = args.get("simulation_legend_names") or {}
        if isinstance(simulation_legend_names, dict):
            session_context.setdefault("spec_iso_sim_legend_names", {}).update(simulation_legend_names)
        sim_legend_names = session_context.get("spec_iso_sim_legend_names") or {}
        curve_legend_names = args.get("curve_legend_names") or {}
        if isinstance(curve_legend_names, dict):
            session_context.setdefault("spec_iso_legends", {}).update(curve_legend_names)
        curve_legends = session_context.get("spec_iso_legends") or {}
        default_curve_labels = {"E11": "E<sub>11</sub>(k)", "E22": "E<sub>22</sub>(k)", "E33": "E<sub>33</sub>(k)"}
        for k, v in default_curve_labels.items():
            curve_legends.setdefault(k, v)

        style_updates = args.get("style_updates") or {}
        style_config = session_context.get("component_spectra_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({
                "x_axis_type": "log", "y_axis_type": "log", "line_width": 2.2,
                "margin_left": 60, "margin_right": 20, "margin_top": 40, "margin_bottom": 50,
            })
            session_context.setdefault("plot_styles", {})["Component Spectra"] = style_config
            session_context["component_spectra_style_config"] = style_config
        if style_updates:
            for _name, ref in [("Component Spectra", style_config)]:
                if isinstance(ref, dict):
                    ref.update(style_updates)
                    if "custom_colors" in style_updates:
                        ref["palette"] = "Custom"
        y_limits = args.get("y_limits") or style_updates.get("y_limits")
        x_limits = args.get("x_limits") or style_updates.get("x_limits")
        if y_limits and isinstance(y_limits, dict):
            ymin, ymax = y_limits.get("min"), y_limits.get("max")
            if ymin is not None and ymax is not None:
                style_config["enable_y_limits"] = True
                style_config["y_min"] = float(ymin)
                style_config["y_max"] = float(ymax)
        if x_limits and isinstance(x_limits, dict):
            xmin, xmax = x_limits.get("min"), x_limits.get("max")
            if xmin is not None and xmax is not None:
                style_config["enable_x_limits"] = True
                style_config["x_min"] = float(xmin)
                style_config["x_max"] = float(xmax)
        axis_labels_spec = session_context.get("axis_labels_spec_iso") or {"x": "k", "y": "E<sub>ii</sub>(k)"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_spec = dict(axis_labels_spec)
            axis_labels_spec.update(agent_axis)
            session_context["axis_labels_spec_iso"] = axis_labels_spec
        axis_labels = agent_axis or axis_labels_spec

        sim_legend_map = {k: simulation_legend_names.get(k) or sim_legend_names.get(k) or _default_labelify(k) for k, _ in sim_items}
        show_curves = args.get("show_curves")
        fig = create_component_spectra_figure(
            sim_items, style_config,
            axis_labels=axis_labels,
            simulation_legend_names=sim_legend_map,
            curve_legend_names=curve_legends,
            show_curves=show_curves,
            apply_style=True,
        )
        if fig is None:
            return "Error: No component spectra data (E11/E22/E33). Run compute_spectral_isotropy first; data may lack E11/E22/E33 columns."
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Component spectra figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "get_spectral_isotropy_theory":
        from content.spectral_isotropy_theory_content import get_spectral_isotropy_theory_markdown
        content = get_spectral_isotropy_theory_markdown()
        return {
            "status": "success",
            "message": "Spectral isotropy theory equations created.",
            "artifact_type": "markdown",
            "artifact_content": content,
            "artifact_title": "Spectral Isotropy — Theory & Equations",
        }

    if name == "get_spectral_isotropy_summary":
        ref = args.get("data_reference") or CACHE_KEY_SPECTRAL_ISOTROPY
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No spectral isotropy data. Run compute_spectral_isotropy first."

        def _default_labelify(s: str) -> str:
            return s.replace("_", " ").title()

        sim_legend_names = args.get("simulation_legend_names") or {}
        if isinstance(sim_legend_names, dict):
            session_context.setdefault("spec_iso_sim_legend_names", {}).update(sim_legend_names)
        legends = session_context.get("spec_iso_sim_legend_names") or {}

        if "simulations" in cached:
            sim_items = sorted(cached["simulations"].items())
        else:
            sim_items = [("default", cached)]

        summary_rows = []
        for sim_prefix, data in sim_items:
            IC_mean = np.asarray(data.get("IC_mean", []))
            IC_std = data.get("IC_std")
            files = data.get("files", [])
            n_snapshots = len(files) if files else 0
            if len(IC_mean) == 0:
                continue
            legend_name = sim_legend_names.get(sim_prefix) or legends.get(sim_prefix) or _default_labelify(sim_prefix)
            ic_std_arr = np.asarray(IC_std) if IC_std is not None else None
            std_val = float(np.nanmean(ic_std_arr)) if ic_std_arr is not None and len(ic_std_arr) > 0 else float("nan")
            summary_rows.append({
                "Simulation": legend_name,
                "Snapshots used": n_snapshots,
                "Mean IC": float(np.nanmean(IC_mean)),
                "Std(IC)": std_val,
                "Min IC": float(np.nanmin(IC_mean)),
                "Max IC": float(np.nanmax(IC_mean)),
            })

        if not summary_rows:
            return "Error: No valid spectral isotropy data for summary. Run compute_spectral_isotropy first."

        # Build markdown table for display in chat
        headers = ["Simulation", "Snapshots used", "Mean IC", "Std(IC)", "Min IC", "Max IC"]
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in summary_rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        table_md = "\n".join(lines)

        return {
            "status": "success",
            "message": f"Spectral isotropy summary:\n\n{table_md}",
            "artifact_type": "markdown_table",
            "artifact_content": table_md,
            "summary_rows": summary_rows,
        }

    if name == "export_isotropy_data":
        import pandas as pd

        ref = args.get("data_reference") or CACHE_KEY_SPECTRAL_ISOTROPY
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No data in cache. For spectral (IC/E11/E22/E33): run compute_spectral_isotropy. For real (fractions): run plot_real_isotropy first."
        export_format = (args.get("format") or "ic").strip().lower()
        fname = args.get("filename") or ("spectral_isotropy_summary.csv" if export_format == "summary" else "isotropy_export.csv")
        if not fname.lower().endswith(".csv"):
            fname = f"{fname}.csv"
        try:
            if export_format == "summary":
                if "simulations" in cached:
                    rows = []
                    for sim_prefix, data in cached["simulations"].items():
                        IC_mean = np.asarray(data.get("IC_mean", []))
                        IC_std = data.get("IC_std")
                        files = data.get("files", [])
                        n_snap = len(files) if files else 0
                        if len(IC_mean) == 0:
                            continue
                        ic_std_arr = np.asarray(IC_std) if IC_std is not None else None
                        std_val = float(np.nanmean(ic_std_arr)) if ic_std_arr is not None and len(ic_std_arr) > 0 else float("nan")
                        rows.append({
                            "Simulation": sim_prefix,
                            "Snapshots used": n_snap,
                            "Mean IC": float(np.nanmean(IC_mean)),
                            "Std(IC)": std_val,
                            "Min IC": float(np.nanmin(IC_mean)),
                            "Max IC": float(np.nanmax(IC_mean)),
                        })
                    df = pd.DataFrame(rows)
                elif "IC_mean" in cached:
                    IC_mean = np.asarray(cached.get("IC_mean", []))
                    IC_std = cached.get("IC_std")
                    files = cached.get("files", [])
                    n_snap = len(files) if files else 0
                    ic_std_arr = np.asarray(IC_std) if IC_std is not None else None
                    std_val = float(np.nanmean(ic_std_arr)) if ic_std_arr is not None and len(ic_std_arr) > 0 else float("nan")
                    df = pd.DataFrame([{
                        "Simulation": "default",
                        "Snapshots used": n_snap,
                        "Mean IC": float(np.nanmean(IC_mean)),
                        "Std(IC)": std_val,
                        "Min IC": float(np.nanmin(IC_mean)),
                        "Max IC": float(np.nanmax(IC_mean)),
                    }])
                else:
                    return "Error: No spectral isotropy data for summary. Run compute_spectral_isotropy first."
            elif export_format == "component":
                if "simulations" in cached:
                    rows = []
                    for sim_prefix, data in cached["simulations"].items():
                        k = np.asarray(data.get("k", []))
                        e11 = np.asarray(data.get("E11_mean", []))
                        e22 = np.asarray(data.get("E22_mean", []))
                        e33 = np.asarray(data.get("E33_mean", []))
                        n = len(k)
                        for i in range(n):
                            rows.append({
                                "simulation": sim_prefix,
                                "k": float(k[i]),
                                "E11_mean": float(e11[i]) if i < len(e11) else None,
                                "E22_mean": float(e22[i]) if i < len(e22) else None,
                                "E33_mean": float(e33[i]) if i < len(e33) else None,
                            })
                    df = pd.DataFrame(rows)
                elif cached.get("E11_mean") is not None:
                    df = pd.DataFrame({
                        "k": cached.get("k", []),
                        "E11_mean": cached.get("E11_mean", []),
                        "E22_mean": cached.get("E22_mean", []),
                        "E33_mean": cached.get("E33_mean", []),
                    })
                else:
                    return "Error: No component spectra data. Run compute_spectral_isotropy first."
            elif "simulations" in cached:
                rows = []
                for sim_prefix, data in cached["simulations"].items():
                    k = np.asarray(data.get("k", []))
                    ic_mean = np.asarray(data.get("IC_mean", []))
                    ic_std_raw = data.get("IC_std")
                    n = len(k)
                    for i in range(n):
                        row = {"simulation": sim_prefix, "k": float(k[i]), "IC_mean": float(ic_mean[i]) if i < len(ic_mean) else None}
                        if ic_std_raw is not None and i < len(ic_std_raw):
                            row["IC_std"] = float(ic_std_raw[i])
                        rows.append(row)
                df = pd.DataFrame(rows)
            elif "IC_mean" in cached:
                df = pd.DataFrame({
                    "k": cached.get("k", []),
                    "IC_mean": cached.get("IC_mean", []),
                    "IC_std": cached.get("IC_std", []),
                })
            else:
                df = pd.DataFrame({
                    "iter_norm": cached.get("iter_norm", []),
                    "frac_x": cached.get("frac_x", []),
                    "frac_y": cached.get("frac_y", []),
                    "frac_z": cached.get("frac_z", []),
                })
            buf = df.to_csv(index=False)
            content = buf.encode("utf-8")
            b64 = base64.b64encode(content).decode("ascii")
            return {
                "status": "success",
                "message": f"Data exported as {fname}.",
                "artifact_type": "downloadable_file",
                "filename": fname,
                "mime_type": "text/csv",
                "content_base64": b64,
            }
        except Exception as e:
            return f"Error exporting data: {e}"

    return f"Error: Unknown spectral isotropy tool '{name}'"
