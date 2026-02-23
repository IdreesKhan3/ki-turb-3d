"""
Flatness factor agent tools: F(r) from flatness_data*_*.txt.
"""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core_physics.flatness import compute_flatness_time_avg
from visualizations.flatness_vis import create_flatness_figure
from .._shared import (
    get_from_cache,
    resolve_data_dirs_and_group_files,
    save_to_cache,
)
from ._meta import get_artifact_source_meta


CACHE_KEY_FLATNESS = "current_flatness_data"
PATTERN_FLATNESS_DATA = "flatness_data*_*.txt"


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for flatness."""
    return [
        {
            "name": "compute_flatness",
            "description": "Compute flatness factor F(r) from flatness_data*_*.txt. Data stored in cache for plot_flatness. Supports multi-simulation comparison and time-window selection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Single directory path (optional: uses SESSION DATA PATH if not set)"},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple directories for multi-simulation comparison (overrides data_dir when set)"},
                    "start_idx": {"type": "integer", "description": "1-based start file index for time window (default 1)"},
                    "end_idx": {"type": "integer", "description": "1-based end file index for time window (default: last file)"},
                    "max_files": {"type": "integer", "description": "Max files per simulation group (default 1000)"},
                    "num_errorbars": {"type": "integer", "description": "Number of log-spaced r positions for output (default 20)"},
                },
            },
        },
        {
            "name": "plot_flatness",
            "description": "Plot flatness factor F(r) vs r. Call compute_flatness first. Supports multi-simulation and full style API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "Cache key (default: current_flatness_data)"},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API (matches Flatness sidebar): Fonts: font_family, font_size, title_size, legend_size, tick_font_size, axis_title_size, font_color. Backgrounds: plot_bgcolor, paper_bgcolor. Ticks: tick_len, tick_w, ticks_outside, tick_color. Axis: x_axis_type, y_axis_type, x_tick_format, y_tick_format. Borders: show_axis_lines, axis_line_width, axis_line_color, mirror_axes. Grid: show_grid, grid_on_x, grid_on_y, grid_w, grid_dash, grid_color, grid_opacity. Minor grid: show_minor_grid, minor_grid_w, minor_grid_dash, minor_grid_color, minor_grid_opacity. Curves: line_width, marker_size. Colors: palette, custom_colors. Theme: template. Legend: show_legend. Title: show_plot_title, plot_title. Limits: enable_x_limits, x_min, x_max, enable_y_limits, y_min, y_max. Size: enable_custom_size, figure_width, figure_height. Margins: margin_left, margin_top, margin_right, margin_bottom.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"r\", \"y\": \"F(r)\"}. Partial OK."},
                    "error_display": {"type": "string", "description": "How to show \u00b11\u03c3 uncertainty: Shaded band | Error bars | Both | None. Default: Shaded band."},
                    "simulation_legend_names": {"type": "object", "description": "Override legend names per simulation: {\"sim_prefix\": \"Display Name\"}. Partial OK."},
                    "y_limits": {"type": "object", "description": "Y-axis limits: {\"min\": N, \"max\": M}. Overrides style_updates when user asks for y-axis range."},
                    "x_limits": {"type": "object", "description": "X-axis limits: {\"min\": N, \"max\": M}. Overrides style_updates when user asks for x-axis range."},
                },
            },
        },
        {
            "name": "get_flatness_summary",
            "description": "Show flatness summary table (Simulation, Snapshots used, Mean F(r), Std(F(r)), Min F(r), Max F(r)). Call compute_flatness first. Use when user asks for 'summary', 'table', or 'statistics' of flatness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "Cache key (default: current_flatness_data)"},
                    "simulation_legend_names": {"type": "object", "description": "Override legend names: {\"sim_prefix\": \"Display Name\"}."},
                },
            },
        },
        {
            "name": "export_flatness_data",
            "description": "Export cached flatness data to CSV. Call after compute_flatness. format: flatness | summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "current_flatness_data (default)"},
                    "filename": {"type": "string", "description": "Output filename (default: flatness_export.csv)"},
                    "format": {"type": "string", "description": "Export format: flatness (F(r) data) | summary (Mean F, Std, Min, Max per simulation). Default: flatness."},
                },
            },
        },
        {
            "name": "get_flatness_theory",
            "description": "Show Flatness page Theory & Equations: F_L(r), longitudinal velocity increment, Gaussian reference (F=3), intermittency interpretation. Use when user asks for 'flatness theory', 'flatness equations', 'theory for flatness', 'equations for flatness', 'F(r) theory', 'kurtosis theory'.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def _format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format table as GitHub-flavored markdown."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _read_flatness_file(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Reads a flatness data file (r, F)."""
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 2 and data.shape[1] == 2:
            return data[:, 0], data[:, 1]
        elif data.ndim == 1 and data.shape[0] == 2: # Handle case where only one (r,F) pair
            return np.array([data[0]]), np.array([data[1]])
        else:
            return None
    except Exception:
        return None


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute flatness tool."""
    session_context = session_context or {}

    if name == "get_flatness_theory":
        from content.flatness_theory_content import get_flatness_theory_markdown

        content = get_flatness_theory_markdown()
        return {
            "status": "success",
            "message": "Flatness theory equations created.",
            "artifact_type": "markdown",
            "artifact_content": content,
            "artifact_title": "Flatness — Theory & Equations",
        }

    if name == "compute_flatness":
        data_dir = args.get("data_dir", "")
        data_dirs = args.get("data_directories") or []
        start_idx = max(1, int(args.get("start_idx", 1)))
        end_idx = args.get("end_idx")  # None = use last
        max_files = int(args.get("max_files", 1000))
        num_errorbars = int(args.get("num_errorbars", 20))

        flatness_groups = resolve_data_dirs_and_group_files(
            data_dirs=data_dirs if data_dirs else None,
            data_dir=data_dir,
            pattern=PATTERN_FLATNESS_DATA,
            project_root=project_root,
            session_context=session_context,
            max_files_per_group=max_files,
        )
        if not flatness_groups:
            return json.dumps({
                "status": "error",
                "message": "No flatness_data*_*.txt found. Try data_dir='examples/LES/64' or set Data directory in sidebar. Use find_file(pattern='flatness_data*.txt', directory='.') to locate.",
            })

        results_by_sim: Dict[str, Dict[str, Any]] = {}
        for sim_prefix, files in flatness_groups.items():
            n = len(files)
            end = int(end_idx) if end_idx is not None else n
            end = min(max(end, start_idx), n)
            start = min(start_idx, end)
            selected_files = files[start - 1 : end]
            if not selected_files:
                continue

            data_list = []
            for f in selected_files:
                data = _read_flatness_file(Path(f) if not isinstance(f, Path) else f)
                if data is not None:
                    data_list.append(data)
            
            if not data_list:
                continue

            r_plot, F_mean, F_std = compute_flatness_time_avg(data_list, num_errorbars)

            if r_plot is not None and F_mean is not None:
                results_by_sim[sim_prefix] = {
                    "r": r_plot.tolist(),
                    "F_mean": F_mean.tolist(),
                    "F_std": F_std.tolist() if F_std is not None else [],
                    "files": [str(f) for f in selected_files],
                }

        if not results_by_sim:
            return json.dumps({"status": "error", "message": "Could not compute flatness from files."})

        if len(results_by_sim) == 1:
            cache_data = list(results_by_sim.values())[0]
        else:
            cache_data = {"simulations": results_by_sim}

        save_to_cache(session_context, CACHE_KEY_FLATNESS, cache_data)
        from pages.AutonomousLab.session_sync import update_data_directory_in_context
        first_sim = next(iter(results_by_sim.values()))
        if first_sim.get("files"):
            update_data_directory_in_context(session_context, Path(first_sim["files"][0]).parent)
        n_sims = len(results_by_sim)
        return json.dumps({
            "status": "success",
            "message": f"Computed flatness from {n_sims} simulation(s). Data ready for plot_flatness.",
        })

    if name == "plot_flatness":
        from utils.plot_style import default_plot_style
        # Assuming a visualization function will be created in visualizations/flatness_vis.py
        # For now, we'll return an error if it's not implemented.
        # from visualizations.flatness_vis import create_flatness_figure 

        ref = args.get("data_reference") or CACHE_KEY_FLATNESS
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No flatness data. Run compute_flatness first."

        from utils.plot_style import default_plot_style

        ref = args.get("data_reference") or CACHE_KEY_FLATNESS
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No flatness data. Run compute_flatness first."

        if "simulations" in cached:
            sim_items = list(sorted(cached["simulations"].items()))
        else:
            sim_items = [("default", dict(cached))]

        def _default_labelify(s: str) -> str:
            return s.replace("_", " ").title()

        simulation_legend_names = args.get("simulation_legend_names") or {}
        if isinstance(simulation_legend_names, dict):
            session_context.setdefault("flatness_legend_names", {}).update(simulation_legend_names)
        sim_legends = session_context.get("flatness_legend_names") or {}

        style_updates = args.get("style_updates") or {}
        style_config = session_context.get("flatness_style_config")
        if style_config is None:
            style_config = default_plot_style()
            style_config.update({
                "x_axis_type": "log", "y_axis_type": "linear", "line_width": 2.2,
                "margin_left": 60, "margin_right": 20, "margin_top": 40, "margin_bottom": 50,
                "enable_y_limits": False, "y_min": 0.5, "y_max": 1.5,
                "enable_x_limits": False, "x_min": 0.01, "x_max": 10.0,
                "std_alpha": 0.18, "reference_dash": "dot", "reference_color": "#000000",
                "reference_width": 1.5, "per_sim_style_flatness": {},
            })
            session_context.setdefault("plot_styles", {})["Flatness Factors"] = style_config
            session_context["flatness_style_config"] = style_config
        if style_updates:
            for _name, ref in [("Flatness Factors", style_config)]:
                if isinstance(ref, dict):
                    ref.update(style_updates)
                    if "custom_colors" in style_updates:
                        ref["palette"] = "Custom"
                    if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                        ref["enable_custom_size"] = True
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

        axis_labels_spec = session_context.get("axis_labels_flatness") or {"x": "r", "y": "F(r)"}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_spec = dict(axis_labels_spec)
            axis_labels_spec.update(agent_axis)
            session_context["axis_labels_flatness"] = axis_labels_spec
        axis_labels = agent_axis or axis_labels_spec

        error_display = (args.get("error_display") or "Shaded band").strip()
        if error_display not in ("Shaded band", "Error bars", "Both", "None"):
            error_display = "Shaded band"
        show_std_band = error_display in ("Shaded band", "Both")
        show_error_bars = error_display in ("Error bars", "Both")

        sim_legend_map = {k: simulation_legend_names.get(k) or sim_legends.get(k) or _default_labelify(k) for k, _ in sim_items}

        datasets = [
            {"sim_prefix": sp, "r": d["r"], "F_mean": d["F_mean"], "F_std": d.get("F_std")}
            for sp, d in sim_items
        ]
        fig = create_flatness_figure(
            datasets,
            style_config,
            show_std=show_std_band,
            show_error_bars=show_error_bars,
            show_reference=True,
            axis_labels=axis_labels,
            legend_names=sim_legend_map,
            apply_style=True,
        )
        if fig is None:
            return "Error: No valid flatness data to plot."
        session_context["last_figure"] = fig
        session_context.setdefault("figure_queue", []).append(fig)
        return {
            "status": "success",
            "message": "Flatness figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }
    
    if name == "get_flatness_summary":
        ref = args.get("data_reference") or CACHE_KEY_FLATNESS
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No flatness data. Run compute_flatness first."
        
        if not cached:
            return "Error: No flatness data. Run compute_flatness first."

        if "simulations" in cached:
            sim_items = list(sorted(cached["simulations"].items()))
        else:
            sim_items = [("default", dict(cached))]

        def _default_labelify(s: str) -> str:
            return s.replace("_", " ").title()

        simulation_legend_names = args.get("simulation_legend_names") or {}
        if isinstance(simulation_legend_names, dict):
            session_context.setdefault("flatness_legend_names", {}).update(simulation_legend_names)
        sim_legends = session_context.get("flatness_legend_names") or {}

        table_data = []
        headers = ["Simulation", "Snapshots Used", "Mean F(r)", "Std(F(r))", "Min F(r)", "Max F(r)"]

        for sim_prefix, data in sim_items:
            F_mean_values = np.array(data["F_mean"])
            num_snapshots = len(data.get("files", []))

            if F_mean_values.size > 0:
                mean_F = np.mean(F_mean_values)
                std_F = np.std(F_mean_values)
                min_F = np.min(F_mean_values)
                max_F = np.max(F_mean_values)
            else:
                mean_F, std_F, min_F, max_F = np.nan, np.nan, np.nan, np.nan

            display_name = simulation_legend_names.get(sim_prefix) or sim_legends.get(sim_prefix) or _default_labelify(sim_prefix)
            table_data.append([
                display_name,
                num_snapshots,
                f"{mean_F:.4f}",
                f"{std_F:.4f}",
                f"{min_F:.4f}",
                f"{max_F:.4f}",
            ])

        table_str = _format_markdown_table(headers, table_data)

        # Store for add_report_section when table_data not provided (convention: last_table_summary_rows)
        session_context["last_table_summary_rows"] = [
            dict(zip(headers, row)) for row in table_data
        ]

        return {
            "status": "success",
            "message": "Flatness summary table created.",
            "artifact_type": "markdown",
            "artifact_content": table_str,
            "artifact_title": "Flatness — Summary",
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "export_flatness_data":
        import pandas as pd

        ref = args.get("data_reference") or CACHE_KEY_FLATNESS
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No flatness data. Run compute_flatness first."

        export_format = (args.get("format") or "flatness").strip().lower()
        fname = args.get("filename") or ("flatness_summary.csv" if export_format == "summary" else "flatness_export.csv")
        if not fname.lower().endswith(".csv"):
            fname = f"{fname}.csv"
        try:
            if export_format == "summary":
                if "simulations" in cached:
                    sim_items = list(sorted(cached["simulations"].items()))
                else:
                    sim_items = [("default", dict(cached))]
                rows = []
                for sim_prefix, data in sim_items:
                    F_mean = np.asarray(data.get("F_mean", []))
                    files = data.get("files", [])
                    n_snap = len(files) if files else 0
                    if len(F_mean) == 0:
                        continue
                    rows.append({
                        "Simulation": sim_prefix,
                        "Snapshots used": n_snap,
                        "Mean F(r)": float(np.nanmean(F_mean)),
                        "Std(F(r))": float(np.nanstd(F_mean)),
                        "Min F(r)": float(np.nanmin(F_mean)),
                        "Max F(r)": float(np.nanmax(F_mean)),
                    })
                df = pd.DataFrame(rows)
            else:
                if "simulations" in cached:
                    sim_items = list(sorted(cached["simulations"].items()))
                    rows = []
                    for sim_prefix, data in sim_items:
                        r = np.asarray(data.get("r", []))
                        F_mean = np.asarray(data.get("F_mean", []))
                        F_std_raw = data.get("F_std")
                        n = len(r)
                        for i in range(n):
                            row = {"simulation": sim_prefix, "r": float(r[i]), "F_mean": float(F_mean[i]) if i < len(F_mean) else None}
                            if F_std_raw is not None and i < len(F_std_raw):
                                row["F_std"] = float(F_std_raw[i])
                            rows.append(row)
                    df = pd.DataFrame(rows)
                else:
                    r = cached.get("r", [])
                    F_mean = cached.get("F_mean", [])
                    F_std = cached.get("F_std") or []
                    n = len(r)
                    F_std_padded = [F_std[i] if i < len(F_std) else None for i in range(n)]
                    df = pd.DataFrame({"r": r, "F_mean": F_mean, "F_std": F_std_padded})
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
            return f"Error exporting flatness data: {e}"

    return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})