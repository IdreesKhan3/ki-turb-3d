"""
Structure functions agent tools: S_p(r), ESS, scaling exponents from structure_functions_*.txt / structure_funcs*_t*.bin.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core_physics import compute_structure_time_avg
from .._shared import (
    get_from_cache,
    resolve_data_dirs_and_group_structure_functions,
    save_to_cache,
    update_data_directory_in_context)
from ._meta import get_artifact_source_meta

CACHE_KEY_STRUCTURE = "current_structure_functions_data"


def _read_structure_bin(filepath: Path) -> Optional[Dict]:
    """Read binary structure function file."""
    try:
        from data_readers.binary_reader import read_structure_function_file
        return read_structure_function_file(str(filepath))
    except Exception:
        return None


def _read_structure_txt(filepath: Path) -> Optional[Dict]:
    """Read text structure function file (dict shape compatible with binary reader)."""
    try:
        from data_readers.text_reader import read_structure_function_txt
        data = read_structure_function_txt(str(filepath))
        if not isinstance(data, dict) or "r" not in data or "S_p" not in data:
            return None
        return data
    except Exception:
        return None


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for structure functions."""
    return [
        {
            "name": "compute_structure_functions",
            "description": "Compute time-averaged structure functions S_p(r) from structure_functions_*.txt or structure_funcs*_t*.bin. Data stored in cache for plot_structure_functions. Supports multi-simulation and time-window selection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Single directory path (optional: uses SESSION DATA PATH if not set)"},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple directories for multi-simulation comparison"},
                    "start_idx": {"type": "integer", "description": "1-based start file index for time window (default 1)"},
                    "end_idx": {"type": "integer", "description": "1-based end file index (default: last file)"},
                    "max_files": {"type": "integer", "description": "Max files per simulation group (default 1000)"},
                },
            },
        },
        {
            "name": "plot_structure_functions",
            "description": "Plot structure functions. Call compute_structure_functions first. mode: sp (S_p vs r) | ess (ESS) | anomalies (ξₚ − p/3). Supports multi-simulation and full style API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "Cache key (default: current_structure_functions_data)"},
                    "mode": {"type": "string", "description": "sp | ess | anomalies. Default: sp."},
                    "selected_ps": {"type": "array", "items": {"type": "integer"}, "description": "Orders p to plot (default: [1,2,3,4,5,6])"},
                    "ref_p": {"type": "integer", "description": "ESS reference order (default: 3)"},
                    "normalize_by_urms": {"type": "boolean", "description": "Normalize S_p by u_rms^p (default: true)"},
                    "error_display": {"type": "string", "description": "Shaded band | Error bars | Both | None. Default: Shaded band."},
                    "show_inset": {"type": "boolean", "description": "Show ESS inset when mode=ess (default: true)"},
                    "show_sl_theory": {"type": "boolean", "description": "Show She-Leveque theory (default: true)"},
                    "show_exp_anom": {"type": "boolean", "description": "Show experimental B93 anomalies (default: true)"},
                    "fit_rmin": {"type": "number", "description": "ESS fit range min r"},
                    "fit_rmax": {"type": "number", "description": "ESS fit range max r"},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API. Styles apply ONLY to the subplot being plotted (mode=sp|ess|anomalies). Per-simulation: per_sim_style_structure={sim_prefix: {enabled:true, color:\"#hex\", dash:\"solid\"|\"dot\"|\"dash\"|\"dashdot\", marker:\"circle\"|\"square\"|\"diamond\"|\"triangle-up\"|\"x\", msize:8}}. sim_prefix=dir name (\"256\",\"512\"). Use hex colors (e.g. #1f77b4). width 0.5–8, msize 0–18. Also: font_family, font_size, plot_bgcolor, paper_bgcolor, line_width, marker_size, palette, custom_colors, template, axis limits, figure size, margins. Legend position: legend_x, legend_y, legend_xanchor, legend_yanchor.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x_r\": \"r\", \"y_sp\": \"S_p(r)\", \"x_ess\": \"S_3(r)\", \"y_ess\": \"S_p(r)\", \"x_anom\": \"p\", \"y_anom\": \"ξₚ−p/3\"}. Partial OK."},
                    "simulation_legend_names": {"type": "object", "description": "Override legend names per simulation: {\"sim_prefix\": \"Display Name\"}. Partial OK."},
                    "x_limits": {"type": "object", "description": "X-axis limits: {\"min\": N, \"max\": M}. Overrides style_updates when user asks for x-axis range."},
                    "y_limits": {"type": "object", "description": "Y-axis limits: {\"min\": N, \"max\": M}. Overrides style_updates when user asks for y-axis range."},
                },
            },
        },
        {
            "name": "get_structure_functions_theory",
            "description": "Show Structure Functions page Theory & Equations: S_p(r), ESS, She-Leveque scaling. Use when user asks for 'structure functions theory', 'structure functions equations', 'theory for structure functions', 'She-Leveque equations'.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute structure functions tool."""
    session_context = session_context or {}

    if name == "get_structure_functions_theory":
        from content.structure_functions_theory_content import get_structure_functions_theory_markdown
        content = get_structure_functions_theory_markdown()
        return {
            "status": "success",
            "message": "Structure functions theory equations created.",
            "artifact_type": "markdown",
            "artifact_content": content,
            "artifact_title": "Structure Functions — Theory & Equations",
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "plot_structure_functions":
        from utils.plot_style import default_plot_style
        from visualizations.structure_functions_vis import (
            create_sp_figure,
            create_ess_figure,
            create_anomalies_figure,
        )

        ref = args.get("data_reference") or CACHE_KEY_STRUCTURE
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No structure functions data. Run compute_structure_functions first."

        sim_items = list(sorted(cached["simulations"].items()))
        if not sim_items:
            return "Error: No structure functions data to plot."

        def _default_labelify(s: str) -> str:
            return s.replace("_", " ").title()

        simulation_legend_names = args.get("simulation_legend_names") or {}
        if isinstance(simulation_legend_names, dict):
            session_context.setdefault("structure_sim_legend_names", {}).update(simulation_legend_names)
        sim_legends = session_context.get("structure_sim_legend_names") or {}
        legend_map = {k: simulation_legend_names.get(k) or sim_legends.get(k) or _default_labelify(k) for k, _ in sim_items}

        mode = (args.get("mode") or "sp").strip().lower()
        if mode not in ("sp", "ess", "anomalies"):
            mode = "sp"
        MODE_TO_PLOT = {"sp": "S_p(r) vs r", "ess": "ESS (S_p vs S_3)", "anomalies": "Anomalies (ξₚ − p/3)"}
        plot_name = MODE_TO_PLOT.get(mode, "S_p(r) vs r")

        struct_configs = session_context.setdefault("structure_style_configs", {})
        style_config = struct_configs.get(plot_name)
        if style_config is None:
            style_config = default_plot_style()
            # Anomalies: linear axes (values can be negative). S_p/ESS: log scale.
            axis_types = ("linear", "linear") if mode == "anomalies" else ("log", "log")
            style_config.update({
                "x_axis_type": axis_types[0], "y_axis_type": axis_types[1],
                "line_width": 2.4, "marker_size": 6,
                "std_alpha": 0.18, "per_sim_style_structure": {},
                "she_leveque_color": "#000000", "experimental_b93_color": "#00BFC4",
            })
            struct_configs[plot_name] = style_config
        elif mode == "anomalies":
            # Linear axes required (anomalies can be negative; log invalid)
            style_config["x_axis_type"] = "linear"
            style_config["y_axis_type"] = "linear"
        session_context["structure_style_config"] = style_config  # backward compat

        style_updates = args.get("style_updates") or {}
        if style_updates:
            for k, v in style_updates.items():
                if k == "per_sim_style_structure" and isinstance(v, dict):
                    style_config.setdefault(k, {})
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            style_config[k].setdefault(sk, {}).update(sv)
                        else:
                            style_config[k][sk] = sv
                    # Map short keys to full cache keys for per-simulation style matching
                    for cache_key, _ in sim_items:
                        for short_key, short_style in list(v.items()):
                            if isinstance(short_style, dict) and (
                                cache_key == short_key or cache_key.startswith(short_key + "_")
                            ):
                                style_config[k].setdefault(cache_key, {}).update(short_style.copy())
                                break
                else:
                    style_config[k] = v
            if "custom_colors" in style_updates:
                style_config["palette"] = "Custom"
            if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                style_config["enable_custom_size"] = True
            if "per_sim_style_structure" in style_updates:
                style_config["enable_per_sim_style"] = True
        y_limits = args.get("y_limits") or (style_updates.get("y_limits") if style_updates else None)
        x_limits = args.get("x_limits") or (style_updates.get("x_limits") if style_updates else None)
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

        axis_labels_raw = session_context.get("axis_labels_structure") or {}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            axis_labels_raw = dict(axis_labels_raw)
            axis_labels_raw.update(agent_axis)
            session_context["axis_labels_structure"] = axis_labels_raw
        axis_labels = {
            "x": axis_labels_raw.get("x_r", "Separation distance r"),
            "y": axis_labels_raw.get("y_sp", "Structure functions S<sub>p</sub>(r)"),
            "x_ess": axis_labels_raw.get("x_ess", "S<sub>3</sub>(r)"),
            "y_ess": axis_labels_raw.get("y_ess", "S<sub>p</sub>(r)"),
            "x_inset": axis_labels_raw.get("x_inset", "p"),
            "y_inset": axis_labels_raw.get("y_inset", "ξ<sub>p</sub> - p/3"),
            "inset_legend_sl": axis_labels_raw.get("inset_legend_sl", "SL94"),
            "inset_legend_b93": axis_labels_raw.get("inset_legend_b93", "B93"),
            "x_anom": axis_labels_raw.get("x_anom", "p"),
            "y_anom": axis_labels_raw.get("y_anom", "ξ<sub>p</sub> - p/3"),
        }

        error_display = (args.get("error_display") or "Shaded band").strip()
        if error_display not in ("Shaded band", "Error bars", "Both", "None"):
            error_display = "Shaded band"
        show_std = error_display in ("Shaded band", "Both")
        show_error_bars = error_display in ("Error bars", "Both")

        selected_ps = args.get("selected_ps")
        if selected_ps is None:
            selected_ps = sorted(sim_items[0][1].get("ps", [1, 2, 3, 4, 5, 6]))
        ref_p = int(args.get("ref_p", 3))
        normalize_by_urms = args.get("normalize_by_urms", True)
        show_inset = args.get("show_inset", True)
        show_sl_theory = args.get("show_sl_theory", True)
        show_exp_anom = args.get("show_exp_anom", True)
        fit_rmin = args.get("fit_rmin")
        fit_rmax = args.get("fit_rmax")

        datasets = [{"sim_prefix": sp, **d} for sp, d in sim_items]

        if mode == "sp":
            fig = create_sp_figure(
                datasets, style_config,
                selected_ps=selected_ps,
                normalize_by_urms=normalize_by_urms,
                show_std=show_std,
                show_error_bars=show_error_bars,
                axis_labels={"x": axis_labels["x"], "y": axis_labels["y"]},
                legend_names=legend_map,
                apply_style=True,
            )
        elif mode == "ess":
            fig = create_ess_figure(
                datasets, style_config,
                ref_p=ref_p,
                selected_ps=selected_ps,
                normalize_by_urms=normalize_by_urms,
                show_std=show_std,
                show_error_bars=show_error_bars,
                fit_rmin=fit_rmin,
                fit_rmax=fit_rmax,
                show_inset=show_inset,
                show_sl_theory=show_sl_theory,
                show_exp_anom=show_exp_anom,
                axis_labels=axis_labels,
                legend_names=legend_map,
                apply_style=True,
            )
        else:
            # Standalone anomalies subplot (style_config has linear axes enforced above)
            fig = create_anomalies_figure(
                datasets, style_config,
                ref_p=ref_p,
                selected_ps=selected_ps,
                normalize_by_urms=normalize_by_urms,
                fit_rmin=fit_rmin,
                fit_rmax=fit_rmax,
                show_sl_theory=show_sl_theory,
                show_exp_anom=show_exp_anom,
                axis_labels={"x": axis_labels.get("x_anom", "p"), "y": axis_labels.get("y_anom", "ξ<sub>p</sub> - p/3")},
                legend_names=legend_map,
                apply_style=True,
            )

        if fig is None:
            return "Error: No valid structure functions data to plot."
        session_context["last_figure"] = fig
        session_context.setdefault("figure_queue", []).append(fig)
        session_context["structure_style_configs"] = struct_configs
        session_context["structure_plotted_mode"] = mode
        return {
            "status": "success",
            "message": f"Structure functions ({mode}) figure created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name != "compute_structure_functions":
        return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})

    data_dir = args.get("data_dir", "")
    data_dirs = args.get("data_directories") or []
    start_idx = max(1, int(args.get("start_idx", 1)))
    end_idx = args.get("end_idx")
    max_files = int(args.get("max_files", 1000))

    groups = resolve_data_dirs_and_group_structure_functions(
        data_dirs=data_dirs if data_dirs else None,
        data_dir=data_dir,
        project_root=project_root,
        session_context=session_context,
        max_files_per_group=max_files,
    )

    if not groups:
        return json.dumps({
            "status": "error",
            "message": (
                "No structure_functions*.txt or structure_funcs*_t*.bin found in the "
                "active session/job (check processed/structure_functions/). "
                "Load a dataset manifest or set the Data directory first."
            ),
        })

    results_by_sim: Dict[str, Dict[str, Any]] = {}
    actual_start, actual_end = start_idx, None
    for sim_prefix, (files, kind) in groups.items():
        n = len(files)
        end = int(end_idx) if end_idx is not None else n
        end = min(max(end, start_idx), n)
        start = min(start_idx, end)
        if actual_end is None:
            actual_start, actual_end = start, end
        selected = files[start - 1 : end]
        if not selected:
            continue

        data_list = []
        read_fn = _read_structure_bin if kind == "bin" else _read_structure_txt
        for f in selected:
            d = read_fn(Path(f) if not isinstance(f, Path) else f)
            if d is not None:
                data_list.append(d)

        if not data_list:
            continue

        r_mean, Sp_mean, Sp_std, urms, ps = compute_structure_time_avg(data_list)
        if r_mean is None:
            continue

        # Convert to JSON-serializable
        results_by_sim[sim_prefix] = {
            "r": r_mean.tolist(),
            "Sp_mean": {int(p): arr.tolist() for p, arr in Sp_mean.items()},
            "Sp_std": {int(p): arr.tolist() for p, arr in Sp_std.items()},
            "urms": float(urms),
            "ps": ps,
            "files": [str(f) for f in selected],
            "kind": kind,
        }

    if not results_by_sim:
        return json.dumps({"status": "error", "message": "Could not compute structure functions from files."})

    cache_data = {"simulations": results_by_sim}
    save_to_cache(session_context, CACHE_KEY_STRUCTURE, cache_data)
    session_context["structure_start_idx"] = actual_start
    session_context["structure_end_idx"] = actual_end
    dirs_used = []
    seen = set()
    for _prefix, sim_data in results_by_sim.items():
        files = sim_data.get("files", [])
        if files:
            d = str(Path(files[0]).resolve().parent) if not isinstance(files[0], Path) else str(Path(files[0]).parent.resolve())
            if d not in seen:
                seen.add(d)
                dirs_used.append(d)
    if dirs_used:
        update_data_directory_in_context(
            session_context,
            dirs_used[0],
            data_dirs_list=dirs_used if len(dirs_used) > 1 else None,
        )
    n_sims = len(results_by_sim)
    return json.dumps({
        "status": "success",
        "message": f"Computed structure functions from {n_sims} simulation(s). Data ready for plot_structure_functions.",
    })
