"""
Spectra agent tools: compute_spectra, plot_spectrum.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core_physics import compute_spectrum_time_avg

from .._shared import get_from_cache, resolve_data_dir_and_find_files, save_to_cache
from .._shared import _natural_sort_key
from ._meta import get_artifact_source_meta


CACHE_KEY_SPECTRA = "current_spectra_data"
CACHE_KEY_SPECTRA_NORM = "current_spectra_norm"
CACHE_KEY_SPECTRA_EVOLUTION = "current_spectra_evolution"


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for spectra."""
    return [
        {
            "name": "compute_spectra",
            "description": "Compute spectra for plotting. Data stored in cache. Modes: raw (time-averaged from spectrum*.dat), normalized (from norm*.dat with Pope), evolution (individual curves over time).",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Directory path (spectrum*.dat for raw/evolution, norm*.dat for normalized)"},
                    "files": {"type": "array", "description": "Explicit file paths (raw mode only)"},
                    "mode": {"type": "string", "description": "raw | normalized | evolution"},
                    "start_idx": {"type": "integer", "description": "First file index (1-based). Use for '10 or 20 files'."},
                    "end_idx": {"type": "integer", "description": "Last file index (1-based). Omit for all."},
                    "every_n": {"type": "integer", "description": "For evolution: show every Nth curve (default 5)"},
                    "max_files": {"type": "integer", "description": "Max files to use (default 100)"},
                },
            },
        },
        {
            "name": "plot_spectrum",
            "description": "Create energy spectrum figure. Data from compute_spectra is cached—use data_reference. Use style_updates to modify plot appearance (LLM reasons about user requests and sets knobs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "Cache key: current_spectra_data (raw), current_spectra_norm (normalized), current_spectra_evolution (evolution). Do NOT pass k/E arrays."},
                    "mode": {"type": "string", "description": "raw | normalized | evolution (must match compute_spectra mode)"},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API (Energy Spectra sidebar). Set any keys. Fonts: font_family, font_size, title_size, legend_size, tick_font_size, axis_title_size, font_color. Backgrounds: plot_bgcolor, paper_bgcolor. Ticks: tick_len, tick_w, ticks_outside, tick_color. Axis: x_axis_type, y_axis_type, x_tick_format, y_tick_format. Borders: show_axis_lines, axis_line_width, axis_line_color, mirror_axes. Grid: show_grid, grid_on_x, grid_on_y, grid_w, grid_dash, grid_color, grid_opacity. Minor grid: show_minor_grid, minor_grid_w, minor_grid_dash, minor_grid_color, minor_grid_opacity. Curves: line_width, marker_size, line_dash, std_alpha. Colors: palette (Plotly|D3|G10|T10|Dark2|Set1|Custom), custom_colors, kolmogorov_color, pope_color, highlight_color. Theme: template (plotly_white|plotly_dark|simple_white). Legend: show_legend. Title: show_plot_title, plot_title. Limits: enable_x_limits, x_min, x_max, enable_y_limits, y_min, y_max. Size: enable_custom_size (must be true for width/height), figure_width, figure_height. Margins: margin_left, margin_top, margin_right, margin_bottom. Reference: show_kolmogorov, show_pope."
                    },
                    "k": {"type": "array", "description": "Wavenumber array (simple mode only)"},
                    "E": {"type": "array", "description": "Energy array (simple mode only)"},
                    "E_std": {"type": "array", "description": "Optional std for error bars"},
                    "datasets": {"type": "array", "description": "Full mode: [{sim_prefix, x, y, y_std, y_pope?}, ...]"},
                    "style_config": {"type": "object", "description": "Legacy: full style overrides (prefer style_updates)"},
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"...\", \"y\": \"...\"}. Partial OK: {\"y\": \"E(k)\"} updates y only. Mode-specific: raw/evolution→axis_labels_raw; normalized→axis_labels_norm. Same as Legend & Axis Labels sidebar."},
                    "legend_names": {"type": "object", "description": "Override legend trace names: {sim_prefix: display_name}. Mode-specific: raw→spectrum_legend_names; normalized→norm_legend_names. Evolution has no per-trace legend. Same as Legend & Axis Labels sidebar."},
                    "kolm_scale_data": {"type": "object", "description": "x,y for Kolmogorov line scaling"},
                    "kmin": {"type": "number", "description": "Kolmogorov inertial range min (default 3)"},
                    "kmax": {"type": "number", "description": "Kolmogorov inertial range max (default 20)"},
                    "kolm_scale_factor": {"type": "number", "description": "Scale Kolmogorov -5/3 line up (>1) or down (<1), default 1.0"},
                    "show_std": {"type": "boolean", "description": "Show shaded std band (Shaded band / Both)"},
                    "show_error_bars": {"type": "boolean", "description": "Show error bars (Error bars / Both)"},
                    "pope_scaling_prefix": {"type": "string", "description": "Show only this sim's Pope model; omit for all"},
                },
            },
        },
        {
            "name": "export_figure",
            "description": "Export the last plotted figure to PNG, PDF, SVG, or HTML. Call after plot_spectrum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "description": "png | pdf | svg | html (default png)"},
                    "filename": {"type": "string", "description": "Output filename without path (default: spectrum.png)"},
                },
            },
        },
        {
            "name": "export_data",
            "description": "Export cached spectrum data to CSV. Call after compute_spectra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {"type": "string", "description": "current_spectra_data | current_spectra_norm | current_spectra_evolution (default: current_spectra_data)"},
                    "filename": {"type": "string", "description": "Output filename (default: spectra_export.csv)"},
                },
            },
        },
    ]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Execute spectra tool. Returns result string or dict (for plot artifact)."""
    session_context = session_context or {}

    if name == "compute_spectra":
        mode = args.get("mode", "raw")
        start_idx = int(args.get("start_idx", 1))
        end_idx = args.get("end_idx")
        every_n = int(args.get("every_n", 5))
        max_files = int(args.get("max_files", 100))
        data_dir = args.get("data_dir", "")

        if mode == "normalized":
            from data_readers.norm_spectrum_reader import read_norm_spectrum_file
            from core_physics import compute_spectrum_time_avg_norm
            file_paths = resolve_data_dir_and_find_files(
                data_dir, "norm*.dat", project_root, session_context, max_files
            )
            files = [str(f) for f in file_paths]
            if not files:
                return json.dumps({"status": "error", "message": "No norm*.dat files found. Use data_dir with normalized spectrum files."})
            files = files[start_idx - 1:end_idx] if end_idx else files[start_idx - 1:]
            files = files[:max_files]
            data_list = []
            for f in files:
                try:
                    keta, En, Ep = read_norm_spectrum_file(f)
                    data_list.append((np.asarray(keta, float), np.asarray(En, float), np.asarray(Ep, float)))
                except Exception:
                    continue
            keta_vals, En_avg, En_std, Ep_avg = compute_spectrum_time_avg_norm(data_list)
            if keta_vals is None:
                return json.dumps({"status": "error", "message": "Could not compute normalized spectrum."})
            cached = {
                "mode": "normalized",
                "datasets": [{
                    "sim_prefix": "norm",
                    "x": keta_vals.tolist(),
                    "y": En_avg.tolist(),
                    "y_std": En_std.tolist() if En_std is not None else None,
                    "y_pope": Ep_avg.tolist(),
                }],
            }
            save_to_cache(session_context, CACHE_KEY_SPECTRA_NORM, cached)
            return json.dumps({
                "status": "success",
                "message": f"Computed normalized spectrum from {len(files)} files. Data ready for plot_spectrum.",
                "data_reference": CACHE_KEY_SPECTRA_NORM,
                "mode": "normalized",
                "files_used": len(files),
            })

        if mode == "evolution":
            from data_readers.spectrum_reader import read_spectrum_file
            file_paths = resolve_data_dir_and_find_files(
                data_dir, "spectrum*.dat", project_root, session_context, max_files
            )
            files = [str(f) for f in file_paths]
            if not files:
                return json.dumps({"status": "error", "message": "No spectrum*.dat found. Use find_file to locate data."})
            files = files[start_idx - 1:end_idx] if end_idx else files[start_idx - 1:]
            thin_idx = list(range(0, len(files), every_n))
            thin_files = [files[i] for i in thin_idx]
            thin_curves = []
            for f in thin_files:
                try:
                    k, E = read_spectrum_file(f)
                    thin_curves.append({"x": np.asarray(k, float).tolist(), "y": np.asarray(E, float).tolist()})
                except Exception:
                    continue
            if not thin_curves:
                return json.dumps({"status": "error", "message": "Could not read any spectrum files."})
            try:
                kH, EH = read_spectrum_file(thin_files[-1])
                highlight = {"x": np.asarray(kH, float).tolist(), "y": np.asarray(EH, float).tolist(),
                            "label": f"Highlighted (file {len(thin_files)} of {len(files)})"}
            except Exception:
                highlight = None
            cached = {"mode": "evolution", "thin_curves": thin_curves, "highlight": highlight}
            save_to_cache(session_context, CACHE_KEY_SPECTRA_EVOLUTION, cached)
            return json.dumps({
                "status": "success",
                "message": f"Prepared time evolution from {len(files)} files (showing every {every_n}th). Data ready for plot_spectrum.",
                "data_reference": CACHE_KEY_SPECTRA_EVOLUTION,
                "mode": "evolution",
                "curves_shown": len(thin_curves),
            })

        # raw (time-averaged)
        from data_readers.spectrum_reader import read_spectrum_file
        explicit_files = args.get("files")
        if explicit_files and isinstance(explicit_files, list):
            files = []
            for f in explicit_files:
                if not f:
                    continue
                p = Path(f)
                if not p.is_absolute():
                    p = (project_root / f).resolve()
                if p.exists() and (p.suffix == ".dat" or "spectrum" in p.name.lower()):
                    files.append(str(p))
            files = sorted(set(files), key=_natural_sort_key)
        else:
            file_paths = resolve_data_dir_and_find_files(
                data_dir, "spectrum*.dat", project_root, session_context, max_files
            )
            files = [str(f) for f in file_paths]
        if not files:
            return json.dumps({"status": "error", "message": "No spectrum*.dat found. Use find_file to locate data, or set data_dir."})
        files = files[start_idx - 1:end_idx] if end_idx else files[start_idx - 1:]
        files = files[:max_files]
        data_list = []
        for f in files:
            try:
                k, E = read_spectrum_file(f)
                data_list.append((np.asarray(k, float), np.asarray(E, float)))
            except Exception:
                continue
        k_vals, E_avg, E_std = compute_spectrum_time_avg(data_list)
        raw_data = {
            "mode": "raw",
            "k": k_vals.tolist() if k_vals is not None else [],
            "E": E_avg.tolist() if E_avg is not None else [],
            "E_std": E_std.tolist() if E_std is not None else [],
        }
        save_to_cache(session_context, CACHE_KEY_SPECTRA, raw_data)
        return json.dumps({
            "status": "success",
            "message": f"Computed raw time-averaged spectrum from {len(files)} files. Data ready for plot_spectrum.",
            "data_reference": CACHE_KEY_SPECTRA,
            "mode": "raw",
            "files_used": len(files),
            "n_points": len(raw_data["k"]),
        })

    if name == "plot_spectrum":
        from utils.plot_style import default_plot_style
        from visualizations.spectra_vis import (
            create_spectrum_figure,
            create_raw_spectrum_figure,
            create_normalized_spectrum_figure,
            create_time_evolution_figure,
        )
        # Persistence: read from spectra_options, merge with args, persist overrides
        opts = session_context.setdefault("spectra_options", {
            "show_std": True, "show_error_bars": True, "pope_scaling_prefix": None,
            "kmin": 3.0, "kmax": 20.0, "kolm_scale_factor": 1.0,
        })
        args = dict(args)
        for k in ("show_std", "show_error_bars", "pope_scaling_prefix", "kmin", "kmax", "kolm_scale_factor"):
            if args.get(k) is not None:
                opts[k] = args[k]
            elif k in opts:
                args[k] = opts[k]
        mode = args.get("mode", "raw")
        style_updates = args.get("style_updates") or {}
        plot_styles_refs = session_context.get("spectra_plot_styles") or {}
        if not plot_styles_refs:
            style_config = session_context.get("style_config") or session_context.get("spectra_style")
            if style_config is None:
                style_config = default_plot_style()
                session_context.setdefault("plot_styles", {})["Raw Energy Spectrum"] = style_config
                session_context["style_config"] = style_config
                session_context["spectra_style"] = style_config
            plot_styles_refs = {"Raw Energy Spectrum": style_config}
        if style_updates:
            for _name, ref in plot_styles_refs.items():
                if isinstance(ref, dict):
                    ref.update(style_updates)
                    if "custom_colors" in style_updates:
                        ref["palette"] = "Custom"
                    # Auto-enable custom size when agent sets figure dimensions
                    if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
                        ref["enable_custom_size"] = True
        style_config = plot_styles_refs.get("Raw Energy Spectrum") or (list(plot_styles_refs.values())[0] if plot_styles_refs else default_plot_style())
        axis_labels_raw = session_context.get("axis_labels_raw") or {}
        axis_labels_norm = session_context.get("axis_labels_norm") or {}
        agent_axis = args.get("axis_labels")
        if agent_axis and isinstance(agent_axis, dict):
            target = axis_labels_norm if mode == "normalized" else axis_labels_raw
            target.update(agent_axis)
            axis_labels = target
        else:
            axis_labels = args.get("axis_labels") or (axis_labels_norm if mode == "normalized" else axis_labels_raw)
        agent_legend = args.get("legend_names")
        if agent_legend and isinstance(agent_legend, dict):
            leg_key = "norm_legend_names" if mode == "normalized" else "spectrum_legend_names"
            target_leg = session_context.get(leg_key) or {}
            target_leg.update(agent_legend)
            session_context[leg_key] = target_leg

        ref_key = args.get("data_reference")
        if not ref_key:
            ref_key = (CACHE_KEY_SPECTRA_EVOLUTION if mode == "evolution" else
                      CACHE_KEY_SPECTRA_NORM if mode == "normalized" else CACHE_KEY_SPECTRA)
        cached = get_from_cache(session_context, ref_key)
        if not cached and mode != "raw":
            cached = get_from_cache(session_context, CACHE_KEY_SPECTRA)

        if cached and cached.get("mode") == "evolution":
            thin_curves = cached.get("thin_curves", [])
            highlight = cached.get("highlight")
            if not thin_curves:
                return "Error: No evolution data in cache. Run compute_spectra with mode='evolution' first."
            ps = default_plot_style()
            ps.update({"line_width": 2.4, "x_axis_type": "log", "y_axis_type": "log"})
            if style_config:
                ps.update(style_config)
            fig = create_time_evolution_figure(
                thin_curves, highlight, ps,
                axis_labels=axis_labels or {"x": "Wavenumber k", "y": "E(k)"},
                apply_style=True,
            )
            session_context["last_figure"] = fig
            return {
                "status": "success", "message": "Figure created successfully.",
                "artifact_type": "plotly_figure", "artifact_content": fig.to_json(),
                **get_artifact_source_meta(__file__, project_root, name),
            }

        if cached and cached.get("mode") == "normalized":
            datasets = cached.get("datasets", [])
            if not datasets:
                return "Error: No normalized data in cache. Run compute_spectra with mode='normalized' first."
            ps = default_plot_style()
            ps.update({"line_width": 2.4, "x_axis_type": "log", "y_axis_type": "log"})
            if style_config:
                ps.update(style_config)
            leg_names = session_context.get("norm_legend_names") or args.get("legend_names")
            fig = create_normalized_spectrum_figure(
                datasets, ps,
                show_std=args.get("show_std", True),
                show_error_bars=args.get("show_error_bars", True),
                pope_scaling_prefix=args.get("pope_scaling_prefix"),
                axis_labels=axis_labels or {"x": "Normalized wavenumber kη", "y": "Normalized spectrum E<sub>norm</sub>(kη)"},
                legend_names=leg_names,
                apply_style=True,
            )
            session_context["last_figure"] = fig
            return {
                "status": "success", "message": "Figure created successfully.",
                "artifact_type": "plotly_figure", "artifact_content": fig.to_json(),
                **get_artifact_source_meta(__file__, project_root, name),
            }

        if cached and isinstance(cached, dict) and (cached.get("k") or cached.get("E")):
            args = dict(args)
            if not args.get("k"):
                args["k"] = cached.get("k")
            if not args.get("E"):
                args["E"] = cached.get("E")
            if not args.get("E_std") and cached.get("E_std"):
                args["E_std"] = cached.get("E_std")

        def _spectra_ps():
            ps = default_plot_style()
            ps.update({
                "line_width": 2.4, "kolmogorov_color": "#666666",
                "x_axis_type": "log", "y_axis_type": "log",
            })
            if style_config:
                ps.update(style_config)
            return ps

        datasets = args.get("datasets")
        if datasets and isinstance(datasets, list):
            for d in datasets:
                if "x" not in d or "y" not in d:
                    return "Error: datasets must have 'x' and 'y' keys"
            ps = _spectra_ps()
            if mode == "normalized":
                leg_names = session_context.get("norm_legend_names") or args.get("legend_names")
                fig = create_normalized_spectrum_figure(
                    datasets, ps,
                    show_std=args.get("show_std", True),
                    show_error_bars=args.get("show_error_bars", True),
                    pope_scaling_prefix=args.get("pope_scaling_prefix"),
                    axis_labels=axis_labels,
                    legend_names=leg_names,
                    apply_style=True,
                )
            else:
                kolm_data = args.get("kolm_scale_data")
                leg_names = session_context.get("spectrum_legend_names") or args.get("legend_names")
                fig = create_raw_spectrum_figure(
                    datasets, ps,
                    show_std=args.get("show_std", True),
                    show_error_bars=args.get("show_error_bars", True),
                    show_kolmogorov=bool(kolm_data),
                    kmin=args.get("kmin", 3.0),
                    kmax=args.get("kmax", 20.0),
                    kolm_scale_factor=args.get("kolm_scale_factor", 1.0),
                    kolm_scale_data=kolm_data,
                    axis_labels=axis_labels,
                    legend_names=leg_names,
                    apply_style=True,
                )
            session_context["last_figure"] = fig
            return {
                "status": "success",
                "message": "Figure created successfully.",
                "artifact_type": "plotly_figure",
                "artifact_content": fig.to_json(),
                **get_artifact_source_meta(__file__, project_root, name),
            }

        k = args.get("k", [])
        E = args.get("E", [])
        if not k or not E:
            if ref_key and not cached:
                return "Error: No spectra data in cache. Ask the Analyst to run compute_spectra first."
            return "Error: k and E arrays required (or pass data_reference from compute_spectra)"

        def to_float_array(x):
            if isinstance(x, str):
                try:
                    x = json.loads(x)
                except json.JSONDecodeError:
                    parts = [p.strip() for p in x.replace("[", "").replace("]", "").split(",") if p.strip()]
                    x = [float(p) for p in parts]
            if hasattr(x, "__iter__") and not isinstance(x, (str, dict)):
                flat = []
                for v in x:
                    if isinstance(v, (list, tuple)):
                        flat.extend(float(w) for w in v if w is not None)
                    elif v is not None:
                        flat.append(float(v))
                return np.asarray(flat, dtype=float)
            return np.asarray(x, dtype=float)

        try:
            k = to_float_array(k)
            E = to_float_array(E)
            if len(k) == 0 or len(E) == 0:
                return "Error: k and E arrays are empty after conversion"
            if len(k) != len(E):
                n = min(len(k), len(E))
                k, E = k[:n], E[:n]
        except (ValueError, TypeError, json.JSONDecodeError):
            return "Error: k and E must be numeric arrays (lists of numbers)"
        valid = np.isfinite(k) & np.isfinite(E) & (k > 0) & (E > 0)
        if not np.any(valid):
            return "Error: No valid (k,E) points (need positive finite values)"
        k, E = k[valid], E[valid]
        E_std = args.get("E_std")
        if E_std is not None:
            E_std = to_float_array(E_std)
            if len(E_std) != len(E):
                E_std = None
        axis_labels = axis_labels or {}
        kmin = args.get("kmin", 3.0)
        kmax = args.get("kmax", 20.0)
        kolm_scale_factor = args.get("kolm_scale_factor", 1.0)
        show_kolm = style_config.get("show_kolmogorov", True)
        show_std = args.get("show_std", True)
        show_error_bars = args.get("show_error_bars", True)
        fig = create_spectrum_figure(
            k, E, E_std=E_std, style_config=style_config,
            show_kolmogorov=show_kolm,
            kmin=kmin, kmax=kmax, kolm_scale_factor=kolm_scale_factor,
            show_std=show_std, show_error_bars=show_error_bars,
            x_label=axis_labels.get("x", "Wavenumber k"),
            y_label=axis_labels.get("y", "Energy E(k)"),
        )
        session_context["last_figure"] = fig
        return {
            "status": "success",
            "message": "Figure created successfully.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "export_figure":
        import base64
        fig = session_context.get("last_figure")
        if fig is None:
            return "Error: No figure to export. Run plot_spectrum first."
        fmt = (args.get("format") or "png").lower()
        if fmt == "jpg":
            fmt = "jpeg"
        fname = args.get("filename") or f"spectrum.{fmt}"
        if not fname.lower().endswith(f".{fmt}"):
            fname = f"{fname}.{fmt}" if "." not in fname else fname
        try:
            if fmt == "html":
                buf = fig.to_html()
                content = buf.encode("utf-8")
                mime = "text/html"
            else:
                content = fig.to_image(format=fmt, scale=2)
                mime = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml", "jpeg": "image/jpeg"}.get(fmt, "application/octet-stream")
            # Save to project exports folder so user can find it
            out_dir = project_root / "exports"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / fname
            out_path.write_bytes(content)
            b64 = base64.b64encode(content).decode("ascii")
            rel_path = out_path.relative_to(project_root)
            return {
                "status": "success",
                "message": f"Figure saved to {rel_path}. Use the download button below for a copy.",
                "artifact_type": "downloadable_file",
                "filename": fname,
                "mime_type": mime,
                "content_base64": b64,
            }
        except Exception as e:
            return f"Error exporting figure: {e}. Install kaleido: pip install kaleido"

    if name == "export_data":
        import base64
        ref = args.get("data_reference") or CACHE_KEY_SPECTRA
        cached = get_from_cache(session_context, ref)
        if not cached:
            return "Error: No spectrum data in cache. Run compute_spectra first."
        fname = args.get("filename") or "spectra_export.csv"
        if not fname.lower().endswith(".csv"):
            fname = f"{fname}.csv"
        try:
            import pandas as pd
            if cached.get("mode") == "evolution":
                rows = []
                for i, c in enumerate(cached.get("thin_curves", [])):
                    x, y = c.get("x", []), c.get("y", [])
                    for j in range(min(len(x), len(y))):
                        rows.append({"curve": i, "k": x[j], "E": y[j]})
                df = pd.DataFrame(rows)
            elif cached.get("mode") == "normalized":
                ds = cached.get("datasets", [{}])[0]
                x, y = ds.get("x", []), ds.get("y", [])
                n = len(x)
                y_std = ds.get("y_std") or []
                y_pope = ds.get("y_pope") or []
                df = pd.DataFrame({
                    "k_eta": x,
                    "E_norm": y,
                    "E_norm_std": [y_std[i] if i < len(y_std) else None for i in range(n)],
                    "E_pope": [y_pope[i] if i < len(y_pope) else None for i in range(n)],
                })
            else:
                df = pd.DataFrame({
                    "k": cached.get("k", []),
                    "E": cached.get("E", []),
                    "E_std": cached.get("E_std") or [None] * len(cached.get("k", [])),
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

    return f"Error: Unknown spectra tool '{name}'"
