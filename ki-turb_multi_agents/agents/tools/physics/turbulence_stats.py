"""Agent tools for Other Turbulence Stats: plot_turbulence_stats, get_turbulence_stats_summary."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .._shared import resolve_data_dir_and_find_files, update_data_directory_in_context
from ._meta import get_artifact_source_meta

PATTERN_TURB_STATS = "turbulence_stats*.csv"
PATTERN_EPS_VALIDATION = "eps_real_validation*.csv"
PATTERN_TURB_VALIDATION = "turbulence_validation*.csv"


def _load_csv(filepath: Path) -> pd.DataFrame:
    """Load turbulence_stats or eps_validation CSV."""
    from data_readers.csv_reader import read_csv_data, read_eps_validation_csv

    if "eps_real_validation" in filepath.name or "turbulence_validation" in filepath.name:
        return read_eps_validation_csv(str(filepath))
    return read_csv_data(str(filepath))


def _resolve_dir_candidates(data_dir_path: str, project_root: Path) -> List[Path]:
    """Resolve directory path. Tries direct path, then project_root/examples/<path> when path does not exist."""
    s = str(data_dir_path).strip().lstrip("/")
    # Direct resolve
    p = (project_root / s).resolve()
    if p.exists() and p.is_dir():
        return [p]
    # Try examples/ prefix (e.g. DNS/512 -> examples/DNS/512)
    if not s.startswith("examples"):
        alt = (project_root / "examples" / s).resolve()
        if alt.exists() and alt.is_dir():
            return [alt]
    return []


def _load_all_dataframes(
    data_dirs: List[str],
    project_root: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict], List[Path]]:
    """
    Load turbulence_stats and eps_validation CSVs from data dirs.
    Mirrors pages/OtherTurbStats/file_loading.py logic (no Streamlit).
    Returns (all_dataframes, table_data).
    """
    from utils.file_detector import detect_simulation_files

    all_dataframes: Dict[str, pd.DataFrame] = {}
    table_data: Dict[str, Dict] = {}

    # Resolve data dirs
    resolved_paths: List[Path] = []
    for data_dir_path in data_dirs:
        for p in _resolve_dir_candidates(data_dir_path, project_root):
            if p not in resolved_paths:
                resolved_paths.append(p)
            break
        else:
            # No valid path found for this dir, skip
            pass

    all_files: Dict[str, List] = {}
    for p in resolved_paths:
        dir_files = detect_simulation_files(str(p))
        for ft, flist in dir_files.items():
            all_files.setdefault(ft, []).extend(
                [str(f) if isinstance(f, Path) else f for f in flist]
            )

    # Fallback: search project when no files found in given dirs
    if not all_files.get("real_turb_stats") and not all_files.get("spectral_turb_stats"):
        turb_found = list(project_root.rglob("turbulence_stats*.csv")) + list(
            project_root.rglob("eps_real_validation*.csv")
        ) + list(project_root.rglob("turbulence_validation*.csv"))
        if turb_found:
            # Use parent dirs of found files
            by_dir: Dict[Path, List[Path]] = {}
            for f in turb_found:
                by_dir.setdefault(f.parent, []).append(f)
            resolved_paths = list(by_dir.keys())
            all_files = {}
            for p in resolved_paths:
                dir_files = detect_simulation_files(str(p))
                for ft, flist in dir_files.items():
                    all_files.setdefault(ft, []).extend(
                        [str(f) if isinstance(f, Path) else f for f in flist]
                    )

    # Turbulence stats (real_turb_stats)
    csv_files = all_files.get("real_turb_stats", [])
    resolved_dirs = [(str(p), p.name) for p in resolved_paths]
    if csv_files:
        if len(resolved_dirs) > 1:
            for csv_file in csv_files:
                csv_path = Path(csv_file).resolve()
                dir_name = None
                for abs_path, name in resolved_dirs:
                    try:
                        csv_path.relative_to(Path(abs_path))
                        dir_name = name
                        break
                    except (ValueError, TypeError):
                        continue
                if not dir_name:
                    dir_name = csv_path.parent.name
                try:
                    df = _load_csv(csv_path)
                    key = f"turbulence_stats_{dir_name}"
                    all_dataframes[key] = df
                    table_data[key] = {"df": df, "dir_name": dir_name, "type": "turbulence_stats"}
                except Exception:
                    pass
        else:
            if len(csv_files) > 1:
                for idx, csv_file in enumerate(csv_files, 1):
                    csv_path = Path(csv_file)
                    suffix = csv_path.stem.replace("turbulence_stats", "").strip("_")
                    key_suffix = f"_{suffix}" if suffix else f"_{idx}"
                    try:
                        df = _load_csv(csv_path)
                        key = f"turbulence_stats{key_suffix}"
                        all_dataframes[key] = df
                        table_data[key] = {"df": df, "dir_name": csv_path.stem, "type": "turbulence_stats"}
                    except Exception:
                        pass
            else:
                try:
                    df = _load_csv(Path(csv_files[0]))
                    all_dataframes["turbulence_stats"] = df
                    table_data["turbulence_stats"] = {"df": df, "dir_name": None, "type": "turbulence_stats"}
                except Exception:
                    pass

    # Eps validation (spectral_turb_stats)
    eps_files = all_files.get("spectral_turb_stats", [])
    if not eps_files and resolved_paths:
        d = resolved_paths[0]
        eps_files = list(d.glob("eps_real_validation*.csv")) + list(d.glob("turbulence_validation*.csv"))
        eps_files = [str(f) for f in eps_files]

    if eps_files:
        if len(resolved_dirs) > 1:
            for eps_file in eps_files:
                eps_path = Path(eps_file).resolve()
                dir_name = None
                for abs_path, name in resolved_dirs:
                    try:
                        eps_path.relative_to(Path(abs_path))
                        dir_name = name
                        break
                    except (ValueError, TypeError):
                        continue
                if not dir_name:
                    dir_name = eps_path.parent.name
                try:
                    df = pd.read_csv(str(eps_file))
                    if "iter" in df.columns:
                        df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
                    key = f"eps_validation_{dir_name}"
                    all_dataframes[key] = df
                    table_data[key] = {"df": df, "dir_name": dir_name, "type": "eps_validation"}
                except Exception:
                    pass
        else:
            if len(eps_files) > 1:
                for idx, eps_file in enumerate(eps_files, 1):
                    eps_path = Path(eps_file)
                    suffix = eps_path.stem.replace("eps_real_validation", "").strip("_")
                    key_suffix = f"_{suffix}" if suffix else f"_{idx}"
                    try:
                        df = pd.read_csv(str(eps_file))
                        if "iter" in df.columns:
                            df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
                        key = f"eps_validation{key_suffix}"
                        all_dataframes[key] = df
                        table_data[key] = {"df": df, "dir_name": eps_path.stem, "type": "eps_validation"}
                    except Exception:
                        pass
            else:
                try:
                    df = pd.read_csv(str(eps_files[0]))
                    if "iter" in df.columns:
                        df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
                    all_dataframes["eps_validation"] = df
                    table_data["eps_validation"] = {"df": df, "dir_name": None, "type": "eps_validation"}
                except Exception:
                    pass

    return all_dataframes, table_data, resolved_paths


def _resolve_data_dirs(
    data_dir: str,
    data_directories: Optional[List[str]],
    csv_path: Optional[str],
    session_context: Dict[str, Any],
    project_root: Path,
) -> List[str]:
    """Resolve list of data directories to search. Accepts dir path, file path (uses parent), or session."""
    sess = session_context or {}

    def _to_dir(p: str) -> str:
        """If path is a file (ends with .csv), return parent directory."""
        s = str(p).strip()
        if s.lower().endswith(".csv"):
            return str(Path(s).parent)
        return s

    if data_directories and isinstance(data_directories, list) and len(data_directories) > 0:
        return [_to_dir(d) for d in data_directories if d and str(d).strip()]
    if data_dir and str(data_dir).strip():
        return [_to_dir(data_dir)]
    if csv_path and str(csv_path).strip():
        return [_to_dir(csv_path)]
    if sess.get("data_directories"):
        d = sess["data_directories"]
        return [_to_dir(x) for x in (list(d) if isinstance(d, list) else [d]) if x]
    if sess.get("data_directory"):
        return [_to_dir(sess["data_directory"])]
    return []


def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _resolve_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Resolve column name: exact or case-insensitive match. No aliases—use exact names from CSV."""
    if not name:
        return None
    name = str(name).strip()
    if name in df.columns:
        return name
    name_lower = name.lower()
    for col in df.columns:
        if col.lower() == name_lower:
            return col
    return None


def _sensible_default_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Pick first two numeric columns for x/y. Works with any CSV schema."""
    numeric = _get_numeric_columns(df)
    if len(numeric) < 2:
        return (None, None)
    return (numeric[0], numeric[1])


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for Other Turbulence Stats (full page parity)."""
    return [
        {
            "name": "plot_turbulence_stats",
            "description": "Create custom x-y plot from turbulence_stats*.csv and eps_real_validation*.csv (Other Turbulence Stats page). Works with any columns in the CSV—use exact column names from the file. Have steward read_file the CSV to get column names if needed. Use when user asks for plots, time series, turbulence stats, custom curves, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Directory path (e.g. examples/DNS/512). REQUIRED unless session has data_directory. If task says 'from path', pass it here. File path (e.g. .../file.csv) is OK—parent dir is used."},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple directories for multi-sim (e.g. DNS/512 and LES/64)"},
                    "csv_path": {"type": "string", "description": "Alternative: path to a specific CSV file. Its parent directory is used to find all CSVs."},
                    "traces": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "data_source": {"type": "string", "description": "Key: turbulence_stats, eps_validation, turbulence_stats_512, eps_validation_128, etc. (from loaded CSVs)"},
                                "x_col": {"type": "string", "description": "X-axis column. Use exact name from CSV."},
                                "y_col": {"type": "string", "description": "Y-axis column. Use exact name from CSV."},
                                "label": {"type": "string", "description": "Trace label for legend"},
                            },
                            "required": ["data_source", "x_col", "y_col"],
                        },
                        "description": "Array of traces. REQUIRED when user asks for 2+ curves. Each trace: {data_source, x_col, y_col, label}. Use exact column names from the CSV. Single trace: can use x_col/y_col/label at top level instead.",
                    },
                    "x_col": {"type": "string", "description": "Single-trace mode: X column. Omit if using traces array."},
                    "y_col": {"type": "string", "description": "Single-trace mode: Y column. Omit if using traces array."},
                    "label": {"type": "string", "description": "Single-trace mode: trace label."},
                    "use_abs": {"type": "boolean", "description": "Use absolute value (Y-axis). Default false."},
                    "smooth_window": {"type": "integer", "description": "Moving average window (0=off, 1-500). Default 0."},
                    "normalize_x": {"type": "boolean", "description": "Normalize X-axis. Default false."},
                    "x_norm": {"type": "number", "description": "X normalization constant when normalize_x. Default 1000."},
                    "normalize_y": {"type": "boolean", "description": "Normalize Y-axis by maximum. Default false."},
                    "legend_names": {"type": "object", "description": "Override legend: {trace_key: display_name}. trace_key = data_source_x_col_y_col."},
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"...\", \"y\": \"...\"}."},
                    "style_updates": {"type": "object", "description": "Plot style: font_family, font_size, plot_bgcolor, paper_bgcolor, line_width, palette, template, x_axis_type, y_axis_type, enable_custom_size, figure_width, figure_height. Per-trace colors/dash: per_sim_style_turb_stats with keys = data_source_x_col_y_col (e.g. turbulence_stats_iter_eps_spectral). Each: {enabled, color, dash, width, marker, msize}. Legend position: legend_x, legend_y, legend_xanchor, legend_yanchor (x,y: 0–1 inside, >1 outside; xanchor: left/center/auto; yanchor: top/middle/bottom/auto)."},
                },
            },
        },
        {
            "name": "get_turbulence_stats_columns",
            "description": "Discover available data sources and columns before plotting. Returns data_source -> list of numeric columns. Call this first when plotting turbulence stats to get exact column names for any CSV. Works with any file schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Directory path (e.g. examples/DNS/512). REQUIRED unless session has data_directory."},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple dirs for multi-sim"},
                    "csv_path": {"type": "string", "description": "Alternative: path to a CSV file; parent dir is used."},
                },
            },
        },
        {
            "name": "get_turbulence_stats_summary",
            "description": "Show turbulence statistics: latest values and optionally full time series table from turbulence_stats*.csv or eps_real_validation*.csv (Other Turbulence Stats page). Use when user asks for table, summary, latest values, time series data, statistics overview, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Directory path (e.g. examples/DNS/512). REQUIRED unless session has data_directory."},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple dirs for multi-sim"},
                    "csv_path": {"type": "string", "description": "Alternative: path to a CSV file; parent dir is used."},
                    "include_time_series": {"type": "boolean", "description": "Include full time series table (like manual page). Default false."},
                },
            },
        },
    ]


def _build_multi_trace_plot(
    all_dataframes: Dict[str, pd.DataFrame],
    traces: List[Dict[str, Any]],
    legend_names: Dict[str, str],
    axis_labels: Dict[str, str],
    use_abs: bool,
    smooth_window: int,
    normalize_x: bool,
    x_norm: float,
    normalize_y: bool,
    style_updates: Dict[str, Any],
) -> go.Figure:
    """Build multi-trace plot matching page _render_plot logic."""
    from utils.plot_style import (
        default_plot_style,
        _get_palette,
        resolve_line_style,
        apply_plot_style,
        apply_axis_limits,
        apply_figure_size,
        get_tick_format,
    )

    ps = default_plot_style()
    ps.update(style_updates or {})
    # Build sim_groups from trace_key (data_source_x_col_y_col) so multiple traces from same
    # data_source (e.g. eps_spectral and eps_real) get distinct per-trace styling
    resolved_traces: List[Tuple[str, str, str, str]] = []
    for trace in traces:
        ds = trace.get("data_source")
        xc = trace.get("x_col")
        yc = trace.get("y_col")
        if not ds or not xc or not yc or ds not in all_dataframes:
            continue
        df = all_dataframes[ds]
        xr = _resolve_column(df, xc) if xc else None
        yr = _resolve_column(df, yc) if yc else None
        if xr and yr:
            resolved_traces.append((ds, xr, yr, trace.get("label", "")))
    sim_groups = {f"{ds}_{xc}_{yc}": [] for ds, xc, yc, _ in resolved_traces}
    from utils.plot_style import ensure_per_sim_defaults
    ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style_turb_stats", include_marker=True)
    colors = _get_palette(ps)
    fig = go.Figure()
    all_x_labels = set()
    all_y_labels = set()

    for idx, (data_source, x_col, y_col, trace_label) in enumerate(resolved_traces):
        trace_key = f"{data_source}_{x_col}_{y_col}"
        label = legend_names.get(trace_key, trace_label or f"{data_source.split('_')[-1]}: {y_col}")

        df = all_dataframes[data_source]
        x_data = pd.to_numeric(df[x_col], errors="coerce").values
        y_data = pd.to_numeric(df[y_col], errors="coerce").values
        valid = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[valid]
        y_data = y_data[valid]
        if len(x_data) == 0 or len(y_data) == 0:
            continue

        if normalize_x:
            x_data = x_data / float(x_norm)
        if use_abs:
            y_data = np.abs(y_data)
        if normalize_y:
            y_max = np.max(np.abs(y_data)) if len(y_data) > 0 else 1.0
            if y_max > 0:
                y_data = y_data / y_max

        x_axis_type = ps.get("x_axis_type", "linear")
        y_axis_type = ps.get("y_axis_type", "linear")
        if x_axis_type == "log":
            m = x_data > 0
            x_data, y_data = x_data[m], y_data[m]
        if y_axis_type == "log":
            m = y_data > 0
            x_data, y_data = x_data[m], y_data[m]
        if len(x_data) == 0 or len(y_data) == 0:
            continue

        color, width, dash, marker, marker_size, override_on = resolve_line_style(
            trace_key, idx, colors, ps, style_key="per_sim_style_turb_stats", include_marker=True
        )
        line_style = dict(width=width, color=color)
        if dash and dash != "solid":
            line_style["dash"] = dash
        mode = "lines"
        marker_dict = None
        if override_on and marker and marker != "none":
            mode = "lines+markers"
            marker_dict = dict(symbol=marker, size=marker_size)

        if smooth_window > 1 and len(y_data) > smooth_window:
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=y_data, mode="lines",
                    name=f"{label} (original)", line=dict(width=1.0, color=color), opacity=0.3,
                    showlegend=False,
                )
            )
            kernel = np.ones(int(smooth_window)) / int(smooth_window)
            y_smooth = np.convolve(y_data, kernel, mode="valid")
            half = int(smooth_window) // 2
            x_smooth = x_data[half : half + len(y_smooth)]
            x_plot, y_plot = x_smooth, y_smooth
        else:
            x_plot, y_plot = x_data, y_data

        scatter_kw = dict(x=x_plot, y=y_plot, mode=mode, name=label, line=line_style)
        if marker_dict:
            scatter_kw["marker"] = marker_dict
        fig.add_trace(go.Scatter(**scatter_kw))
        all_x_labels.add(x_col)
        all_y_labels.add(y_col)

    if len(fig.data) == 0:
        return None, []

    resolved_trace_dicts = [
        {"data_source": ds, "x_col": xc, "y_col": yc, "label": lbl or f"{ds.split('_')[-1]}: {yc}"}
        for ds, xc, yc, lbl in resolved_traces
    ]

    custom_x = axis_labels.get("x", "X")
    custom_y = axis_labels.get("y", "Y")
    x_label = custom_x if custom_x != "X" else (list(all_x_labels)[0] if len(all_x_labels) == 1 else "X (multiple columns)")
    y_label = custom_y if custom_y != "Y" else (list(all_y_labels)[0] if len(all_y_labels) == 1 else "Y (multiple columns)")
    if normalize_x and x_norm:
        x_label = f"{x_label} / {x_norm}"
    if normalize_y:
        y_label = f"{y_label} / max"

    fig = apply_plot_style(fig, ps)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    layout_kw = dict(height=400, margin=dict(l=60, r=20, t=40, b=55))
    layout_kw = apply_axis_limits(layout_kw, ps)
    layout_kw = apply_figure_size(layout_kw, ps)
    fig.update_layout(**layout_kw)
    fig.update_layout(plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"), paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF"))
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    x_fmt = get_tick_format(ps.get("x_tick_format", "auto"), ps.get("x_tick_decimals", 2), normalize_x)
    y_fmt = get_tick_format(ps.get("y_tick_format", "auto"), ps.get("y_tick_decimals", 2), normalize_y)
    fig.update_xaxes(tickformat=x_fmt, separatethousands=False)
    fig.update_yaxes(tickformat=y_fmt, separatethousands=False)
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=ps["plot_title"])
    return fig, resolved_trace_dicts


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute Other Turbulence Stats tool."""
    session_context = session_context or {}
    project_root = project_root or Path(".")

    if name == "get_turbulence_stats_columns":
        data_dir = args.get("data_dir", "")
        data_directories = args.get("data_directories")
        csv_path = args.get("csv_path", "")
        dirs = _resolve_data_dirs(data_dir, data_directories, csv_path, session_context, project_root)
        if not dirs:
            return "Error: No data directory. Pass data_dir (e.g. examples/DNS/512) or csv_path."
        all_dataframes, _, _ = _load_all_dataframes(dirs, project_root)
        if not all_dataframes:
            return "Error: No turbulence_stats*.csv or eps_real_validation*.csv found in the given path(s)."
        schema = {k: list(_get_numeric_columns(d)) for k, d in all_dataframes.items()}
        return {"data_sources": list(schema.keys()), "columns": schema}

    if name == "plot_turbulence_stats":
        data_dir = args.get("data_dir", "")
        data_directories = args.get("data_directories")
        csv_path = args.get("csv_path", "")
        dirs = _resolve_data_dirs(data_dir, data_directories, csv_path, session_context, project_root)
        if not dirs:
            return (
                "Error: No data directory. Pass data_dir (e.g. examples/DNS/512) or csv_path. "
                "When the task says 'from DNS/512' or 'from examples/DNS/512/...', use that path as data_dir. "
                "Session data_directory is used only when user has loaded data."
            )
        all_dataframes, _, resolved_paths = _load_all_dataframes(dirs, project_root)
        if not all_dataframes:
            return "Error: No turbulence_stats*.csv or eps_real_validation*.csv found in the given path(s)."

        def _resolve_data_source(key: str) -> Optional[str]:
            """Resolve data_source: exact match, or first key starting with prefix."""
            if key in all_dataframes:
                return key
            for k in all_dataframes:
                if k.startswith(key) or key.startswith(k.split("_")[0]):
                    return k
            return None

        traces = args.get("traces")
        if not traces:
            x_col = args.get("x_col")
            y_col = args.get("y_col")
            ds = list(all_dataframes.keys())[0]
            df = all_dataframes[ds]
            if not x_col or not y_col:
                x_col, y_col = _sensible_default_columns(df)
            if not x_col or not y_col:
                return "Error: Could not determine x_col/y_col. Specify traces or x_col, y_col."
            traces = [{"data_source": ds, "x_col": x_col, "y_col": y_col, "label": args.get("label") or f"{ds}: {y_col}"}]
        else:
            resolved_traces = []
            for t in traces:
                ds = t.get("data_source")
                resolved = _resolve_data_source(ds) if ds else None
                if resolved:
                    t = dict(t)
                    t["data_source"] = resolved
                    resolved_traces.append(t)
            traces = resolved_traces

        legend_names = args.get("legend_names") or {}
        axis_labels = args.get("axis_labels") or {}
        use_abs = bool(args.get("use_abs", False))
        smooth_window = int(args.get("smooth_window", 0) or 0)
        normalize_x = bool(args.get("normalize_x", False))
        x_norm = float(args.get("x_norm", 1000.0) or 1000.0)
        normalize_y = bool(args.get("normalize_y", False))
        style_updates = args.get("style_updates") or {}

        # Read base style from session_context (like structure_functions) and merge
        turb_stats_plot_name = "Custom Multi-Trace Plot"
        turb_plot_styles = session_context.setdefault("turb_stats_plot_styles", {})
        base_style = turb_plot_styles.get(turb_stats_plot_name)
        if base_style is None:
            from utils.plot_style import default_plot_style
            base_style = default_plot_style()
            base_style.update({"line_width": 2.2, "x_axis_type": "linear", "y_axis_type": "linear"})
            turb_plot_styles[turb_stats_plot_name] = base_style
        merged_style = dict(base_style)
        for k, v in (style_updates or {}).items():
            if k == "per_sim_style_turb_stats" and isinstance(v, dict):
                merged_style.setdefault(k, {})
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        merged_style[k].setdefault(sk, {}).update(sv)
                    else:
                        merged_style[k][sk] = sv
            else:
                merged_style[k] = v
        if "custom_colors" in style_updates:
            merged_style["palette"] = "Custom"
        if ("figure_width" in style_updates or "figure_height" in style_updates) and "enable_custom_size" not in style_updates:
            merged_style["enable_custom_size"] = True
        if "per_sim_style_turb_stats" in style_updates:
            merged_style["enable_per_sim_style"] = True

        # Merge legend_names and axis_labels with context (agent args override)
        ctx_legends = session_context.get("custom_plot_legend_names") or {}
        ctx_axis = session_context.get("custom_plot_axis_labels") or {"x": "X", "y": "Y"}
        merged_legends = dict(ctx_legends)
        merged_legends.update(args.get("legend_names") or {})
        legend_names = merged_legends
        merged_axis = dict(ctx_axis)
        merged_axis.update(args.get("axis_labels") or {})
        axis_labels = merged_axis

        fig, resolved_trace_dicts = _build_multi_trace_plot(
            all_dataframes, traces, legend_names, axis_labels,
            use_abs, smooth_window, normalize_x, x_norm, normalize_y, merged_style,
        )
        if fig is None:
            schema = {k: list(_get_numeric_columns(d)) for k, d in all_dataframes.items()}
            return (
                f"Error: No valid traces. Call get_turbulence_stats_columns(data_dir=...) first, then use its exact data_sources and column names. "
                f"Available: {schema}"
            )

        resolved_dir_strs = [str(p) for p in resolved_paths]
        if resolved_dir_strs:
            update_data_directory_in_context(
                session_context,
                resolved_dir_strs[0],
                data_dirs_list=resolved_dir_strs if len(resolved_dir_strs) > 1 else None,
            )
        session_context["last_figure"] = fig
        session_context.setdefault("figure_queue", []).append(fig)

        # Write back to session_context so sync copies to manual page
        turb_plot_styles[turb_stats_plot_name] = merged_style
        session_context["turb_stats_plot_styles"] = turb_plot_styles
        session_context["custom_plot_legend_names"] = legend_names
        session_context["custom_plot_axis_labels"] = axis_labels
        session_context["custom_plot_traces"] = resolved_trace_dicts
        session_context["turb_stats_use_abs"] = use_abs
        session_context["turb_stats_smooth_window"] = smooth_window
        session_context["turb_stats_normalize_x"] = normalize_x
        session_context["turb_stats_x_norm"] = x_norm
        session_context["turb_stats_normalize_y"] = normalize_y

        return {
            "status": "success",
            "message": "Turbulence stats plot created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "get_turbulence_stats_summary":
        data_dir = args.get("data_dir", "")
        data_directories = args.get("data_directories")
        csv_path = args.get("csv_path", "")
        dirs = _resolve_data_dirs(data_dir, data_directories, csv_path, session_context, project_root)
        if not dirs:
            return (
                "Error: No data directory. Pass data_dir (e.g. examples/DNS/512) or csv_path. "
                "When the task says 'from DNS/512' or 'from examples/DNS/512/...', use that path as data_dir."
            )
        all_dataframes, table_data, resolved_paths = _load_all_dataframes(dirs, project_root)
        turb_tables = {k: v for k, v in table_data.items() if v.get("type") == "turbulence_stats"}
        if not turb_tables:
            turb_tables = table_data
        if not turb_tables:
            return "Error: No turbulence_stats*.csv or eps_real_validation*.csv found."

        parts = []
        for key, info in turb_tables.items():
            df = info.get("df")
            if df is None or len(df) == 0:
                continue
            dir_name = info.get("dir_name") or key
            latest = df.iloc[-1]
            rows = []
            for col in df.columns:
                val = latest[col]
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        rows.append(f"| {col} | {float(val):.6g} |")
                    except (TypeError, ValueError):
                        rows.append(f"| {col} | {val} |")
                else:
                    rows.append(f"| {col} | {val} |")
            header = "| Column | Latest Value |\n|--------|-------------|"
            parts.append(f"### {dir_name}\n\n{header}\n" + "\n".join(rows))

            if args.get("include_time_series"):
                ts_header = "| " + " | ".join(df.columns[:10]) + " |\n" + "|" + "---|" * min(10, len(df.columns))
                sample = df.head(100) if len(df) > 100 else df
                ts_rows = []
                for _, row in sample.iterrows():
                    ts_rows.append("| " + " | ".join(str(row[c])[:20] for c in df.columns[:10]) + " |")
                parts.append(f"\n**Time Series (first {len(sample)} rows):**\n\n" + ts_header + "\n" + "\n".join(ts_rows[:20]))

        table_md = "\n\n".join(parts) if parts else "No data."
        resolved_dir_strs = [str(p) for p in resolved_paths]
        if resolved_dir_strs:
            update_data_directory_in_context(
                session_context,
                resolved_dir_strs[0],
                data_dirs_list=resolved_dir_strs if len(resolved_dir_strs) > 1 else None,
            )
        return {
            "status": "success",
            "message": "Turbulence stats summary created.",
            "artifact_type": "markdown_table",
            "artifact_content": table_md,
            "artifact_title": "Turbulence Stats — Latest Values",
            **get_artifact_source_meta(__file__, project_root, name),
        }

    return f"Error: Unknown tool '{name}'"
