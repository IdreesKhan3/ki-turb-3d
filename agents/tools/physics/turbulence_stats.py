"""
Other Turbulence Stats agent tools: plot_turbulence_stats, get_turbulence_stats_summary.

Full parity with pages/OtherTurbStats: multi-trace custom plotting, plot options,
tables (latest values + time series). Matches manual page exactly.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .._shared import resolve_data_dir_and_find_files
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


def _load_all_dataframes(
    data_dirs: List[str],
    project_root: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Load turbulence_stats and eps_validation CSVs from data dirs.
    Mirrors pages/OtherTurbStats/file_loading.py logic (no Streamlit).
    Returns (all_dataframes, table_data).
    """
    from utils.file_detector import detect_simulation_files

    all_dataframes: Dict[str, pd.DataFrame] = {}
    table_data: Dict[str, Dict] = {}

    all_files: Dict[str, List] = {}
    for data_dir_path in data_dirs:
        p = Path(data_dir_path)
        if not p.is_absolute():
            p = (project_root / str(data_dir_path).lstrip("/")).resolve()
        if not p.exists() or not p.is_dir():
            alt = project_root / "examples" / str(data_dir_path).lstrip("/")
            if alt.exists() and alt.is_dir():
                p = alt
            else:
                continue
        dir_files = detect_simulation_files(str(p))
        for ft, flist in dir_files.items():
            all_files.setdefault(ft, []).extend(
                [str(f) if isinstance(f, Path) else f for f in flist]
            )

    # Turbulence stats (real_turb_stats)
    csv_files = all_files.get("real_turb_stats", [])
    resolved_dirs = []
    for dd in data_dirs:
        pd_ = Path(dd)
        if not pd_.is_absolute():
            pd_ = (project_root / str(dd).lstrip("/")).resolve()
        if not pd_.exists():
            pd_ = project_root / "examples" / str(dd).lstrip("/")
        if pd_.exists() and pd_.is_dir():
            resolved_dirs.append((str(pd_), Path(dd).name))
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
    if not eps_files and data_dirs:
        d = Path(data_dirs[0])
        if not d.is_absolute():
            d = (project_root / str(d).lstrip("/")).resolve()
        if not d.exists():
            d = project_root / "examples" / str(data_dirs[0]).lstrip("/")
        if d.exists() and d.is_dir():
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

    return all_dataframes, table_data


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


# Common column aliases (user/agent may say "iteration" but CSV has "iter")
_COLUMN_ALIASES = {
    "iteration": ["iter", "iter_norm", "t", "time"],
    "iterations": ["iter", "iter_norm", "t", "time"],
    "time": ["iter_norm", "iter", "t"],
    "t": ["iter", "iter_norm", "time"],
    "spectral_dissipation": ["eps_spectral"],
    "spectral dissipation": ["eps_spectral"],
    "real_dissipation": ["eps_real"],
    "real dissipation": ["eps_real"],
}


def _resolve_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Resolve column name: exact match, or alias. Returns actual column name or None."""
    if not name:
        return None
    name = str(name).strip().lower()
    if name in df.columns:
        return name
    for col in df.columns:
        if col.lower() == name:
            return col
    aliases = _COLUMN_ALIASES.get(name, [name])
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _sensible_default_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    numeric = _get_numeric_columns(df)
    if len(numeric) < 2:
        return (None, None)
    x_candidates = ["iter_norm", "iter", "t", "time"]
    x_col = next((c for c in x_candidates if c in df.columns), numeric[0])
    y_candidates = ["energy_balance_ratio", "TKE_real", "eps_real", "eps_spectral", "frac_x", "frac_y", "frac_z"]
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if y_col is None:
        remaining = [c for c in numeric if c != x_col]
        y_col = remaining[0] if remaining else numeric[1]
    return (x_col, y_col)


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for Other Turbulence Stats (full page parity)."""
    return [
        {
            "name": "plot_turbulence_stats",
            "description": "Create custom multi-trace x-y plot from turbulence_stats*.csv and eps_real_validation*.csv (Other Turbulence Stats page). Supports Add Traces, Plot Options (moving average, normalize x/y, use absolute value). Use when user asks for 'other stats plot', 'turbulence stats plot', 'custom plot', 'add trace', 'energy balance', 'time series'.",
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
                                "x_col": {"type": "string", "description": "X-axis column (iter, iter_norm, t)"},
                                "y_col": {"type": "string", "description": "Y-axis column (TKE_real, energy_balance_ratio, frac_x, etc.)"},
                                "label": {"type": "string", "description": "Trace label for legend"},
                            },
                            "required": ["data_source", "x_col", "y_col"],
                        },
                        "description": "Array of traces. Each: {data_source, x_col, y_col, label}. For single trace, can use x_col/y_col/label at top level instead.",
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
                    "style_updates": {"type": "object", "description": "Plot style: font_family, font_size, plot_bgcolor, paper_bgcolor, line_width, palette, template, x_axis_type, y_axis_type, enable_custom_size, figure_width, figure_height, etc."},
                },
            },
        },
        {
            "name": "get_turbulence_stats_summary",
            "description": "Show turbulence statistics: latest values and optionally full time series table from turbulence_stats*.csv or eps_real_validation*.csv (Other Turbulence Stats page). Use when user asks for 'turbulence stats table', 'turbulence stats summary', 'other stats table', 'latest values', 'time series data'.",
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
    sim_groups = {k: [] for k in all_dataframes.keys()}
    from utils.plot_style import ensure_per_sim_defaults
    ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style_turb_stats", include_marker=True)
    colors = _get_palette(ps)
    fig = go.Figure()
    all_x_labels = set()
    all_y_labels = set()

    for idx, trace in enumerate(traces):
        data_source = trace.get("data_source")
        x_col = trace.get("x_col")
        y_col = trace.get("y_col")
        if not data_source or not x_col or not y_col:
            continue
        if data_source not in all_dataframes:
            continue
        df = all_dataframes[data_source]
        x_resolved = _resolve_column(df, x_col) if x_col else None
        y_resolved = _resolve_column(df, y_col) if y_col else None
        if not x_resolved or not y_resolved:
            continue
        x_col, y_col = x_resolved, y_resolved
        trace_key = f"{data_source}_{x_col}_{y_col}"
        label = legend_names.get(trace_key, trace.get("label", f"{data_source.split('_')[-1]}: {y_col}"))

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
            data_source, idx, colors, ps, style_key="per_sim_style_turb_stats", include_marker=True
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
        return None

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
    return fig


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute Other Turbulence Stats tool."""
    session_context = session_context or {}
    project_root = project_root or Path(".")

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
        all_dataframes, _ = _load_all_dataframes(dirs, project_root)
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

        fig = _build_multi_trace_plot(
            all_dataframes, traces, legend_names, axis_labels,
            use_abs, smooth_window, normalize_x, x_norm, normalize_y, style_updates,
        )
        if fig is None:
            avail = ", ".join(all_dataframes.keys())
            sample_cols = []
            for k, d in list(all_dataframes.items())[:2]:
                nc = _get_numeric_columns(d)
                sample_cols.append(f"{k}: {nc[:8]}{'...' if len(nc) > 8 else ''}")
            return (
                f"Error: No valid traces to plot. "
                f"Available data_sources: {avail}. "
                f"Sample columns: {'; '.join(sample_cols)}. "
                f"Use data_source from the list. For x_col/y_col use actual column names (e.g. iter not iteration, eps_spectral, eps_real)."
            )

        from pages.AutonomousLab.session_sync import update_data_directory_in_context
        update_data_directory_in_context(session_context, Path(dirs[0]).resolve())
        session_context["last_figure"] = fig
        session_context.setdefault("figure_queue", []).append(fig)
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
        all_dataframes, table_data = _load_all_dataframes(dirs, project_root)
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
        from pages.AutonomousLab.session_sync import update_data_directory_in_context
        update_data_directory_in_context(session_context, Path(dirs[0]).resolve())
        return {
            "status": "success",
            "message": "Turbulence stats summary created.",
            "artifact_type": "markdown_table",
            "artifact_content": table_md,
            "artifact_title": "Turbulence Stats — Latest Values",
            **get_artifact_source_meta(__file__, project_root, name),
        }

    return f"Error: Unknown tool '{name}'"
