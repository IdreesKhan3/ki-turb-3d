"""
PDFs agent tools: plot_pdf from velocity fields (*.vti, *.h5, *.hdf5).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core_physics import (
    compute_velocity_magnitude_pdf,
    compute_velocity_pdf,
    compute_vorticity_pdf,
    compute_enstrophy_pdf,
    compute_dissipation_pdf,
    compute_velocity_dissipation_joint_pdf,
    compute_velocity_enstrophy_joint_pdf,
    compute_dissipation_enstrophy_joint_pdf,
    compute_rq_joint_pdf,
)
from .._shared import resolve_data_dir_and_find_files, _natural_sort_key
from ._meta import get_artifact_source_meta


PDF_TYPES = ["velocity_components", "velocity_magnitude", "vorticity", "enstrophy", "dissipation", "joint_velocity_dissipation", "joint_velocity_enstrophy", "joint_dissipation_enstrophy", "rq_joint"]

# Mapping to Legend & Axis Labels sidebar keys (same as pages/PDFs/pdfs_plot_style.py)
_PDF_TYPE_AXIS_KEYS = {
    "velocity_components": ("velocity_x", "velocity_y"),
    "velocity_magnitude": ("velocity_mag_x", "velocity_mag_y"),
    "vorticity": ("vorticity_x", "vorticity_y"),
    "enstrophy": ("enstrophy_x", "enstrophy_y"),
    "dissipation": ("dissipation_x", "dissipation_y"),
    "joint_velocity_dissipation": ("joint_ud_x", "joint_ud_y"),
    "joint_velocity_enstrophy": ("joint_uo_x", "joint_uo_y"),
    "joint_dissipation_enstrophy": ("joint_do_x", "joint_do_y"),
    "rq_joint": ("rq_x", "rq_y"),
}
_PDF_TYPE_LEGEND_KEY = {
    "velocity_components": "velocity_pdf",
    "velocity_magnitude": "velocity_mag_pdf",
    "vorticity": "vorticity_pdf",
    "enstrophy": "enstrophy_pdf",
    "dissipation": "dissipation_pdf",
    "joint_velocity_dissipation": "joint_ud_pdf",
    "joint_velocity_enstrophy": "joint_uo_pdf",
    "joint_dissipation_enstrophy": "joint_do_pdf",
    "rq_joint": "rq_pdf",
}
# Map pdf_type to sidebar plot name for session style lookup (must match pages/09_PDFs.py plot_names)
_PDF_TYPE_TO_PLOT_NAME = {
    "velocity_components": "Velocity PDF",
    "velocity_magnitude": "Velocity Magnitude PDF",
    "vorticity": "Vorticity PDF",
    "enstrophy": "Enstrophy PDF",
    "dissipation": "Dissipation PDF",
    "joint_velocity_dissipation": "Velocity-Dissipation Joint PDF",
    "joint_velocity_enstrophy": "Velocity-Enstrophy Joint PDF",
    "joint_dissipation_enstrophy": "Dissipation-Enstrophy Joint PDF",
    "rq_joint": "R-Q Topological Space",
}


def _find_velocity_files(
    data_dir: str,
    project_root: Path,
    session_context: Optional[Dict[str, Any]],
    max_files: int = 10,
) -> List[Path]:
    """Find *.vti, *.h5, or *.hdf5 files in data_dir."""
    all_files: List[Path] = []
    for pattern in ["*.vti", "*.VTI", "*.h5", "*.H5", "*.hdf5", "*.HDF5"]:
        files = resolve_data_dir_and_find_files(
            data_dir, pattern, project_root, session_context, max_files
        )
        all_files.extend(files)
    if not all_files:
        return []
    seen = set()
    unique = []
    for f in sorted(all_files, key=lambda x: _natural_sort_key(str(x))):
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            unique.append(f)
    return unique[:max_files]


def _resolve_velocity_file_groups(
    data_dirs: Optional[List[str]],
    data_dir: str,
    project_root: Path,
    session_context: Optional[Dict[str, Any]],
    max_files_per_group: int = 1,
    max_groups: int = 10,
) -> Dict[str, List[Path]]:
    """Resolve velocity files for multi-simulation comparison. Returns {sim_prefix: [Path, ...]}."""
    sess = session_context or {}
    dirs_to_search: List[str] = []
    if data_dirs and isinstance(data_dirs, list) and len(data_dirs) > 0:
        dirs_to_search = list(data_dirs)
    elif data_dir:
        dirs_to_search = [data_dir]
    elif sess.get("data_directories"):
        dirs_to_search = list(sess["data_directories"]) if isinstance(sess["data_directories"], list) else [sess["data_directories"]]
    elif sess.get("data_directory"):
        dirs_to_search = [sess["data_directory"]]

    groups: Dict[str, List[Path]] = {}
    multi_dir = len(dirs_to_search) > 1
    for search_dir in dirs_to_search[:max_groups]:
        if not search_dir:
            continue
        files = _find_velocity_files(search_dir, project_root, sess, max_files=max_files_per_group)
        if not files:
            continue
        p = Path(search_dir)
        if not p.is_absolute():
            p = (project_root / search_dir.lstrip("/")).resolve()
        if multi_dir:
            # One file per dir for cross-sim comparison
            sim_prefix = p.name
            if sim_prefix in groups:
                sim_prefix = f"{p.name}_{Path(files[0].name).stem}"
            groups[sim_prefix] = files[:1]
        else:
            # Single dir: each file as separate trace (per-sim style)
            for i, f in enumerate(files[:max_files_per_group]):
                sp = Path(f.name).stem
                if sp in groups:
                    sp = f"{sp}_{i}"
                groups[sp] = [f]

    if not groups and (data_dir or dirs_to_search):
        search = data_dir or (dirs_to_search[0] if dirs_to_search else "")
        files = _find_velocity_files(search, project_root, sess, max_files=max(5, max_files_per_group))
        if files:
            for i, f in enumerate(files[:5]):
                sp = Path(f.name).stem
                if sp in groups:
                    sp = f"{sp}_{i}"
                groups[sp] = [f]
    return groups


def _load_velocity(filepath: Path, fortran_order: bool = True) -> Optional[np.ndarray]:
    """Load velocity array (nx, ny, nz, 3) from VTI or HDF5."""
    path_str = str(filepath.resolve())
    path_lower = path_str.lower()
    try:
        if path_lower.endswith((".h5", ".hdf5")):
            from data_readers.hdf5_reader import read_hdf5_file
            data = read_hdf5_file(path_str, fortran_order=fortran_order)
        elif path_lower.endswith(".vti"):
            from data_readers.vti_reader import read_vti_file
            data = read_vti_file(path_str)
        else:
            return None
        velocity = data.get("velocity")
        if velocity is None or len(velocity.shape) != 4:
            return None
        return np.asarray(velocity, dtype=np.float64)
    except Exception:
        return None


def _get_dx_dy_dz(data_dir: Path, dx_override: Optional[float] = None) -> tuple:
    """Get grid spacing (dx, dy, dz). dx_override: use (dx,dx,dx). Else LBM (1,1,1) or NS from simulation.json."""
    if dx_override is not None:
        return (float(dx_override), float(dx_override), float(dx_override))
    try:
        from pages.PDFs.pdf_params import get_grid_spacing_options
        options = get_grid_spacing_options(data_dir)
        for label, (dx, dy, dz) in options.items():
            return (dx, dy, dz)
    except Exception:
        pass
    return (1.0, 1.0, 1.0)


def _get_nu_from_params(data_dir: Path) -> Optional[float]:
    """Read kinematic viscosity from simulation.input or simulation.json. Same as manual page."""
    try:
        from data_readers.parameter_reader import read_parameters
        for candidate in (data_dir / "simulation.input", data_dir / "simulation.json"):
            if candidate.exists():
                params = read_parameters(str(candidate))
                if "nu" in params:
                    return float(params["nu"])
    except Exception:
        pass
    return None


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for PDFs."""
    return [
        {
            "name": "plot_pdf",
            "description": "Plot probability density function (PDF) from velocity fields. Data: *.vti, *.h5, *.hdf5. pdf_type: velocity_components (u,v,w) | velocity_magnitude | vorticity | enstrophy | dissipation | joint_velocity_dissipation P(|u|,ε) | joint_velocity_enstrophy P(|u|,|ω|) | joint_dissipation_enstrophy P(ε,|ω|) | rq_joint (R-Q topological). Use when user asks for 'velocity pdf', 'vorticity pdf', 'enstrophy pdf', 'dissipation pdf', 'joint pdf', 'r-q', 'pdfs page', or 'probability density'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Directory path containing *.vti, *.h5, or *.hdf5 velocity files. Uses session data_directory if empty."},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple directories for multi-simulation comparison (one file per dir). Overrides data_dir when set."},
                    "file_paths": {"type": "array", "items": {"type": "string"}, "description": "Explicit file selection: filenames or paths (e.g. [\"Velocity_100000.vti\", \"Velocity_200000.vti\"]). Same as manual File Selection. Overrides auto-resolution when set."},
                    "max_files": {"type": "integer", "description": "Max files per simulation for comparison (default 5). Use 1 for one snapshot per sim. Ignored when file_paths is set."},
                    "pdf_type": {
                        "type": "string",
                        "description": "Type of PDF: velocity_components | velocity_magnitude | vorticity | enstrophy | dissipation | joint_velocity_dissipation | joint_velocity_enstrophy | joint_dissipation_enstrophy | rq_joint. Default: velocity_magnitude.",
                        "enum": PDF_TYPES,
                    },
                    "bins": {"type": "integer", "description": "Number of bins for PDF (default 100)"},
                    "normalize": {"type": "boolean", "description": "Normalize PDF by RMS/mean (default false)"},
                    "log_scale": {"type": "boolean", "description": "Log scale for joint PDFs P(|u|,ε), P(|u|,|ω|), P(ε,|ω|) (default true)"},
                    "log_scale_rq": {"type": "boolean", "description": "Log scale for R-Q topological space (default true)"},
                    "nu": {"type": "number", "description": "Kinematic viscosity for dissipation PDF. Auto from simulation.input/simulation.json when not set (default 0.004 if found, else 1.0). Override when user specifies."},
                    "dx": {"type": "number", "description": "Grid spacing override (dx=dy=dz). LBM: 1. Auto from simulation.json when not set. Same as Advanced (grid spacing) sidebar."},
                    "style_updates": {
                        "type": "object",
                        "description": "Full Plot Style API (matches PDFs sidebar): Fonts: font_family, font_size, title_size, legend_size, tick_font_size, axis_title_size, font_color. Backgrounds: plot_bgcolor, paper_bgcolor. Ticks: tick_len, tick_w, ticks_outside, tick_color. Axis: x_axis_type, y_axis_type, x_tick_format, y_tick_format. Borders: show_axis_lines, axis_line_width, axis_line_color, mirror_axes. Grid: show_grid, grid_on_x, grid_on_y, grid_w, grid_dash, grid_color, grid_opacity. Minor grid: show_minor_grid, minor_grid_*. Curves: line_width, marker_size. Colors: palette, custom_colors. Theme: template. Legend: show_legend. Title: show_plot_title, plot_title. Limits: enable_x_limits, x_min, x_max, enable_y_limits, y_min, y_max. Size: enable_custom_size, figure_width, figure_height. Margins: margin_left, margin_top, margin_right, margin_bottom. Per-sim: enable_per_sim_style, per_sim_style_comparison.",
                    },
                    "axis_labels": {"type": "object", "description": "Override axis labels: {\"x\": \"...\", \"y\": \"...\"}. Partial OK. Same as Legend & Axis Labels sidebar."},
                    "legend_names": {"type": "object", "description": "Override legend trace names: {filename_stem: display_name}. Same as Legend & Axis Labels sidebar."},
                    "simulation_legend_names": {"type": "object", "description": "Override legend names per simulation: {\"sim_prefix\": \"Display Name\"}. Partial OK. For multi-file comparison."},
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
    """Execute PDF tool."""
    session_context = session_context or {}

    if name != "plot_pdf":
        return f"Error: Unknown PDF tool '{name}'"

    data_dir = args.get("data_dir", "")
    data_dirs = args.get("data_directories") or []
    file_paths_arg = args.get("file_paths") or []
    pdf_type = args.get("pdf_type", "velocity_magnitude")
    # Fallback to manual-page file selections when file_paths not specified
    sess = session_context or {}
    _fallback_map = {
        "joint_velocity_dissipation": "pdfs_selected_files_ud",
        "joint_velocity_enstrophy": "pdfs_selected_files_uo",
        "joint_dissipation_enstrophy": "pdfs_selected_files_do",
        "rq_joint": "pdfs_selected_files_rq",
        "dissipation": "pdfs_selected_files_dissipation",
        "vorticity": "pdfs_selected_files_vorticity",
        "enstrophy": "pdfs_selected_files_enstrophy",
        "velocity_components": "pdfs_selected_files_velocity_components",
        "velocity_magnitude": "pdfs_selected_files_velocity_magnitude",
    }
    if not file_paths_arg and pdf_type in _fallback_map:
        sess_val = sess.get(_fallback_map[pdf_type])
        if sess_val and isinstance(sess_val, list):
            file_paths_arg = list(sess_val)

    def _write_back_pdf_files(sctx: dict, ptype: str, file_list: list) -> None:
        """Write chosen files to session_context for sync to manual page."""
        if not sctx or ptype not in _fallback_map:
            return
        sctx[_fallback_map[ptype]] = [Path(f).name for f in file_list]

    def _resolve_bins(ptype: str) -> int:
        if "bins" in args and args["bins"] is not None:
            return int(args["bins"])
        sess = session_context or {}
        if ptype == "dissipation":
            v = sess.get("pdfs_bins_dissipation")
        elif ptype == "rq_joint":
            v = sess.get("pdfs_bins_rq")
        elif ptype in ("vorticity", "enstrophy"):
            v = sess.get("pdfs_bins_vorticity")
        elif ptype == "velocity_components":
            v = sess.get("pdfs_bins_velocity_components")
        elif ptype == "velocity_magnitude":
            v = sess.get("pdfs_bins_velocity_magnitude")
        elif ptype in ("joint_velocity_dissipation", "joint_velocity_enstrophy", "joint_dissipation_enstrophy"):
            v = sess.get("pdfs_bins_joint")
        else:
            v = sess.get("pdfs_bins_joint")
        return int(v) if v is not None else 100

    def _resolve_normalize(ptype: str) -> bool:
        if "normalize" in args and args["normalize"] is not None:
            return bool(args["normalize"])
        sess = session_context or {}
        if ptype == "dissipation":
            v = sess.get("pdfs_normalize_dissipation")
        elif ptype in ("vorticity", "enstrophy"):
            v = sess.get("pdfs_normalize_vorticity")
        elif ptype == "velocity_components":
            v = sess.get("pdfs_normalize_velocity_components")
        elif ptype == "velocity_magnitude":
            v = sess.get("pdfs_normalize_velocity_magnitude")
        elif ptype in ("joint_velocity_dissipation", "joint_velocity_enstrophy", "joint_dissipation_enstrophy"):
            v = sess.get("pdfs_normalize_joint")
        else:
            v = sess.get("pdfs_normalize_joint")
        return bool(v) if v is not None else False

    def _write_back_bins_normalize(sctx: dict, ptype: str, bins_val: int, norm_val: bool) -> None:
        """Write bins and normalize to session_context for sync to manual page."""
        if not sctx:
            return
        if ptype == "dissipation":
            sctx["pdfs_bins_dissipation"] = bins_val
            sctx["pdfs_normalize_dissipation"] = norm_val
        elif ptype == "rq_joint":
            sctx["pdfs_bins_rq"] = bins_val
        elif ptype in ("vorticity", "enstrophy"):
            sctx["pdfs_bins_vorticity"] = bins_val
            sctx["pdfs_normalize_vorticity"] = norm_val
        elif ptype == "velocity_components":
            sctx["pdfs_bins_velocity_components"] = bins_val
            sctx["pdfs_normalize_velocity_components"] = norm_val
        elif ptype == "velocity_magnitude":
            sctx["pdfs_bins_velocity_magnitude"] = bins_val
            sctx["pdfs_normalize_velocity_magnitude"] = norm_val
        elif ptype in ("joint_velocity_dissipation", "joint_velocity_enstrophy", "joint_dissipation_enstrophy"):
            sctx["pdfs_bins_joint"] = bins_val
            sctx["pdfs_normalize_joint"] = norm_val
        else:
            sctx["pdfs_bins_joint"] = bins_val
            sctx["pdfs_normalize_joint"] = norm_val

    def _resolve_log_scale(ptype: str) -> bool:
        if ptype == "rq_joint":
            if "log_scale_rq" in args and args["log_scale_rq"] is not None:
                return bool(args["log_scale_rq"])
            v = (session_context or {}).get("pdfs_log_scale_rq")
            return bool(v) if v is not None else True
        if ptype in ("joint_velocity_dissipation", "joint_velocity_enstrophy", "joint_dissipation_enstrophy"):
            if "log_scale" in args and args["log_scale"] is not None:
                return bool(args["log_scale"])
            v = (session_context or {}).get("pdfs_log_scale_joint")
            return bool(v) if v is not None else True
        return True

    def _write_back_log_scale(sctx: dict, ptype: str, log_val: bool) -> None:
        """Write log scale options to session_context for sync to manual page."""
        if not sctx:
            return
        if ptype == "rq_joint":
            sctx["pdfs_log_scale_rq"] = log_val
        elif ptype in ("joint_velocity_dissipation", "joint_velocity_enstrophy", "joint_dissipation_enstrophy"):
            sctx["pdfs_log_scale_joint"] = log_val

    def _write_back_nu_dx(sctx: dict, nu_val: float, dx_val: float) -> None:
        """Write viscosity and grid spacing to session_context for sync to manual page."""
        if not sctx:
            return
        sctx["pdfs_nu"] = nu_val
        sctx["pdfs_dx_override"] = dx_val

    max_files = int(args.get("max_files", 5))
    bins = _resolve_bins(pdf_type)
    normalize = _resolve_normalize(pdf_type)
    use_log_scale = _resolve_log_scale(pdf_type)
    dx_override = args.get("dx")
    if dx_override is None:
        dx_override = (session_context or {}).get("pdfs_dx_override")
    fortran_order = session_context.get("hdf5_fortran_order", True)

    # Resolve files: explicit file_paths (like manual File Selection) or auto from data_dir
    if file_paths_arg and isinstance(file_paths_arg, list):
        # Explicit file selection — resolve paths relative to data_dir or project_root
        sess = session_context or {}
        base_dir = data_dir or sess.get("data_directory") or ""
        if not base_dir:
            return "Error: data_dir required when using file_paths."
        base = Path(base_dir)
        if not base.is_absolute():
            base = (project_root / base_dir.lstrip("/")).resolve()
        if not base.exists():
            base = project_root / "examples" / base_dir.lstrip("/")
        sim_items = []
        for fp in file_paths_arg:
            p = Path(fp)
            if p.is_absolute() and p.exists():
                f = p
            else:
                f = (base / p.name) if (base / p.name).exists() else (base / fp)
            if not f.exists():
                f = project_root / fp
            if f.exists() and f.is_file():
                stem = f.stem
                if any(sp == stem for sp, _ in sim_items):
                    stem = f"{stem}_{len(sim_items)}"
                sim_items.append((stem, f))
        if not sim_items:
            return f"Error: No valid files from file_paths. Check paths relative to data_dir={base_dir}."
    else:
        # Auto-resolve from data_dir / data_directories
        file_groups = _resolve_velocity_file_groups(
            data_dirs if data_dirs else None,
            data_dir,
            project_root,
            session_context,
            max_files_per_group=1 if data_dirs else max_files,
            max_groups=10,
        )
        if not file_groups:
            return (
                "Error: No velocity files (*.vti, *.h5, *.hdf5) found. "
                "Use data_dir with a path containing velocity fields, or set Data directory in sidebar."
            )
        sim_items = [(sp, flist[0] if flist else None) for sp, flist in sorted(file_groups.items()) if flist]
        sim_items = [(sp, f) for sp, f in sim_items if f is not None]
        if not sim_items:
            return "Error: No valid velocity files in resolved groups."

    files = [f for _, f in sim_items]

    data_dir_path = Path(files[0]).parent.resolve()
    from pages.AutonomousLab.session_sync import update_data_directory_in_context
    update_data_directory_in_context(session_context, data_dir_path)
    dx, dy, dz = _get_dx_dy_dz(data_dir_path, dx_override)

    # ν (kinematic viscosity): args > session_context > simulation files > default
    if "nu" in args and args["nu"] is not None:
        nu = float(args["nu"])
    else:
        sess_nu = (session_context or {}).get("pdfs_nu")
        if sess_nu is not None:
            nu = float(sess_nu)
        else:
            nu_from_file = _get_nu_from_params(data_dir_path)
            nu = float(nu_from_file) if nu_from_file is not None else 0.004

    # Session style (same as other pages): read from pdfs_style_config or pdfs_plot_styles
    from utils.plot_style import default_plot_style
    plot_name = _PDF_TYPE_TO_PLOT_NAME.get(pdf_type, "Velocity Magnitude PDF")
    pdfs_styles = session_context.get("pdfs_plot_styles") or {}
    style_config = session_context.get("pdfs_style_config") or pdfs_styles.get(plot_name) or default_plot_style()
    if not isinstance(style_config, dict):
        style_config = default_plot_style()
    else:
        style_config = dict(style_config)
    style_config.setdefault("per_sim_style_comparison", {})
    style_config.setdefault("enable_per_sim_style", False)
    style_updates = args.get("style_updates") or {}
    if style_updates:
        style_config.update(style_updates)
        if "custom_colors" in style_updates:
            style_config["palette"] = "Custom"
    session_context.setdefault("plot_styles", {})[plot_name] = style_config

    simulation_legend_names = args.get("simulation_legend_names") or {}
    if isinstance(simulation_legend_names, dict):
        session_context.setdefault("pdfs_sim_legend_names", {}).update(simulation_legend_names)
    sim_legends = session_context.get("pdfs_sim_legend_names") or {}

    velocity = _load_velocity(files[0], fortran_order=fortran_order)
    if velocity is None:
        return f"Error: Could not load velocity from {files[0].name}"

    if pdf_type == "velocity_components":
        u_grid, pdf_u, pdf_v, pdf_w = compute_velocity_pdf(velocity, bins=bins, normalize=normalize)
        if len(u_grid) == 0:
            return "Error: No valid data for velocity components PDF"
        x_label = "u / σ<sub>u</sub>" if normalize else "u"
        y_label = "σ<sub>u</sub> P(u / σ<sub>u</sub>)" if normalize else "P(u)"
        title = "Velocity PDF"
        # Legend & Axis Labels
        axis_key_x, axis_key_y = _PDF_TYPE_AXIS_KEYS["velocity_components"]
        legend_key = _PDF_TYPE_LEGEND_KEY["velocity_components"]
        agent_axis = args.get("axis_labels") or {}
        agent_legend = args.get("legend_names") or {}
        axis_labels_pdfs = session_context.setdefault("axis_labels_pdfs", {})
        legend_titles_pdfs = session_context.setdefault("legend_titles_pdfs", {})
        if agent_axis and isinstance(agent_axis, dict):
            if "x" in agent_axis:
                axis_labels_pdfs[axis_key_x] = agent_axis["x"]
            if "y" in agent_axis:
                axis_labels_pdfs[axis_key_y] = agent_axis["y"]
        x_label = axis_labels_pdfs.get(axis_key_x, x_label)
        y_label = axis_labels_pdfs.get(axis_key_y, y_label)
        legend_title = legend_titles_pdfs.get(legend_key, "")
        if agent_legend and isinstance(agent_legend, dict):
            legend_titles_pdfs[legend_key] = next(iter(agent_legend.values()), legend_title)
        label_base = Path(files[0].name).stem
        from visualizations.pdfs_vis import create_velocity_components_pdf_figure
        fig = create_velocity_components_pdf_figure(
            u_grid, pdf_u, pdf_v, pdf_w, style_config,
            x_label=x_label, y_label=y_label, title=title,
            legend_title=legend_title or None,
            label_base=label_base,
            legend_names=agent_legend if isinstance(agent_legend, dict) else None,
        )
        session_context["last_figure"] = fig
        _write_back_pdf_files(session_context, pdf_type, files)
        _write_back_bins_normalize(session_context, pdf_type, bins, normalize)
        _write_back_nu_dx(session_context, nu, dx)
        return {
            "status": "success",
            "message": f"{title} created from {files[0].name}.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }
    elif pdf_type in ("velocity_magnitude", "vorticity", "enstrophy", "dissipation"):
        # 1D PDFs: support multi-file with per-sim styling
        trace_data: List[tuple] = []  # (sim_prefix, x_vals, pdf_vals)
        for sim_prefix, fpath in sim_items:
            vel = _load_velocity(fpath, fortran_order=fortran_order)
            if vel is None:
                continue
            d = Path(fpath).parent.resolve()
            _dx, _dy, _dz = _get_dx_dy_dz(d, dx_override)
            if pdf_type == "velocity_magnitude":
                xv, pv = compute_velocity_magnitude_pdf(vel, bins=bins, normalize=normalize)
            elif pdf_type == "vorticity":
                xv, pv = compute_vorticity_pdf(vel, dx=_dx, dy=_dy, dz=_dz, bins=bins, normalize=normalize)
            elif pdf_type == "enstrophy":
                xv, pv = compute_enstrophy_pdf(vel, dx=_dx, dy=_dy, dz=_dz, bins=bins, normalize=normalize)
            else:  # dissipation
                xv, pv = compute_dissipation_pdf(vel, nu=nu, dx=_dx, dy=_dy, dz=_dz, bins=bins, normalize=normalize)
            if len(xv) > 0 and len(pv) > 0:
                trace_data.append((sim_prefix, xv, pv))
        if not trace_data:
            return f"Error: No valid data for {pdf_type} PDF"
        x_vals = trace_data[0][1]  # for shape check
        pdf_vals = trace_data[0][2]
        if pdf_type == "velocity_magnitude":
            x_label = "|u| / σ<sub>|u|</sub>" if normalize else "|u|"
            y_label = "σ<sub>|u|</sub> P(|u|)" if normalize else "P(|u|)"
            title = "Velocity Magnitude PDF"
        elif pdf_type == "vorticity":
            x_label = "|ω| / σ<sub>|ω|</sub>" if normalize else "|ω|"
            y_label = "σ<sub>|ω|</sub> P(|ω|)" if normalize else "P(|ω|)"
            title = "Vorticity Magnitude PDF"
        elif pdf_type == "enstrophy":
            x_label = "Ω / ⟨Ω⟩" if normalize else "Ω"
            y_label = "⟨Ω⟩ P(Ω)" if normalize else "P(Ω)"
            title = "Enstrophy PDF"
        else:
            x_label = "ε / ⟨ε⟩" if normalize else "ε"
            y_label = "⟨ε⟩ P(ε)" if normalize else "P(ε)"
            title = "Dissipation Rate PDF"
    elif pdf_type == "joint_velocity_dissipation":
        u_centers, eps_centers, joint_pdf = compute_velocity_dissipation_joint_pdf(
            velocity, nu=nu, dx=dx, dy=dy, dz=dz, bins=bins, normalize=normalize
        )
        if u_centers.size == 0 or eps_centers.size == 0:
            return "Error: No valid data for P(|u|, ε) joint PDF"
        x_label = "|u| / σ<sub>|u|</sub>" if normalize else "|u|"
        y_label = "ε / ⟨ε⟩" if normalize else "ε"
        title = "P(|u|, ε)"
        # 2D contour plot
        axis_key_x, axis_key_y = _PDF_TYPE_AXIS_KEYS["joint_velocity_dissipation"]
        legend_key = _PDF_TYPE_LEGEND_KEY["joint_velocity_dissipation"]
        agent_axis = args.get("axis_labels") or {}
        agent_legend = args.get("legend_names") or {}
        axis_labels_pdfs = session_context.setdefault("axis_labels_pdfs", {})
        legend_titles_pdfs = session_context.setdefault("legend_titles_pdfs", {})
        if agent_axis and isinstance(agent_axis, dict):
            if "x" in agent_axis:
                axis_labels_pdfs[axis_key_x] = agent_axis["x"]
            if "y" in agent_axis:
                axis_labels_pdfs[axis_key_y] = agent_axis["y"]
        x_label = axis_labels_pdfs.get(axis_key_x, x_label)
        y_label = axis_labels_pdfs.get(axis_key_y, y_label)
        legend_title = legend_titles_pdfs.get(legend_key, "")
        if agent_legend and isinstance(agent_legend, dict):
            legend_titles_pdfs[legend_key] = next(iter(agent_legend.values()), legend_title)
        from visualizations.pdfs_vis import create_2d_contour_pdf_figure
        fig = create_2d_contour_pdf_figure(
            u_centers, eps_centers, joint_pdf, style_config,
            x_label=x_label, y_label=y_label, z_label="PDF", title=title,
            legend_title=legend_title or None,
            trace_name=Path(files[0].name).stem,
            hovertemplate="|u| = %{x:.4f}<br>ε = %{y:.4e}<br>PDF = %{z:.2e}<extra></extra>",
            use_log_scale=use_log_scale,
        )
        session_context["last_figure"] = fig
        _write_back_pdf_files(session_context, pdf_type, files)
        _write_back_bins_normalize(session_context, pdf_type, bins, normalize)
        _write_back_log_scale(session_context, pdf_type, use_log_scale)
        _write_back_nu_dx(session_context, nu, dx)
        return {
            "status": "success",
            "message": f"{title} created from {files[0].name}.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }
    elif pdf_type == "joint_velocity_enstrophy":
        u_centers, omega_centers, joint_pdf = compute_velocity_enstrophy_joint_pdf(
            velocity, dx=dx, dy=dy, dz=dz, bins=bins
        )
        if u_centers.size == 0 or omega_centers.size == 0:
            return "Error: No valid data for P(|u|, |ω|) joint PDF"
        x_label = "|u|"
        y_label = "|ω|"
        title = "P(|u|, |ω|)"
        axis_key_x, axis_key_y = _PDF_TYPE_AXIS_KEYS["joint_velocity_enstrophy"]
        legend_key = _PDF_TYPE_LEGEND_KEY["joint_velocity_enstrophy"]
        agent_axis = args.get("axis_labels") or {}
        agent_legend = args.get("legend_names") or {}
        axis_labels_pdfs = session_context.setdefault("axis_labels_pdfs", {})
        legend_titles_pdfs = session_context.setdefault("legend_titles_pdfs", {})
        if agent_axis and isinstance(agent_axis, dict):
            if "x" in agent_axis:
                axis_labels_pdfs[axis_key_x] = agent_axis["x"]
            if "y" in agent_axis:
                axis_labels_pdfs[axis_key_y] = agent_axis["y"]
        x_label = axis_labels_pdfs.get(axis_key_x, x_label)
        y_label = axis_labels_pdfs.get(axis_key_y, y_label)
        legend_title = legend_titles_pdfs.get(legend_key, "")
        if agent_legend and isinstance(agent_legend, dict):
            legend_titles_pdfs[legend_key] = next(iter(agent_legend.values()), legend_title)
        from visualizations.pdfs_vis import create_2d_contour_pdf_figure
        fig = create_2d_contour_pdf_figure(
            u_centers, omega_centers, joint_pdf, style_config,
            x_label=x_label, y_label=y_label, z_label="PDF", title=title,
            legend_title=legend_title or None,
            trace_name=Path(files[0].name).stem,
            hovertemplate="|u| = %{x:.4f}<br>|ω| = %{y:.4f}<br>PDF = %{z:.2e}<extra></extra>",
            use_log_scale=use_log_scale,
        )
        session_context["last_figure"] = fig
        _write_back_pdf_files(session_context, pdf_type, files)
        _write_back_bins_normalize(session_context, pdf_type, bins, normalize)
        _write_back_log_scale(session_context, pdf_type, use_log_scale)
        _write_back_nu_dx(session_context, nu, dx)
        return {
            "status": "success",
            "message": f"{title} created from {files[0].name}.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }
    elif pdf_type == "joint_dissipation_enstrophy":
        eps_centers, omega_centers, joint_pdf = compute_dissipation_enstrophy_joint_pdf(
            velocity, nu=nu, dx=dx, dy=dy, dz=dz, bins=bins, normalize=normalize
        )
        if eps_centers.size == 0 or omega_centers.size == 0:
            return "Error: No valid data for P(ε, |ω|) joint PDF"
        x_label = "ε / ⟨ε⟩" if normalize else "ε"
        y_label = "|ω| / σ<sub>|ω|</sub>" if normalize else "|ω|"
        title = "P(ε, |ω|)"
        axis_key_x, axis_key_y = _PDF_TYPE_AXIS_KEYS["joint_dissipation_enstrophy"]
        legend_key = _PDF_TYPE_LEGEND_KEY["joint_dissipation_enstrophy"]
        agent_axis = args.get("axis_labels") or {}
        agent_legend = args.get("legend_names") or {}
        axis_labels_pdfs = session_context.setdefault("axis_labels_pdfs", {})
        legend_titles_pdfs = session_context.setdefault("legend_titles_pdfs", {})
        if agent_axis and isinstance(agent_axis, dict):
            if "x" in agent_axis:
                axis_labels_pdfs[axis_key_x] = agent_axis["x"]
            if "y" in agent_axis:
                axis_labels_pdfs[axis_key_y] = agent_axis["y"]
        x_label = axis_labels_pdfs.get(axis_key_x, x_label)
        y_label = axis_labels_pdfs.get(axis_key_y, y_label)
        legend_title = legend_titles_pdfs.get(legend_key, "")
        if agent_legend and isinstance(agent_legend, dict):
            legend_titles_pdfs[legend_key] = next(iter(agent_legend.values()), legend_title)
        from visualizations.pdfs_vis import create_2d_contour_pdf_figure
        fig = create_2d_contour_pdf_figure(
            eps_centers, omega_centers, joint_pdf, style_config,
            x_label=x_label, y_label=y_label, z_label="PDF", title=title,
            legend_title=legend_title or None,
            trace_name=Path(files[0].name).stem,
            hovertemplate="ε = %{x:.4e}<br>|ω| = %{y:.4f}<br>PDF = %{z:.2e}<extra></extra>",
            use_log_scale=use_log_scale,
        )
        session_context["last_figure"] = fig
        _write_back_pdf_files(session_context, pdf_type, files)
        _write_back_bins_normalize(session_context, pdf_type, bins, normalize)
        _write_back_log_scale(session_context, pdf_type, use_log_scale)
        _write_back_nu_dx(session_context, nu, dx)
        return {
            "status": "success",
            "message": f"{title} created from {files[0].name}.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }
    elif pdf_type == "rq_joint":
        R_centers, Q_centers, joint_pdf = compute_rq_joint_pdf(
            velocity, r_bins=bins, q_bins=bins, dx=dx, dy=dy, dz=dz
        )
        if R_centers.size == 0 or Q_centers.size == 0:
            return "Error: No valid data for R-Q joint PDF"
        x_label = "R"
        y_label = "Q"
        title = "R-Q Topological Space"
        axis_key_x, axis_key_y = _PDF_TYPE_AXIS_KEYS["rq_joint"]
        legend_key = _PDF_TYPE_LEGEND_KEY["rq_joint"]
        agent_axis = args.get("axis_labels") or {}
        agent_legend = args.get("legend_names") or {}
        axis_labels_pdfs = session_context.setdefault("axis_labels_pdfs", {})
        legend_titles_pdfs = session_context.setdefault("legend_titles_pdfs", {})
        if agent_axis and isinstance(agent_axis, dict):
            if "x" in agent_axis:
                axis_labels_pdfs[axis_key_x] = agent_axis["x"]
            if "y" in agent_axis:
                axis_labels_pdfs[axis_key_y] = agent_axis["y"]
        x_label = axis_labels_pdfs.get(axis_key_x, x_label)
        y_label = axis_labels_pdfs.get(axis_key_y, y_label)
        legend_title = legend_titles_pdfs.get(legend_key, "")
        if agent_legend and isinstance(agent_legend, dict):
            legend_titles_pdfs[legend_key] = next(iter(agent_legend.values()), legend_title)
        from visualizations.pdfs_vis import create_2d_contour_pdf_figure
        fig = create_2d_contour_pdf_figure(
            R_centers, Q_centers, joint_pdf, style_config,
            x_label=x_label, y_label=y_label, z_label="PDF", title=title,
            legend_title=legend_title or None,
            trace_name=Path(files[0].name).stem,
            hovertemplate="R = %{x:.4f}<br>Q = %{y:.4f}<br>PDF = %{z:.2e}<extra></extra>",
            use_log_scale=use_log_scale,
        )
        session_context["last_figure"] = fig
        _write_back_pdf_files(session_context, pdf_type, files)
        _write_back_bins_normalize(session_context, pdf_type, bins, normalize)
        _write_back_log_scale(session_context, pdf_type, use_log_scale)
        _write_back_nu_dx(session_context, nu, dx)
        return {
            "status": "success",
            "message": f"{title} created from {files[0].name}.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }
    else:
        return f"Error: Unknown pdf_type '{pdf_type}'. Use one of: {PDF_TYPES}"

    agent_axis = args.get("axis_labels") or {}
    agent_legend = args.get("legend_names") or {}

    # Legend & Axis Labels (persistent) — read from session, merge agent overrides, persist
    axis_key_x, axis_key_y = _PDF_TYPE_AXIS_KEYS.get(pdf_type, ("velocity_mag_x", "velocity_mag_y"))
    legend_key = _PDF_TYPE_LEGEND_KEY.get(pdf_type, "velocity_mag_pdf")
    axis_labels_pdfs = session_context.setdefault("axis_labels_pdfs", {})
    legend_titles_pdfs = session_context.setdefault("legend_titles_pdfs", {})
    sess_x = axis_labels_pdfs.get(axis_key_x, x_label)
    sess_y = axis_labels_pdfs.get(axis_key_y, y_label)
    if agent_axis and isinstance(agent_axis, dict):
        if "x" in agent_axis:
            axis_labels_pdfs[axis_key_x] = agent_axis["x"]
            sess_x = agent_axis["x"]
        if "y" in agent_axis:
            axis_labels_pdfs[axis_key_y] = agent_axis["y"]
            sess_y = agent_axis["y"]
    x_label = sess_x
    y_label = sess_y
    legend_title = legend_titles_pdfs.get(legend_key, "")
    if agent_legend and isinstance(agent_legend, dict):
        legend_titles_pdfs[legend_key] = next(iter(agent_legend.values()), legend_title)

    from visualizations.pdfs_vis import create_1d_pdf_figure
    fig = create_1d_pdf_figure(
        trace_data, style_config,
        x_label=x_label, y_label=y_label, title=title,
        legend_title=legend_title or None,
        simulation_legend_names=simulation_legend_names,
        sim_legends=sim_legends,
    )

    session_context["last_figure"] = fig
    _write_back_pdf_files(session_context, pdf_type, files)
    _write_back_bins_normalize(session_context, pdf_type, bins, normalize)
    _write_back_nu_dx(session_context, nu, dx)

    return {
        "status": "success",
        "message": f"{title} created from {files[0].name}.",
        "artifact_type": "plotly_figure",
        "artifact_content": fig.to_json(),
        **get_artifact_source_meta(__file__, project_root, name),
    }
