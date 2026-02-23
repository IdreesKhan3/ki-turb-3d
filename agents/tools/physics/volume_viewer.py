"""
3D Volume Viewer agent tools: plot_volume_3d, get_volume_viewer_theory.

Wires the 3D Volume Viewer page (Page 11) into the agentic schema.
Loads *.vti, *.h5, *.hdf5 velocity fields; computes scalar fields (velocity magnitude,
vorticity, Q_S^S, Q/R invariants); produces 3D Plotly figures (slices, volume, isosurface).
"""

import glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._meta import get_artifact_source_meta

FIELD_TYPES = [
    "ux", "uy", "uz", "Velocity Magnitude",
    "Vorticity Magnitude", "ωx", "ωy", "ωz",
    "Q_S^S", "Q Invariant", "R Invariant",
]
COLORMAPS = [
    "viridis", "cividis", "plasma", "magma", "inferno",
    "turbo", "rainbow", "jet", "rdbu", "spectral",
    "ice", "electric", "hot", "icefire", "greys",
]


def _apply_clip(field: np.ndarray, xmin: int, xmax: int, ymin: int, ymax: int, zmin: int, zmax: int) -> np.ndarray:
    """Apply clipping box to field (set outside region to NaN)."""
    clipped = field.copy()
    mask = np.ones_like(clipped, dtype=bool)
    mask &= np.arange(clipped.shape[0])[:, None, None] >= xmin
    mask &= np.arange(clipped.shape[0])[:, None, None] <= xmax
    mask &= np.arange(clipped.shape[1])[None, :, None] >= ymin
    mask &= np.arange(clipped.shape[1])[None, :, None] <= ymax
    mask &= np.arange(clipped.shape[2])[None, None, :] >= zmin
    mask &= np.arange(clipped.shape[2])[None, None, :] <= zmax
    clipped[~mask] = np.nan
    return clipped


def _collect_volume_files(
    data_dirs: List[str],
    project_root: Path,
    file_type_filter: str = "both",
) -> List[str]:
    """Collect VTI and HDF5 files from data directories. file_type_filter: vti, hdf5, or both."""

    def _nsort(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]

    all_files: List[str] = []
    for d in data_dirs:
        p = Path(d)
        if not p.is_absolute():
            p = (project_root / str(d).lstrip("/")).resolve()
        if not p.exists() or not p.is_dir():
            alt = project_root / "examples" / str(d).lstrip("/")
            if alt.exists() and alt.is_dir():
                p = alt
            else:
                continue
        vti = glob.glob(str(p / "*.vti")) + glob.glob(str(p / "*.VTI"))
        h5 = glob.glob(str(p / "*.h5")) + glob.glob(str(p / "*.H5")) + \
             glob.glob(str(p / "*.hdf5")) + glob.glob(str(p / "*.HDF5"))
        if file_type_filter.lower() == "vti":
            all_files.extend(vti)
        elif file_type_filter.lower() in ("hdf5", "h5"):
            all_files.extend(h5)
        else:
            all_files.extend(vti)
            all_files.extend(h5)
    return sorted(set(all_files), key=_nsort)


def _load_velocity_file(filepath: str) -> Dict[str, Any]:
    """Load velocity from VTI or HDF5. Returns dict with 'velocity' key."""
    import streamlit as st
    from data_readers.vti_reader import read_vti_file
    from data_readers.hdf5_reader import read_hdf5_file

    p = Path(filepath).resolve()
    s = str(p).lower()
    if s.endswith((".h5", ".hdf5")):
        fortran_order = st.session_state.get("hdf5_fortran_order", True)
        return read_hdf5_file(str(p), fortran_order=fortran_order)
    if s.endswith(".vti"):
        return read_vti_file(str(p))
    raise ValueError(f"Unsupported format: {filepath}. Expected .vti, .h5, or .hdf5")


def _compute_field(
    velocity: np.ndarray,
    field_type: str,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> np.ndarray:
    """Compute scalar field from velocity based on field_type. dx,dy,dz for gradient-based fields."""
    from data_readers.vti_reader import compute_velocity_magnitude, compute_vorticity
    from utils.iso_surfaces import compute_qs_s, compute_q_invariant, compute_r_invariant

    if field_type == "Velocity Magnitude":
        return compute_velocity_magnitude(velocity)
    if field_type == "ux":
        return velocity[:, :, :, 0]
    if field_type == "uy":
        return velocity[:, :, :, 1]
    if field_type == "uz":
        return velocity[:, :, :, 2]
    if field_type == "Vorticity Magnitude":
        vort = compute_vorticity(velocity, dx=dx, dy=dy, dz=dz)
        return np.sqrt(vort[:, :, :, 0] ** 2 + vort[:, :, :, 1] ** 2 + vort[:, :, :, 2] ** 2)
    if field_type.startswith("ω"):
        vort = compute_vorticity(velocity, dx=dx, dy=dy, dz=dz)
        if field_type == "ωx":
            return vort[:, :, :, 0]
        if field_type == "ωy":
            return vort[:, :, :, 1]
        return vort[:, :, :, 2]
    if field_type == "Q_S^S":
        return compute_qs_s(velocity, dx=dx, dy=dy, dz=dz)
    if field_type == "Q Invariant":
        return compute_q_invariant(velocity, dx=dx, dy=dy, dz=dz)
    if field_type == "R Invariant":
        return compute_r_invariant(velocity, dx=dx, dy=dy, dz=dz)
    return velocity[:, :, :, 0]


def _safe_minmax(a: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(a.min()), float(a.max())
    if vmin == vmax:
        vmax = vmin * 1.001 if vmin != 0 else 1.0
    return vmin, vmax


def _downsample3d(field: np.ndarray, step: int) -> np.ndarray:
    if step <= 1:
        return field
    return field[::step, ::step, ::step]


def _make_grid(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    return x, y, z


def _resolve_data_dirs(
    data_dir: str,
    data_directories: Optional[List[str]],
    file_path: Optional[str],
    session_context: Dict[str, Any],
    project_root: Path,
) -> List[str]:
    sess = session_context or {}
    if data_directories and isinstance(data_directories, list) and len(data_directories) > 0:
        return [str(d).strip() for d in data_directories if d and str(d).strip()]
    if data_dir and str(data_dir).strip():
        return [str(data_dir).strip()]
    if file_path and str(file_path).strip():
        p = Path(str(file_path).strip())
        if p.suffix.lower() in (".vti", ".h5", ".hdf5"):
            return [str(p.parent)]
        return [str(file_path).strip()]
    if sess.get("data_directories"):
        d = sess["data_directories"]
        return [str(x) for x in (list(d) if isinstance(d, list) else [d]) if x]
    if sess.get("data_directory"):
        return [str(sess["data_directory"])]
    return []


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for 3D Volume Viewer."""
    return [
        {
            "name": "plot_volume_3d",
            "description": "Create 3D volume visualization from *.vti, *.h5, *.hdf5 velocity fields (3D Volume Viewer page). Fields: ux, uy, uz, Velocity Magnitude, Vorticity Magnitude, ωx/ωy/ωz, Q_S^S, Q Invariant, R Invariant. Display: orthogonal slices (default), volume rendering, isosurface. Grid spacing auto from simulation.json for NS data (dx=L/nx); use dx=1 for LBM. Use when user asks for '3d volume', 'volume viewer', 'velocity field 3d', 'vorticity 3d', 'vti visualization'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Directory path (e.g. examples/DNS/512). REQUIRED unless session has data_directory."},
                    "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple directories."},
                    "file_path": {"type": "string", "description": "Alternative: path to a specific .vti or .h5 file."},
                    "file_index": {"type": "integer", "description": "Time step index (0 = first file). Default 0."},
                    "file_type_filter": {"type": "string", "description": "File extension: vti, hdf5, or both. Default both."},
                    "field_type": {"type": "string", "description": f"Field: {', '.join(FIELD_TYPES[:8])}... Default Velocity Magnitude."},
                    "downsample_step": {"type": "integer", "description": "Downsample (1-8). Default 2."},
                    "show_volume": {"type": "boolean", "description": "Volume rendering. Default false."},
                    "show_slices": {"type": "boolean", "description": "Orthogonal slices. Default true."},
                    "show_iso": {"type": "boolean", "description": "Isosurface. Default false."},
                    "show_surface": {"type": "boolean", "description": "6-face surface. Default false."},
                    "colormap": {"type": "string", "description": f"Colormap. Default rdbu."},
                    "color_max": {"type": "number", "description": "Color max (contrast). Default 0.6*vmax."},
                    "vmin": {"type": "number", "description": "Value range min (optional)."},
                    "vmax": {"type": "number", "description": "Value range max (optional)."},
                    "vol_opacity": {"type": "number", "description": "Volume opacity 0.01-0.8. Default 0.15."},
                    "vol_surface_count": {"type": "integer", "description": "Volume surfaces 5-40. Default 20."},
                    "iso_value": {"type": "number", "description": "Isosurface value. For Q_S^S, Q, R: this is the threshold (e.g. 1e-4 for 10^-4)."},
                    "iso_value_log10": {"type": "number", "description": "Alternative: log10 of threshold. When user says 'Q_S^S threshold 10^-4', pass iso_value_log10: -4. Actual iso_value = 10^iso_value_log10."},
                    "iso_opacity": {"type": "number", "description": "Isosurface opacity 0.05-1. Default 0.4."},
                    "slice_x": {"type": "integer", "description": "X slice index. Default center."},
                    "slice_y": {"type": "integer", "description": "Y slice index. Default center."},
                    "slice_z": {"type": "integer", "description": "Z slice index. Default center."},
                    "slice_opacity": {"type": "number", "description": "Slice opacity 0.05-1. Default 0.9."},
                    "surface_opacity": {"type": "number", "description": "Surface opacity (6 faces). Default 0.8."},
                    "use_clip": {"type": "boolean", "description": "Enable clipping box. Default false."},
                    "clip_x": {"type": "array", "items": {"type": "integer"}, "description": "Clip X [min,max]. Default [0,nx-1]."},
                    "clip_y": {"type": "array", "items": {"type": "integer"}, "description": "Clip Y [min,max]."},
                    "clip_z": {"type": "array", "items": {"type": "integer"}, "description": "Clip Z [min,max]."},
                    "show_axes": {"type": "boolean", "description": "Show coordinate axes. Default false."},
                    "show_axis_labels": {"type": "boolean", "description": "Show axis labels. Default false."},
                    "camera_preset": {"type": "string", "description": "View: Isometric, XY, XZ, YZ. Default Isometric."},
                    "dx": {"type": "number", "description": "Grid spacing. Auto from simulation.json for NS. Use 1 for LBM."},
                    "style_updates": {"type": "object", "description": "Plot style: plot_bgcolor, paper_bgcolor, font_family, height, figure_height, plot_title, show_plot_title, template."},
                },
            },
        },
        {
            "name": "get_volume_viewer_theory",
            "description": "Return theory & equations for 3D Volume Viewer: velocity magnitude, vorticity, Q_S^S method, Q/R invariants. Use when user asks for 'volume viewer theory', '3d volume equations', 'vorticity equations', 'Q invariant theory', 'Q_S^S equations'.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute 3D Volume Viewer tool."""
    session_context = session_context or {}
    project_root = project_root or Path(".")

    if name == "get_volume_viewer_theory":
        from content.volume_viewer_theory_content import get_volume_viewer_theory_markdown
        return {
            "status": "success",
            "message": "3D Volume Viewer theory.",
            "artifact_type": "markdown",
            "artifact_content": get_volume_viewer_theory_markdown(),
            "artifact_title": "3D Volume Viewer — Theory & Equations",
            **get_artifact_source_meta(__file__, project_root, name),
        }

    if name == "plot_volume_3d":
        data_dir = args.get("data_dir", "")
        data_directories = args.get("data_directories")
        file_path = args.get("file_path", "")
        dirs = _resolve_data_dirs(data_dir, data_directories, file_path, session_context, project_root)
        if not dirs:
            return (
                "Error: No data directory. Pass data_dir (e.g. examples/DNS/512) or file_path. "
                "When the task says 'from DNS/512' or 'from examples/DNS/512', use that path as data_dir."
            )

        file_type_filter = str(args.get("file_type_filter") or "both").strip().lower()
        all_files = _collect_volume_files(dirs, project_root, file_type_filter=file_type_filter)
        if not all_files:
            return "Error: No *.vti, *.h5, or *.hdf5 files found in the given path(s)."

        file_index = int(args.get("file_index", 0))
        file_index = max(0, min(file_index, len(all_files) - 1))
        selected_file = all_files[file_index]

        try:
            vti_data = _load_velocity_file(selected_file)
        except Exception as e:
            return f"Error loading {Path(selected_file).name}: {e}"

        velocity = vti_data.get("velocity")
        if velocity is None or len(velocity.shape) != 4:
            return f"Error: Invalid velocity shape {velocity.shape if velocity is not None else 'None'}"

        field_type = args.get("field_type") or "Velocity Magnitude"
        if field_type not in FIELD_TYPES:
            field_type = "Velocity Magnitude"

        # Resolve grid spacing (LBM vs NS): args override, else from simulation.json in data dir
        if "dx" in args and args["dx"] is not None:
            dx_val = float(args["dx"])
            dx, dy, dz = dx_val, dx_val, dx_val
        else:
            try:
                from pages.PDFs.pdf_params import get_grid_spacing
                data_dir_path = Path(dirs[0]).resolve()
                dx, dy, dz = get_grid_spacing(data_dir_path)
            except Exception:
                dx, dy, dz = 1.0, 1.0, 1.0

        field = _compute_field(velocity, field_type, dx=dx, dy=dy, dz=dz)
        downsample_step = int(args.get("downsample_step", 2) or 2)
        downsample_step = max(1, min(8, downsample_step))
        field_ds = _downsample3d(field, downsample_step)
        nx_d, ny_d, nz_d = field_ds.shape

        use_clip = bool(args.get("use_clip", False))
        if use_clip:
            clip_x = args.get("clip_x")
            clip_y = args.get("clip_y")
            clip_z = args.get("clip_z")
            xmin = int(clip_x[0]) if isinstance(clip_x, (list, tuple)) and len(clip_x) >= 2 else 0
            xmax = int(clip_x[1]) if isinstance(clip_x, (list, tuple)) and len(clip_x) >= 2 else nx_d - 1
            ymin = int(clip_y[0]) if isinstance(clip_y, (list, tuple)) and len(clip_y) >= 2 else 0
            ymax = int(clip_y[1]) if isinstance(clip_y, (list, tuple)) and len(clip_y) >= 2 else ny_d - 1
            zmin = int(clip_z[0]) if isinstance(clip_z, (list, tuple)) and len(clip_z) >= 2 else 0
            zmax = int(clip_z[1]) if isinstance(clip_z, (list, tuple)) and len(clip_z) >= 2 else nz_d - 1
            xmin, xmax = max(0, min(xmin, xmax)), min(nx_d - 1, max(xmin, xmax))
            ymin, ymax = max(0, min(ymin, ymax)), min(ny_d - 1, max(ymin, ymax))
            zmin, zmax = max(0, min(zmin, zmax)), min(nz_d - 1, max(zmin, zmax))
            field_ds = _apply_clip(field_ds, xmin, xmax, ymin, ymax, zmin, zmax)
        xg, yg, zg = _make_grid(nx_d, ny_d, nz_d)
        vmin, vmax = _safe_minmax(field_ds)
        if "vmin" in args and args["vmin"] is not None:
            vmin = float(args["vmin"])
        if "vmax" in args and args["vmax"] is not None:
            vmax = float(args["vmax"])
        cmax = float(args.get("color_max")) if args.get("color_max") is not None else (float(vmax) * 0.6 if vmax > vmin else vmax)
        cmap = args.get("colormap") or "rdbu"
        if cmap not in COLORMAPS:
            cmap = "rdbu"

        show_volume = bool(args.get("show_volume", False))
        show_slices = bool(args.get("show_slices", True))
        show_iso = bool(args.get("show_iso", False))
        show_surface = bool(args.get("show_surface", False))
        if not show_volume and not show_slices and not show_iso and not show_surface:
            show_slices = True

        vol_opacity = float(args.get("vol_opacity", 0.15) or 0.15)
        vol_opacity = max(0.01, min(0.8, vol_opacity))
        vol_surface_count = int(args.get("vol_surface_count", 20) or 20)
        vol_surface_count = max(5, min(40, vol_surface_count))

        # Sensible iso default for Q_S^S, Q, R (span many orders; vortices at positive values)
        if "iso_value_log10" in args and args["iso_value_log10"] is not None:
            iso_value = 10.0 ** float(args["iso_value_log10"])
        elif "iso_value" in args and args["iso_value"] is not None:
            iso_value = float(args["iso_value"])
        elif show_iso and field_type in ("Q_S^S", "Q Invariant", "R Invariant"):
            f_max = float(np.nanmax(np.abs(field_ds)))
            if f_max > 1e-30:
                iso_value = 0.5 * f_max
            else:
                iso_value = float((vmin + vmax) / 2) if vmax > vmin else vmin
        else:
            iso_value = float((vmin + vmax) / 2) if vmax > vmin else vmin
        iso_opacity = float(args.get("iso_opacity", 0.4) or 0.4)
        iso_opacity = max(0.05, min(1.0, iso_opacity))
        slice_opacity = float(args.get("slice_opacity", 0.9) or 0.9)
        slice_opacity = max(0.05, min(1.0, slice_opacity))
        surface_opacity = float(args.get("surface_opacity", 0.8) or 0.8)
        surface_opacity = max(0.05, min(1.0, surface_opacity))
        slice_x = int(args.get("slice_x", nx_d // 2))
        slice_y = int(args.get("slice_y", ny_d // 2))
        slice_z = int(args.get("slice_z", nz_d // 2))
        slice_x = max(0, min(slice_x, nx_d - 1))
        slice_y = max(0, min(slice_y, ny_d - 1))
        slice_z = max(0, min(slice_z, nz_d - 1))

        show_axes = bool(args.get("show_axes", False))
        show_axis_labels = bool(args.get("show_axis_labels", False))
        camera_preset = str(args.get("camera_preset") or "Isometric").strip()
        style_updates = args.get("style_updates")
        if style_updates is not None and not isinstance(style_updates, dict):
            style_updates = None

        from visualizations.volume_viewer_vis import build_3d_volume_figure
        fig = build_3d_volume_figure(
            xg, yg, zg,
            field_ds,
            vmin, vmax, cmax, cmap,
            field_type,
            show_volume=show_volume,
            vol_opacity=vol_opacity,
            vol_surface_count=vol_surface_count,
            show_iso=show_iso,
            iso_value=iso_value,
            iso_opacity=iso_opacity,
            show_slices=show_slices,
            slice_x=slice_x,
            slice_y=slice_y,
            slice_z=slice_z,
            slice_opacity=slice_opacity,
            show_surface=show_surface,
            surface_opacity=surface_opacity,
            show_axes=show_axes,
            show_axis_labels=show_axis_labels,
            camera_preset=camera_preset,
            style_updates=style_updates,
        )

        from pages.AutonomousLab.session_sync import update_data_directory_in_context
        update_data_directory_in_context(session_context, Path(selected_file).parent.resolve())
        session_context["last_figure"] = fig
        session_context.setdefault("figure_queue", []).append(fig)
        return {
            "status": "success",
            "message": f"3D volume plot created ({Path(selected_file).name}, {field_type}).",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            **get_artifact_source_meta(__file__, project_root, name),
        }

    return f"Error: Unknown tool '{name}'"
