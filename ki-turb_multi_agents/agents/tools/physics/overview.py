"""
Overview page tools: simulation parameters, physics validation, data availability.
Produces markdown/table artifacts matching the Overview page content.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.product_loader import AnalysisProductLoader
from utils.file_detector import detect_simulation_files

from content.overview_theory_content import get_overview_theory_markdown
from .._shared import get_from_cache, save_to_cache, update_data_directory_in_context

CACHE_KEY_OVERVIEW = "current_overview_validation"


def _resolve_overview_dirs(
    data_dir: str,
    data_directories: Optional[List[str]],
    project_root: Path,
    session_context: Dict[str, Any],
) -> List[Path]:
    """Resolve one or more data directories for overview."""
    sess = session_context or {}
    dirs: List[str] = []

    if data_directories and len(data_directories) > 0:
        dirs = [d.strip() for d in data_directories if d and str(d).strip()]
    elif data_dir and str(data_dir).strip():
        dirs = [str(data_dir).strip()]
    elif sess.get("data_directories"):
        dirs = [str(d) for d in sess["data_directories"]]
    elif sess.get("data_directory"):
        dirs = [str(sess["data_directory"])]

    resolved = []
    for d in dirs:
        p = Path(d)
        if not p.is_absolute():
            p = (project_root / d).resolve()
        if p.exists() and p.is_dir():
            resolved.append(p)
    return resolved


def _is_les_dir(data_dir: Path, project_root: Path, backend: Optional[str] = None) -> bool:
    """LES detection: legacy examples/LES path (backend-agnostic file heuristics)."""
    try:
        rel = data_dir.resolve().relative_to(project_root.resolve())
        parts = [p.lower() for p in rel.parts]
        if len(parts) >= 2 and parts[0] == "examples" and parts[1] == "les":
            return True
    except Exception:
        pass
    return bool(detect_simulation_files(str(data_dir)).get("tau_analysis"))


def _compute_overview_data(
    data_dirs: List[Path],
    project_root: Path,
    *,
    loader: Optional[AnalysisProductLoader] = None,
) -> List[Dict[str, Any]]:
    """Compute overview data for each directory (params, Mach, Kn, compressibility, files).
    Supports both LBM (simulation.input) and NS (simulation.json) FHIT.
    """
    from data_readers.parameter_reader import (
        read_parameters,
        format_parameters_for_display,
        is_lbm_params,
    )
    from data_readers.csv_reader import read_eps_validation_csv
    from data_readers.binary_reader import read_tau_analysis_file
    from data_readers.hdf5_reader import compute_compressibility_metrics as compute_compressibility_h5

    all_sim_data = []
    for data_dir in data_dirs:
        local_loader = loader or AnalysisProductLoader(project_root, {"data_directory": str(data_dir)})
        files = detect_simulation_files(str(data_dir))
        dir_name = data_dir.name
        backend = local_loader.backend()
        sim_data = {
            "directory": dir_name,
            "path": str(data_dir),
            "backend": backend,
            "files": files,
            "params": None,
            "raw_params": None,
            "mach_number": None,
            "knudsen_number": None,
            "compressibility": None,
            "is_les": _is_les_dir(data_dir, project_root, backend),
            "is_lbm": None,
            "validation_status": local_loader.validation_status(),
            "analysis_products_loaded": local_loader.products() is not None,
        }
        overview_hint = local_loader.overview_payload()
        sim_data.update({k: v for k, v in overview_hint.items() if k not in sim_data and v is not None})

        # Load parameters (prefer simulation.input for LBM, else simulation.json for NS)
        if files["parameters"]:
            for param_path in files["parameters"]:
                try:
                    params = read_parameters(str(param_path))
                    if params:
                        sim_data["is_lbm"] = is_lbm_params(str(param_path))
                        sim_data["params"] = format_parameters_for_display(
                            params, is_lbm=sim_data["is_lbm"]
                        )
                        sim_data["raw_params"] = params
                        break
                except Exception:
                    continue

        # Get u_rms from eps_real_validation CSV (u_rms_real or u_rms)
        u_rms = None
        if files.get("spectral_turb_stats"):
            candidates = sorted(
                [Path(p) for p in files["spectral_turb_stats"]],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for csv_path in candidates:
                try:
                    df = read_eps_validation_csv(str(csv_path))
                    u_col = "u_rms_real" if "u_rms_real" in df.columns else "u_rms"
                    if u_col in df.columns and len(df) > 0:
                        u_rms = float(df[u_col].iloc[-1])
                        break
                except Exception:
                    continue

        params = sim_data.get("raw_params")
        is_lbm = sim_data.get("is_lbm", True)  # Default LBM for backward compat

        # Mach number: LBM Ma = u_rms/c_s, NS Ma = u_rms/c_sound
        if u_rms is not None and params:
            if is_lbm:
                c_s = 1.0 / np.sqrt(3.0)
                sim_data["mach_number"] = float(u_rms / c_s)
                sim_data["mach_method"] = "LBM"
            else:
                c_sound = params.get("c_sound") or params.get("c_speed")
                if c_sound is not None and float(c_sound) > 0:
                    sim_data["mach_number"] = float(u_rms / float(c_sound))
                    sim_data["mach_method"] = "NS"
                # else: N/A (no c_sound in NS params)

        # Knudsen number: LBM Kn = c_s(tau-0.5), NS Kn = nu/(c_sound*L)
        if params and "nu" in params:
            nu = float(params["nu"])
            if is_lbm:
                c_s2 = 1.0 / 3.0
                c_s = 1.0 / np.sqrt(3.0)
                tau_0 = nu / c_s2 + 0.5
                sim_data["knudsen_number"] = float(c_s * (tau_0 - 0.5))
                sim_data["kn_method"] = "LBM"
                if sim_data["is_les"] and files.get("tau_analysis"):
                    nx, ny, nz = params.get("nx"), params.get("ny"), params.get("nz")
                    if nx and ny and nz:
                        try:
                            tau_file = str(files["tau_analysis"][-1])
                            tau_e = read_tau_analysis_file(tau_file, nx, ny, nz)
                            sim_data["knudsen_number"] = float((tau_e - 0.5) * np.sqrt(3.0))
                            sim_data["kn_method"] = "LBM (turbulent)"
                        except Exception:
                            pass
            else:
                c_sound = params.get("c_sound") or params.get("c_speed")
                L = params.get("L")
                if c_sound is not None and L is not None:
                    c_sound_f = float(c_sound)
                    L_f = float(L)
                    if c_sound_f > 0 and L_f > 0:
                        # Kn = nu / (c_sound * L) for mean-free-path estimate
                        sim_data["knudsen_number"] = float(nu / (c_sound_f * L_f))
                        sim_data["kn_method"] = "NS"

        if files.get("velocity_h5"):
            try:
                from data_readers.hdf5_reader import read_hdf5_file, compute_compressibility_metrics as compute_compressibility_h5

                fortran_order = True
                if loader is not None:
                    fortran_order = loader.hdf5_fortran_order()
                data = read_hdf5_file(str(files["velocity_h5"][0]), fortran_order=fortran_order)
                vel = data["velocity"]
                if vel.size > 128**3:
                    z_mid = vel.shape[0] // 2
                    vel = vel[z_mid : z_mid + 1, :, :, :]
                vel = np.transpose(vel, (2, 1, 0, 3))
                comp = compute_compressibility_h5(vel)
                if comp:
                    sim_data["compressibility"] = comp
            except Exception:
                pass

        products = local_loader.products()
        if products and products.time_history is not None and products.time_history.divergence_rms:
            sim_data["divergence_rms_max"] = float(max(products.time_history.divergence_rms))
        if products and products.resolution and products.resolution.kmax_eta_min is not None:
            sim_data["kmax_eta_min"] = products.resolution.kmax_eta_min

        all_sim_data.append(sim_data)
    return all_sim_data


def _format_overview_markdown(sim_data_list: List[Dict], project_root: Path) -> str:
    """Format overview data as markdown."""
    lines = ["# Overview Summary\n"]

    if len(sim_data_list) > 1:
        lines.append(f"**{len(sim_data_list)} simulations loaded.**\n")
    else:
        lines.append("**Single simulation.**\n")

    # Parameters
    lines.append("## Simulation Parameters\n")
    if len(sim_data_list) > 1:
        for sim in sim_data_list:
            if sim["params"]:
                lines.append(f"### {sim['directory']}")
                for label, info in sim["params"].items():
                    lines.append(f"- **{label}:** {info['value']} {info['unit']}")
                lines.append("")
    else:
        sim = sim_data_list[0]
        if sim["params"]:
            for label, info in sim["params"].items():
                lines.append(f"- **{label}:** {info['value']} {info['unit']}")
            lines.append("")

    # Physics validation
    has_validation = any(
        s["mach_number"] is not None or s["knudsen_number"] is not None or s["compressibility"] is not None
        for s in sim_data_list
    )
    if has_validation:
        lines.append("## Physics Validation\n")
        lines.append(
            "_Acceptance gates divergence only; Mach, Kn, and kmax×eta are informational._\n"
        )
        if len(sim_data_list) > 1:
            lines.append("| Directory | Mach | Knudsen | Max |∇·u| |")
            lines.append("|----------|------|---------|------------|")
            for sim in sim_data_list:
                ma = f"{sim['mach_number']:.4f}" if sim["mach_number"] is not None else "N/A"
                kn = f"{sim['knudsen_number']:.6f}" if sim["knudsen_number"] is not None else "N/A"
                div = f"{sim['compressibility']['max_divergence']:.6e}" if sim["compressibility"] else "N/A"
                lines.append(f"| {sim['directory']} | {ma} | {kn} | {div} |")
        else:
            sim = sim_data_list[0]
            if sim["mach_number"] is not None:
                method = sim.get("mach_method", "LBM")
                lines.append(f"- **Mach Number ({method}):** {sim['mach_number']:.4f}")
            if sim["knudsen_number"] is not None:
                kn_type = sim.get("kn_method") or ("turbulent" if sim["is_les"] else "molecular")
                lines.append(f"- **Knudsen Number ({kn_type}):** {sim['knudsen_number']:.6f}")
            if sim["compressibility"]:
                lines.append(f"- **Max |∇·u|:** {sim['compressibility']['max_divergence']:.6e}")
                lines.append(f"- **RMS ∇·u:** {sim['compressibility']['rms_divergence']:.6e}")
            if sim.get("divergence_rms_max") is not None:
                lines.append(f"- **Divergence RMS (products):** {sim['divergence_rms_max']:.6e}")
            if sim.get("validation_status"):
                lines.append(f"- **Validation status:** {sim['validation_status']}")
            if sim.get("kmax_eta_min") is not None:
                lines.append(f"- **kmax×η (min):** {sim['kmax_eta_min']:.4g}")
        lines.append("")

    # Data availability
    lines.append("## Data Availability\n")
    if len(sim_data_list) > 1:
        lines.append("| Directory | Real Stats | Spectra | Norm | Structure | Flatness | Isotropy | Spectral Stats |")
        lines.append("|----------|------------|---------|------|-----------|----------|----------|----------------|")
        for sim in sim_data_list:
            f = sim["files"]
            r1 = "Yes" if f["real_turb_stats"] else "No"
            r2 = "Yes" if f["spectrum"] else "No"
            r3 = "Yes" if f["norm_spectrum"] else "No"
            r4 = "Yes" if (f["structure_functions_txt"] or f["structure_functions_bin"]) else "No"
            r5 = "Yes" if f["flatness"] else "No"
            r6 = "Yes" if f["isotropy"] else "No"
            r7 = "Yes" if f["spectral_turb_stats"] else "No"
            lines.append(f"| {sim['directory']} | {r1} | {r2} | {r3} | {r4} | {r5} | {r6} | {r7} |")
    else:
        f = sim_data_list[0]["files"]
        items = [
            ("Real Turbulence Stats", f["real_turb_stats"]),
            ("Energy Spectra", f["spectrum"]),
            ("Normalized Spectra", f["norm_spectrum"]),
            ("Structure Functions", f["structure_functions_txt"] or f["structure_functions_bin"]),
            ("Flatness", f["flatness"]),
            ("Isotropy", f["isotropy"]),
            ("Spectral Turbulence Stats", f["spectral_turb_stats"]),
            ("Analysis Products", f.get("analysis_products")),
            ("Velocity PDFs", f.get("velocity_pdf")),
            ("Enstrophy PDFs", f.get("enstrophy_pdf")),
            ("Joint PDFs", f.get("joint_pdf")),
            ("R-Q PDFs", f.get("rq_pdf")),
            ("Tau Analysis (LES)", f.get("tau_analysis")),
        ]
        for name, flist in items:
            status = "Yes" if flist else "No"
            lines.append(f"- **{name}:** {status}")
        lines.append("")

    return "\n".join(lines)


OVERVIEW_TOOL = {
    "name": "get_overview_summary",
    "description": "Show Overview page content: simulation parameters, physics validation (Mach, Knudsen, compressibility), and data availability. Use when user asks for 'overview', 'parameters', 'metadata', 'physics validation', 'data availability', 'what files are available', 'Mach number', 'Knudsen number'.",
    "parameters": {
        "type": "object",
        "properties": {
            "data_dir": {"type": "string", "description": "Single directory path (optional: uses session data)"},
            "data_directories": {"type": "array", "items": {"type": "string"}, "description": "Multiple directories for comparison (optional)"},
        },
    },
}

OVERVIEW_THEORY_TOOL = {
    "name": "get_overview_theory",
    "description": "Show Overview page Physics Validation Equations: Mach number (LBM/NS), Knudsen number (LBM DNS/LES, NS), velocity divergence, compressibility metrics. Use when user asks for 'overview theory', 'overview equations', 'theory for overview', 'equations for overview', 'physics validation equations'.",
    "parameters": {"type": "object", "properties": {}},
}


OVERVIEW_COMPUTE_TOOL = {
    "name": "compute_overview_validation",
    "description": (
        "Compute Overview-page validation metrics (Mach, Knudsen, compressibility, tau_analysis, "
        "manifest diagnostics) and cache them. Solver-neutral: works from manifest products or "
        "legacy simulation files for OpenLB, Palabos, and example datasets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_dir": {"type": "string", "description": "Single directory (optional: uses session data)"},
            "data_directories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple directories for comparison",
            },
        },
    },
}


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [OVERVIEW_TOOL, OVERVIEW_THEORY_TOOL, OVERVIEW_COMPUTE_TOOL]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
):
    if name == "get_overview_theory":
        markdown = get_overview_theory_markdown()
        return {
            "status": "success",
            "message": "Overview theory equations created.",
            "artifact_type": "markdown",
            "artifact_content": markdown,
            "artifact_title": "Overview — Physics Validation Equations",
        }

    if name == "compute_overview_validation":
        session_context = session_context or {}
        data_dir = args.get("data_dir", "")
        data_directories = args.get("data_directories")
        dirs = _resolve_overview_dirs(data_dir, data_directories, project_root, session_context)
        if not dirs:
            return "Error: No data directory. Provide data_dir or load a dataset manifest first."
        loader = AnalysisProductLoader(project_root, session_context)
        sim_data_list = _compute_overview_data(dirs, project_root, loader=loader)
        save_to_cache(session_context, CACHE_KEY_OVERVIEW, sim_data_list)
        resolved_dirs = [str(p.resolve()) for p in dirs if p and p.exists()]
        if resolved_dirs:
            update_data_directory_in_context(
                session_context,
                resolved_dirs[0],
                data_dirs_list=resolved_dirs if len(resolved_dirs) > 1 else None,
            )
        return {
            "status": "success",
            "message": f"Computed overview validation for {len(sim_data_list)} dataset(s).",
            "cache_key": CACHE_KEY_OVERVIEW,
            "simulations": [item.get("directory") for item in sim_data_list],
        }

    if name != "get_overview_summary":
        return f"Error: Unknown overview tool '{name}'"

    session_context = session_context or {}
    data_dir = args.get("data_dir", "")
    data_directories = args.get("data_directories")

    dirs = _resolve_overview_dirs(data_dir, data_directories, project_root, session_context)
    if not dirs:
        return "Error: No data directory. Provide data_dir or data_directories, or load data from the main page sidebar."

    cached = get_from_cache(session_context, CACHE_KEY_OVERVIEW)
    if cached and isinstance(cached, list) and len(cached) == len(dirs):
        sim_data_list = cached
    else:
        loader = AnalysisProductLoader(project_root, session_context)
        sim_data_list = _compute_overview_data(dirs, project_root, loader=loader)
    if not sim_data_list:
        return "Error: Could not process any directory."

    resolved_dirs = [str(p.resolve()) for p in dirs if p and p.exists()]
    if resolved_dirs:
        update_data_directory_in_context(
            session_context,
            resolved_dirs[0],
            data_dirs_list=resolved_dirs if len(resolved_dirs) > 1 else None,
        )
    markdown = _format_overview_markdown(sim_data_list, project_root)
    return {
        "status": "success",
        "message": "Overview summary created.",
        "artifact_type": "markdown",
        "artifact_content": markdown,
        "artifact_title": "Overview",
    }
