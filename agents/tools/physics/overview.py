"""
Overview page tools: simulation parameters, physics validation, data availability.
Produces markdown/table artifacts matching the Overview page content.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.file_detector import detect_simulation_files

from content.overview_theory_content import get_overview_theory_markdown


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


def _is_les_dir(data_dir: Path, project_root: Path) -> bool:
    """Strict path-based LES detection: only examples/LES/*."""
    try:
        rel = data_dir.resolve().relative_to(project_root.resolve())
        parts = [p.lower() for p in rel.parts]
        return len(parts) >= 2 and parts[0] == "examples" and parts[1] == "les"
    except Exception:
        return False


def _compute_overview_data(data_dirs: List[Path], project_root: Path) -> List[Dict[str, Any]]:
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
        files = detect_simulation_files(str(data_dir))
        dir_name = data_dir.name
        sim_data = {
            "directory": dir_name,
            "path": str(data_dir),
            "files": files,
            "params": None,
            "raw_params": None,
            "mach_number": None,
            "knudsen_number": None,
            "compressibility": None,
            "is_les": _is_les_dir(data_dir, project_root),
            "is_lbm": None,  # True=LBM, False=NS
        }

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
                import streamlit as st
                from data_readers.hdf5_reader import read_hdf5_file
                fortran_order = st.session_state.get("hdf5_fortran_order", True)
                data = read_hdf5_file(str(files["velocity_h5"][0]), fortran_order=fortran_order)
                vel = data["velocity"]  # (z, y, x, 3)
                if vel.size > 128**3:
                    z_mid = vel.shape[0] // 2
                    vel = vel[z_mid : z_mid + 1, :, :, :]
                vel = np.transpose(vel, (2, 1, 0, 3))  # -> (nx, ny, nz, 3) for compute_compressibility
                comp = compute_compressibility_h5(vel)
                if comp:
                    sim_data["compressibility"] = comp
            except Exception:
                pass

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


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [OVERVIEW_TOOL, OVERVIEW_THEORY_TOOL]


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

    if name != "get_overview_summary":
        return f"Error: Unknown overview tool '{name}'"

    session_context = session_context or {}
    data_dir = args.get("data_dir", "")
    data_directories = args.get("data_directories")

    dirs = _resolve_overview_dirs(data_dir, data_directories, project_root, session_context)
    if not dirs:
        return "Error: No data directory. Provide data_dir or data_directories, or load data from the main page sidebar."

    sim_data_list = _compute_overview_data(dirs, project_root)
    if not sim_data_list:
        return "Error: Could not process any directory."

    from pages.AutonomousLab.session_sync import update_data_directory_in_context
    update_data_directory_in_context(session_context, Path(dirs[0]).resolve())
    markdown = _format_overview_markdown(sim_data_list, project_root)
    return {
        "status": "success",
        "message": "Overview summary created.",
        "artifact_type": "markdown",
        "artifact_content": markdown,
        "artifact_title": "Overview",
    }
