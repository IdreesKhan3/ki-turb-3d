"""
Theory Equations page tools: NS equations, LBM formulation, D3Q19 lattice, MRT matrix.
Produces markdown or plotly_figure artifacts matching the Theory & Equations page content.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from content.theory_equations_content import (
    get_ns_equations_markdown,
    get_lbm_formulation_markdown,
)


# Default D3Q19 directions from OpenACC_parameters.F90 (used by MRT matrix and page)
DEFAULT_D3Q19_DIRX = [1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0]
DEFAULT_D3Q19_DIRY = [0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0]
DEFAULT_D3Q19_DIRZ = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for Theory Equations page."""
    return [
        {
            "name": "get_theory_ns_equations",
            "description": "Show Navier-Stokes equations and filtered NS (LES) from Theory & Equations page. Use when user asks for 'NS equations', 'Navier-Stokes', 'filtered NS', 'LES equations', 'continuity', 'momentum equations'.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_theory_lbm_formulation",
            "description": "Show LBM formulation (MRT DNS/LES, BGK/SRT, equilibrium, forcing, validation) from Theory & Equations page. Use when user asks for 'LBM formulation', 'MRT equations', 'BGK', 'SRT', 'lattice Boltzmann', 'equilibrium distribution', 'Guo forcing', 'compressibility', 'Mach', 'Knudsen', 'Reynolds'.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "plot_d3q19_lattice",
            "description": "Produce D3Q19 lattice stencil 3D visualization. Use when user asks for 'D3Q19 lattice', 'lattice visualization', 'lattice stencil'. All params optional—use when user asks for custom appearance (e.g. 'longer vectors', 'bigger nodes', 'dark background', 'front view').",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_vectors": {"type": "boolean", "description": "Show velocity vectors. Default true."},
                    "vector_scale": {"type": "number", "description": "Vector length scale (0.1–2.0). Default 1.0. Use when 'longer/shorter vectors'."},
                    "vector_width": {"type": "number", "description": "Vector line width (1–10). Default 3.0."},
                    "show_labels": {"type": "boolean", "description": "Show direction labels (C1, C2, ...). Default true."},
                    "label_prefix": {"type": "string", "description": "Label prefix (e.g. 'C' for C1, C2). Default 'C'."},
                    "label_font_size": {"type": "integer", "description": "Label font size (8–24). Default 13."},
                    "label_color": {"type": "string", "description": "Label color (hex e.g. '#000000'). Default black."},
                    "node_size": {"type": "number", "description": "Node marker size (5–50). Default 10. Use when 'bigger/smaller nodes'."},
                    "node_style": {"type": "string", "enum": ["circle", "circle-open", "square", "square-open", "diamond", "diamond-open", "cross", "x"], "description": "Node marker style. Default circle."},
                    "node_opacity": {"type": "number", "description": "Node opacity (0–1). Default 0.8."},
                    "vector_color": {"type": "string", "description": "Vector color (hex e.g. '#FF0000'). Default red."},
                    "vector_opacity": {"type": "number", "description": "Vector opacity (0–1). Default 0.8."},
                    "vector_linestyle": {"type": "string", "enum": ["solid", "dash", "dot", "dashdot"], "description": "Vector line style. Default dashdot."},
                    "show_faces": {"type": "boolean", "description": "Show colored faces. Default false."},
                    "face_opacity": {"type": "number", "description": "Face opacity (0–1). Default 0.5."},
                    "show_cube_edges": {"type": "boolean", "description": "Show cube edges. Default true."},
                    "cube_edge_color": {"type": "string", "description": "Cube edge color (hex). Default black."},
                    "cube_edge_width": {"type": "number", "description": "Cube edge width (0.5–5). Default 2.0."},
                    "show_grid": {"type": "boolean", "description": "Show grid. Default false."},
                    "background_color": {"type": "string", "description": "Background color (hex). Use '#1e1e1e' for dark. Default white."},
                    "show_axes": {"type": "boolean", "description": "Show coordinate axes. Default false."},
                    "show_axis_labels": {"type": "boolean", "description": "Show axis labels. Default false."},
                    "show_origin_marker": {"type": "boolean", "description": "Show origin marker. Default true."},
                    "camera_elevation": {"type": "number", "description": "Camera elevation in degrees (-90–90). 0=front, 90=top. Default 9."},
                    "camera_azimuth": {"type": "number", "description": "Camera azimuth in degrees (-180–180). 0=front, 90=side. Default 16."},
                    "camera_zoom": {"type": "number", "description": "Camera zoom (0.5–3). Default 1.0."},
                    "origin_size": {"type": "number", "description": "Origin marker size (5–50). Default 15."},
                    "origin_color": {"type": "string", "description": "Origin marker color (hex). Default '#052020'."},
                    "origin_style": {"type": "string", "enum": ["circle", "circle-open", "square", "square-open", "diamond", "diamond-open", "cross", "x"], "description": "Origin marker style. Default circle-open."},
                    "node_edge_color": {"type": "string", "description": "Node edge color (hex). Default black."},
                    "node_edge_width": {"type": "number", "description": "Node edge width (0–5). Default 1.0."},
                },
            },
        },
        {
            "name": "get_theory_mrt_matrix",
            "description": "Show MRT transformation matrix M, inverse M⁻¹, and relaxation vector S for D3Q19. Use when user asks for 'MRT matrix', 'transformation matrix', 'M matrix', 'relaxation rates', 'D3Q19 matrix'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu": {"type": "number", "description": "Kinematic viscosity (optional). Default 0.002546479089469996."},
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
    """Execute a Theory Equations tool."""
    if name == "get_theory_ns_equations":
        markdown = get_ns_equations_markdown()
        return {
            "status": "success",
            "message": "NS equations content created.",
            "artifact_type": "markdown",
            "artifact_content": markdown,
            "artifact_title": "NS Equations",
        }

    if name == "get_theory_lbm_formulation":
        markdown = get_lbm_formulation_markdown()
        return {
            "status": "success",
            "message": "LBM formulation content created.",
            "artifact_type": "markdown",
            "artifact_content": markdown,
            "artifact_title": "LBM Formulation",
        }

    if name == "plot_d3q19_lattice":
        from visualizations.d3q19_lattice import plot_d3q19_lattice, DEFAULT_LATTICE_COLORS

        # Defaults (match Theory Equations page)
        defaults = {
            "show_vectors": True,
            "vector_scale": 1.0,
            "vector_width": 3.0,
            "node_size": 10.0,
            "node_colors": DEFAULT_LATTICE_COLORS.copy(),
            "node_opacity": 0.8,
            "node_style": "circle",
            "node_edge_color": "#000000",
            "node_edge_width": 1.0,
            "origin_size": 15.0,
            "origin_color": "#052020",
            "origin_style": "circle-open",
            "vector_color": "#FF0000",
            "vector_opacity": 0.8,
            "vector_linestyle": "dashdot",
            "show_labels": True,
            "label_prefix": "C",
            "label_font_size": 13,
            "label_color": "#000000",
            "show_faces": False,
            "face_opacity": 0.5,
            "show_cube_edges": True,
            "cube_edge_color": "#000000",
            "cube_edge_width": 2.0,
            "show_grid": False,
            "background_color": "#FFFFFF",
            "show_axes": False,
            "show_axis_labels": False,
            "show_origin_marker": True,
            "camera_elevation": 9.0,
            "camera_azimuth": 16.0,
            "camera_zoom": 1.0,
            "width": 800,
            "height": 800,
            "title": "D3Q19 Lattice Stencil",
        }
        # Override with args (only pass keys that plot_d3q19_lattice accepts)
        plot_params = defaults.copy()
        for key in [
            "show_vectors", "vector_scale", "vector_width", "node_size", "node_opacity",
            "node_style", "node_edge_color", "node_edge_width", "origin_size", "origin_color",
            "origin_style", "vector_color", "vector_opacity", "vector_linestyle",
            "show_labels", "label_prefix", "label_font_size", "label_color",
            "show_faces", "face_opacity", "show_cube_edges", "cube_edge_color", "cube_edge_width",
            "show_grid", "background_color", "show_axes", "show_axis_labels", "show_origin_marker",
            "camera_elevation", "camera_azimuth", "camera_zoom", "width", "height", "title",
        ]:
            if key in args and args[key] is not None:
                plot_params[key] = args[key]
        fig = plot_d3q19_lattice(**plot_params)
        return {
            "status": "success",
            "message": "D3Q19 lattice visualization created.",
            "artifact_type": "plotly_figure",
            "artifact_content": fig.to_json(),
            "artifact_title": "D3Q19 Lattice Stencil",
        }

    if name == "get_theory_mrt_matrix":
        from utils.mrt_matrix import compute_mrt_matrix, compute_relaxation_vector, validate_d3q19_directions

        nu = args.get("nu", 0.002546479089469996)
        tau = 3.0 * nu + 0.5

        is_valid, errors, _ = validate_d3q19_directions(
            DEFAULT_D3Q19_DIRX, DEFAULT_D3Q19_DIRY, DEFAULT_D3Q19_DIRZ
        )
        if not is_valid:
            return f"Error: Invalid D3Q19 directions. {errors[0] if errors else 'Unknown error'}"

        M, M_inv, identity_error_max = compute_mrt_matrix(
            DEFAULT_D3Q19_DIRX, DEFAULT_D3Q19_DIRY, DEFAULT_D3Q19_DIRZ
        )
        S = compute_relaxation_vector(tau=tau, nu=nu)

        moment_names = [
            "ρ (density)", "e (energy)", "ε (higher-order energy)",
            "jx (momentum x)", "qx (energy flux x)", "jy (momentum y)", "qy (energy flux y)",
            "jz (momentum z)", "qz (energy flux z)", "3pxx", "3πxx", "pww", "πww",
            "pxy", "pyz", "pxz", "mx", "my", "mz",
        ]

        lines = [
            "# MRT Matrix Generator (D3Q19)",
            "",
            f"**Parameters:** ν = {nu:.6e}, τ = {tau:.6f}",
            f"**Inversion error (max |M×M⁻¹ − I|):** {identity_error_max:.2e}",
            "",
            "## Relaxation Rate Vector S",
            "",
            "| Moment | Relaxation Rate |",
            "|--------|-----------------|",
        ]
        for i, (mom, s) in enumerate(zip(moment_names, S)):
            lines.append(f"| {mom} | {s:.6f} |")

        lines.extend([
            "",
            "## Transformation Matrix M (19×19)",
            "",
            "Matrix M transforms distribution functions f to moment space: **m = M f**",
            "",
        ])

        # Build markdown table for M (header row)
        header = "| | " + " | ".join(f"a={i+1}" for i in range(19)) + " |"
        sep = "|" + "---|" * 20
        lines.append(header)
        lines.append(sep)
        for i in range(19):
            row_vals = [f"{M[i, j]:.0f}" for j in range(19)]
            lines.append(f"| {moment_names[i]} | " + " | ".join(row_vals) + " |")

        lines.extend([
            "",
            "## Inverse Matrix M⁻¹ (19×19)",
            "",
            "Transforms moments back to distribution functions: **f = M⁻¹ m**",
            "",
        ])

        header_inv = "| | " + " | ".join(f"a={i+1}" for i in range(19)) + " |"
        lines.append(header_inv)
        lines.append(sep)
        for i in range(19):
            row_vals = [f"{M_inv[i, j]:.6f}" for j in range(19)]
            lines.append(f"| {moment_names[i]} | " + " | ".join(row_vals) + " |")

        markdown = "\n".join(lines)
        return {
            "status": "success",
            "message": "MRT matrix content created.",
            "artifact_type": "markdown",
            "artifact_content": markdown,
            "artifact_title": "MRT Matrix (D3Q19)",
        }

    return f"Error: Unknown theory equations tool '{name}'"
