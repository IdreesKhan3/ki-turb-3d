"""
Physics tools: spectra, real isotropy, spectral isotropy, and future page-specific tools.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import spectra, real_isotropy, spectral_isotropy, overview, theory_equations


PHYSICS_TOOL_NAMES = frozenset({
    "compute_spectra", "plot_spectrum", "export_figure", "export_data",
    "compute_isotropy", "compute_spectral_isotropy", "plot_spectral_isotropy", "plot_component_spectra",
    "get_spectral_isotropy_summary", "get_spectral_isotropy_theory", "plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii",
    "plot_cross_correlations", "plot_deviations", "plot_convergence", "get_real_isotropy_summary",
    "get_real_isotropy_theory", "export_isotropy_data",
    "get_overview_summary", "get_overview_theory",
    "get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice", "get_theory_mrt_matrix",
})


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for physics tools."""
    tools = []
    tools.extend(spectra.get_tool_definitions())
    tools.extend(real_isotropy.get_tool_definitions())
    tools.extend(spectral_isotropy.get_tool_definitions())
    tools.extend(overview.get_tool_definitions())
    tools.extend(theory_equations.get_tool_definitions())
    return tools


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    """Execute a physics tool."""
    if name in ("compute_spectra", "plot_spectrum", "export_figure", "export_data"):
        return spectra.execute_tool(name, args, project_root, session_context or {})
    if name in ("compute_isotropy", "plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations", "plot_convergence", "get_real_isotropy_summary", "get_real_isotropy_theory"):
        return real_isotropy.execute_tool(name, args, project_root, session_context or {})
    if name in ("compute_spectral_isotropy", "plot_spectral_isotropy", "plot_component_spectra", "get_spectral_isotropy_summary", "get_spectral_isotropy_theory", "export_isotropy_data"):
        return spectral_isotropy.execute_tool(name, args, project_root, session_context or {})
    if name in ("get_overview_summary", "get_overview_theory"):
        return overview.execute_tool(name, args, project_root, session_context or {})
    if name in ("get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice", "get_theory_mrt_matrix"):
        return theory_equations.execute_tool(name, args, project_root, session_context or {})
    return f"Error: Unknown physics tool '{name}'"
