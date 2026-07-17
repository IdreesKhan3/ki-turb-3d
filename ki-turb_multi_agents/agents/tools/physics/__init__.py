"""
Physics tools: spectra, real isotropy, spectral isotropy, and future page-specific tools.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union



PHYSICS_TOOL_NAMES = frozenset({
    "load_analysis_products", "get_analysis_product_summary",
    "compute_overview_validation",
    "compute_pdfs", "compute_volume_field",
    "compute_spectra", "plot_spectrum", "export_figure", "export_data", "get_energy_spectra_theory",
    "compute_isotropy", "compute_spectral_isotropy", "plot_spectral_isotropy", "plot_component_spectra",
    "get_spectral_isotropy_summary", "get_spectral_isotropy_theory", "plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii",
    "plot_cross_correlations", "plot_deviations", "plot_convergence", "get_real_isotropy_summary",
    "get_real_isotropy_theory", "export_isotropy_data",
    "get_overview_summary", "get_overview_theory",
    "get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice", "get_theory_mrt_matrix",
    "compute_flatness", "plot_flatness", "get_flatness_summary", "get_flatness_theory", "export_flatness_data",
    "compute_structure_functions",
    "plot_structure_functions",
    "get_structure_functions_theory",
    "plot_turbulence_stats",
    "get_turbulence_stats_columns",
    "get_turbulence_stats_summary",
    "plot_volume_3d",
    "get_volume_viewer_theory",
    "plot_pdf",
    "add_report_section",
    "generate_report",
    "preview_report",
    "remove_report_section",
    "reorder_report_section",
    "edit_report_section",
})


def _modules():
    """Import UI-backed physics tools lazily when a role requests their definitions."""
    from . import spectra, real_isotropy, spectral_isotropy, overview, theory_equations, flatness, structure_functions, turbulence_stats, volume_viewer, pdfs, report_generator, analysis_products
    return {
        "spectra": spectra, "real_isotropy": real_isotropy, "spectral_isotropy": spectral_isotropy,
        "overview": overview, "theory_equations": theory_equations, "flatness": flatness,
        "structure_functions": structure_functions, "turbulence_stats": turbulence_stats,
        "volume_viewer": volume_viewer, "pdfs": pdfs, "report_generator": report_generator,
        "analysis_products": analysis_products,
    }


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for physics tools."""
    m = _modules()
    spectra=m["spectra"];real_isotropy=m["real_isotropy"];spectral_isotropy=m["spectral_isotropy"];overview=m["overview"];theory_equations=m["theory_equations"];flatness=m["flatness"];structure_functions=m["structure_functions"];turbulence_stats=m["turbulence_stats"];volume_viewer=m["volume_viewer"];pdfs=m["pdfs"];report_generator=m["report_generator"];analysis_products=m["analysis_products"]
    tools = []
    tools.extend(analysis_products.get_tool_definitions())
    tools.extend(spectra.get_tool_definitions())
    tools.extend(real_isotropy.get_tool_definitions())
    tools.extend(spectral_isotropy.get_tool_definitions())
    tools.extend(overview.get_tool_definitions())
    tools.extend(theory_equations.get_tool_definitions())
    tools.extend(flatness.get_tool_definitions())
    tools.extend(structure_functions.get_tool_definitions())
    tools.extend(turbulence_stats.get_tool_definitions())
    tools.extend(volume_viewer.get_tool_definitions())
    tools.extend(pdfs.get_tool_definitions())
    tools.extend(report_generator.get_tool_definitions())
    return tools


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    """Execute a physics tool."""
    m = _modules()
    spectra=m["spectra"];real_isotropy=m["real_isotropy"];spectral_isotropy=m["spectral_isotropy"];overview=m["overview"];theory_equations=m["theory_equations"];flatness=m["flatness"];structure_functions=m["structure_functions"];turbulence_stats=m["turbulence_stats"];volume_viewer=m["volume_viewer"];pdfs=m["pdfs"];report_generator=m["report_generator"];analysis_products=m["analysis_products"]
    if name in ("load_analysis_products", "get_analysis_product_summary"):
        return analysis_products.execute_tool(name, args, project_root, session_context or {})
    if name in ("compute_spectra", "plot_spectrum", "export_figure", "export_data", "get_energy_spectra_theory"):
        return spectra.execute_tool(name, args, project_root, session_context)
    if name in ("compute_isotropy", "plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations", "plot_convergence", "get_real_isotropy_summary", "get_real_isotropy_theory"):
        return real_isotropy.execute_tool(name, args, project_root, session_context or {})
    if name in ("compute_spectral_isotropy", "plot_spectral_isotropy", "plot_component_spectra", "get_spectral_isotropy_summary", "get_spectral_isotropy_theory", "export_isotropy_data"):
        return spectral_isotropy.execute_tool(name, args, project_root, session_context or {})
    if name in ("get_overview_summary", "get_overview_theory", "compute_overview_validation"):
        return overview.execute_tool(name, args, project_root, session_context or {})
    if name in ("get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice", "get_theory_mrt_matrix"):
        return theory_equations.execute_tool(name, args, project_root, session_context or {})
    if name in ("compute_flatness", "plot_flatness", "get_flatness_summary", "get_flatness_theory", "export_flatness_data"):
        return flatness.execute_tool(name, args, project_root, session_context or {})
    if name in ("compute_structure_functions", "plot_structure_functions", "get_structure_functions_theory"):
        return structure_functions.execute_tool(name, args, project_root, session_context or {})
    if name in ("plot_turbulence_stats", "get_turbulence_stats_columns", "get_turbulence_stats_summary"):
        return turbulence_stats.execute_tool(name, args, project_root, session_context or {})
    if name in ("plot_volume_3d", "get_volume_viewer_theory", "compute_volume_field"):
        return volume_viewer.execute_tool(name, args, project_root, session_context or {})
    if name in ("plot_pdf", "compute_pdfs"):
        return pdfs.execute_tool(name, args, project_root, session_context or {})
    if name in ("add_report_section", "generate_report", "preview_report",
                "remove_report_section", "reorder_report_section", "edit_report_section"):
        return report_generator.execute_tool(name, args, project_root, session_context or {})
    return f"Error: Unknown physics tool '{name}'"
