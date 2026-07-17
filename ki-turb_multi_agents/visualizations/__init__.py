"""Lazy visualization exports.

The package must remain importable in headless solver/agent environments where
optional UI dependencies such as Streamlit are not installed.  Individual
visualization modules are loaded only when their public symbol is requested.
"""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "plot_d3q19_lattice": (".d3q19_lattice", "plot_d3q19_lattice"),
    "DEFAULT_LATTICE_COLORS": (".d3q19_lattice", "DEFAULT_LATTICE_COLORS"),
    "create_spectrum_figure": (".spectra_vis", "create_spectrum_figure"),
    "create_raw_spectrum_figure": (".spectra_vis", "create_raw_spectrum_figure"),
    "create_normalized_spectrum_figure": (".spectra_vis", "create_normalized_spectrum_figure"),
    "create_time_evolution_figure": (".spectra_vis", "create_time_evolution_figure"),
    "add_kolmogorov_line": (".spectra_vis", "add_kolmogorov_line"),
    "create_ic_isotropy_figure": (".spectral_isotropy_vis", "create_ic_isotropy_figure"),
    "create_component_spectra_figure": (".spectral_isotropy_vis", "create_component_spectra_figure"),
    "create_energy_fractions_figure": (".real_isotropy_vis", "create_energy_fractions_figure"),
    "create_lumley_triangle_figure": (".real_isotropy_vis", "create_lumley_triangle_figure"),
    "create_diagonal_bii_figure": (".real_isotropy_vis", "create_diagonal_bii_figure"),
    "create_cross_correlations_figure": (".real_isotropy_vis", "create_cross_correlations_figure"),
    "create_deviations_figure": (".real_isotropy_vis", "create_deviations_figure"),
    "create_convergence_figure": (".real_isotropy_vis", "create_convergence_figure"),
    "create_sp_figure": (".structure_functions_vis", "create_sp_figure"),
    "create_ess_figure": (".structure_functions_vis", "create_ess_figure"),
    "create_anomalies_figure": (".structure_functions_vis", "create_anomalies_figure"),
    "create_velocity_components_pdf_figure": (".pdfs_vis", "create_velocity_components_pdf_figure"),
    "create_1d_pdf_figure": (".pdfs_vis", "create_1d_pdf_figure"),
    "create_2d_contour_pdf_figure": (".pdfs_vis", "create_2d_contour_pdf_figure"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
