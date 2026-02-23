"""
Visualization modules
Plot generation functions
"""

# Only import what's actually used in the app
from .d3q19_lattice import plot_d3q19_lattice, DEFAULT_LATTICE_COLORS
from .spectra_vis import (
    create_spectrum_figure,
    create_raw_spectrum_figure,
    create_normalized_spectrum_figure,
    create_time_evolution_figure,
    add_kolmogorov_line,
)
from .spectral_isotropy_vis import (
    create_ic_isotropy_figure,
    create_component_spectra_figure,
)
from .real_isotropy_vis import (
    create_energy_fractions_figure,
    create_lumley_triangle_figure,
    create_diagonal_bii_figure,
    create_cross_correlations_figure,
    create_deviations_figure,
    create_convergence_figure,
)
from .structure_functions_vis import (
    create_sp_figure,
    create_ess_figure,
    create_anomalies_figure,
)
from .pdfs_vis import (
    create_velocity_components_pdf_figure,
    create_1d_pdf_figure,
    create_2d_contour_pdf_figure,
)

__all__ = [
    # spectra page related
    'plot_d3q19_lattice',
    'DEFAULT_LATTICE_COLORS',
    'create_spectrum_figure',
    'create_raw_spectrum_figure',
    'create_normalized_spectrum_figure',
    'create_time_evolution_figure',
    'add_kolmogorov_line',
    # spectral isotropy
    'create_ic_isotropy_figure',
    'create_component_spectra_figure',
    # real isotropy
    'create_energy_fractions_figure',
    'create_lumley_triangle_figure',
    'create_diagonal_bii_figure',
    'create_cross_correlations_figure',
    'create_deviations_figure',
    'create_convergence_figure',
    'create_sp_figure',
    'create_ess_figure',
    'create_anomalies_figure',
    'create_velocity_components_pdf_figure',
    'create_1d_pdf_figure',
    'create_2d_contour_pdf_figure',
]

