"""
Core physics module — pure mathematical logic for turbulence analysis.

Shared by manual UI pages and AI agents. No Streamlit or UI dependencies.
"""

# 04_Real_Isotropy
from .real_isotropy import (
    load_turbulence_data,
    load_reynolds_stress,
    compute_reynolds_from_fractions,
    anisotropy_tensor,
    invariants,
)
# 05_Spectral_Isotropy
from .spectral_isotropy import (
    read_isotropy_coeff_file,
    avg_isotropy_coeff,
)
# 06_Energy_Spectra
from .spectra import (
    compute_spectrum_time_avg,
    compute_spectrum_time_avg_norm,
)
# 07_Flatness
from .flatness import compute_flatness_time_avg
# 08_Structure_Functions
from .structure_functions import (
    zeta_p_she_leveque,
    TABLE_P,
    EXP_ZETA,
    compute_structure_time_avg,
)
# 09_PDFs (pages/PDFs)
from .pdfs import (
    compute_skewness_kurtosis,
    compute_velocity_magnitude_pdf,
    compute_velocity_magnitude_statistics,
    compute_velocity_pdf,
    compute_velocity_component_statistics,
    compute_dissipation_pdf,
    compute_dissipation_statistics,
    compute_vorticity_pdf,
    compute_vorticity_statistics,
    compute_enstrophy_pdf,
    compute_enstrophy_statistics,
    compute_velocity_dissipation_joint_pdf,
    compute_velocity_enstrophy_joint_pdf,
    compute_dissipation_enstrophy_joint_pdf,
    compute_rq_joint_pdf,
    compute_discriminant_line,
)

__all__ = [
    # 04_Real_Isotropy
    "load_turbulence_data",
    "load_reynolds_stress",
    "compute_reynolds_from_fractions",
    "anisotropy_tensor",
    "invariants",
    # 05_Spectral_Isotropy
    "read_isotropy_coeff_file",
    "avg_isotropy_coeff",
    # 06_Energy_Spectra
    "compute_spectrum_time_avg",
    "compute_spectrum_time_avg_norm",
    # 07_Flatness
    "compute_flatness_time_avg",
    # 08_Structure_Functions
    "zeta_p_she_leveque",
    "TABLE_P",
    "EXP_ZETA",
    "compute_structure_time_avg",
    # 09_PDFs (pages/PDFs)
    "compute_skewness_kurtosis",
    "compute_velocity_magnitude_pdf",
    "compute_velocity_magnitude_statistics",
    "compute_velocity_pdf",
    "compute_velocity_component_statistics",
    "compute_dissipation_pdf",
    "compute_dissipation_statistics",
    "compute_vorticity_pdf",
    "compute_vorticity_statistics",
    "compute_enstrophy_pdf",
    "compute_enstrophy_statistics",
    "compute_velocity_dissipation_joint_pdf",
    "compute_velocity_enstrophy_joint_pdf",
    "compute_dissipation_enstrophy_joint_pdf",
    "compute_rq_joint_pdf",
    "compute_discriminant_line",
]
