"""Map canonical dataset-manifest file kinds to session file-index keys.

This table is backend-neutral: OpenLB, Palabos, Ansys, and legacy example
directories all use the same manifest vocabulary once data is fetched.
"""

from __future__ import annotations

from typing import Dict

# Manifest kind → key in session_context["all_loaded_files"] / file_detector dict
MANIFEST_KIND_TO_SESSION_KEY: Dict[str, str] = {
    "velocity_field": "velocity_files",
    "pressure_field": "pressure_field",
    "density_field": "density_field",
    "vorticity_field": "vorticity_field",
    "forcing_field": "forcing_field",
    "checkpoint": "checkpoint",
    "diagnostics": "diagnostics",
    "energy_spectrum": "spectrum",
    "normalized_spectrum": "norm_spectrum",
    "spectral_isotropy": "isotropy",
    "component_spectra": "component_spectra",
    "flatness": "flatness",
    "structure_functions": "structure_functions_txt",
    "velocity_pdf": "velocity_pdf",
    "gradient_pdf": "gradient_pdf",
    "dissipation_pdf": "dissipation_pdf",
    "enstrophy_pdf": "enstrophy_pdf",
    "joint_pdf": "joint_pdf",
    "rq_pdf": "rq_pdf",
    "turbulence_stats": "real_turb_stats",
    "reynolds_stress": "reynolds_stress",
    "dissipation_validation": "spectral_turb_stats",
    "energy_balance": "energy_balance",
    "analysis_products": "analysis_products",
    "tau_effective_field": "tau_analysis",
    "figure": "figure",
    "log": "log",
    "metadata": "metadata",
}

# Optional aliases used by legacy file_detector keys
SESSION_KEY_ALIASES: Dict[str, str] = {
    "velocity_vti": "velocity_files",
    "velocity_h5": "velocity_files",
}

__all__ = ["MANIFEST_KIND_TO_SESSION_KEY", "SESSION_KEY_ALIASES"]
