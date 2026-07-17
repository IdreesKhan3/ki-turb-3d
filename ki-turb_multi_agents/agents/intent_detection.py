"""
Intent Detection — Deterministic intent parser for plot/tool routing.

Parses user input to detect intent; routing (tool, file_pattern, skip_analyst, overrides)
is derived from agents.page_schema. Single source of truth for all page/tool mappings.

ORGANIZATION: Pattern checks are sectionized by page:
    # =============================================================================
    # PAGE NN — PAGE_NAME
    # =============================================================================
"""

import re
from typing import Any, Dict, List, Optional

from .page_schema import (
    INTENT_APP_SETTINGS_HDF5_FORTRAN,
    INTENT_APP_SETTINGS_HDF5_DEFAULT,
    INTENT_OVERVIEW,
    INTENT_OVERVIEW_THEORY,
    INTENT_COMPONENT_SPECTRA,
    INTENT_CONVERGENCE,
    INTENT_CROSS_CORRELATIONS,
    INTENT_DEVIATIONS,
    INTENT_DIAGONAL_BII,
    INTENT_ENERGY_FRACTIONS,
    INTENT_REAL_ISOTROPY_SUMMARY,
    INTENT_REAL_ISOTROPY_THEORY,
    INTENT_ENERGY_SPECTRA,
    INTENT_ENERGY_SPECTRA_THEORY,
    INTENT_FLATNESS,
    INTENT_FLATNESS_THEORY,
    INTENT_LUMLEY_TRIANGLE,
    INTENT_PDF,
    INTENT_SPECTRAL_ISOTROPY,
    INTENT_SPECTRAL_ISOTROPY_SUMMARY,
    INTENT_SPECTRAL_ISOTROPY_THEORY,
    INTENT_STRUCTURE_FUNCTIONS,
    INTENT_STRUCTURE_FUNCTIONS_THEORY,
    INTENT_OTHER_TURBULENCE_STATS,
    INTENT_OTHER_TURBULENCE_STATS_SUMMARY,
    INTENT_VOLUME_VIEWER_3D,
    INTENT_VOLUME_VIEWER_3D_THEORY,
    INTENT_REPORT_ADD_SECTION,
    INTENT_REPORT_REMOVE,
    INTENT_REPORT_REORDER,
    INTENT_REPORT_EDIT,
    INTENT_REPORT_GENERATE,
    INTENT_REPORT_PREVIEW,
    INTENT_THEORY_D3Q19_LATTICE,
    INTENT_THEORY_EQUATIONS_FULL,
    INTENT_THEORY_LBM_FORMULATION,
    INTENT_THEORY_MRT_MATRIX,
    INTENT_THEORY_NS_EQUATIONS,
    INTENT_UNKNOWN,
    get_routing_for_intent,
)

# Re-export for backward compatibility
__all__ = [
    "INTENT_OVERVIEW",
    "INTENT_THEORY_NS_EQUATIONS",
    "INTENT_THEORY_LBM_FORMULATION",
    "INTENT_THEORY_D3Q19_LATTICE",
    "INTENT_THEORY_MRT_MATRIX",
    "INTENT_THEORY_EQUATIONS_FULL",
    "INTENT_LUMLEY_TRIANGLE",
    "INTENT_DIAGONAL_BII",
    "INTENT_CROSS_CORRELATIONS",
    "INTENT_DEVIATIONS",
    "INTENT_CONVERGENCE",
    "INTENT_REAL_ISOTROPY_SUMMARY",
    "INTENT_ENERGY_FRACTIONS",
    "INTENT_SPECTRAL_ISOTROPY",
    "INTENT_SPECTRAL_ISOTROPY_SUMMARY",
    "INTENT_COMPONENT_SPECTRA",
    "INTENT_ENERGY_SPECTRA",
    "INTENT_FLATNESS",
    "INTENT_FLATNESS_THEORY",
    "INTENT_STRUCTURE_FUNCTIONS",
    "INTENT_OTHER_TURBULENCE_STATS",
    "INTENT_OTHER_TURBULENCE_STATS_SUMMARY",
    "INTENT_VOLUME_VIEWER_3D",
    "INTENT_VOLUME_VIEWER_3D_THEORY",
    "INTENT_REPORT_ADD_SECTION",
    "INTENT_REPORT_REMOVE",
    "INTENT_REPORT_REORDER",
    "INTENT_REPORT_EDIT",
    "INTENT_REPORT_GENERATE",
    "INTENT_APP_SETTINGS_HDF5_FORTRAN",
    "INTENT_APP_SETTINGS_HDF5_DEFAULT",
    "INTENT_PDF",
    "INTENT_UNKNOWN",
    "get_analysis_intent",
    "collect_analysis_intents",
    "get_plot_routing",
]

def _has_spectrum_keywords(t: str) -> bool:
    """True when the user is asking about energy spectra E(k), not merely the word 'spectral'."""
    # Use word-boundary checks so "spectral isotropy" does not count as "spectra".
    # Also accept tool names (underscores are word chars, so \bspectra\b misses them).
    return bool(
        re.search(r"\b(spectrum|spectra|e\s*\(\s*k\s*\)|kolmogorov|-5/3|5/3)\b", t)
        or re.search(r"\b(compute_spectra|plot_spectrum)\b", t)
        or re.search(r"\bevolution\b", t)  # "time evolution", "evolution spectra"
    )



# =============================================================================
# APP SETTINGS — HDF5 format (fortran vs default)
# =============================================================================

def _check_app_settings_hdf5(t: str) -> Optional[str]:
    """Check app settings: HDF5 format (fortran vs default). Skip when user also asks to load data."""
    if "hdf5" not in t and "h5" not in t:
        return None
    # If user asks to load data + format, orchestrator handles via prompt (two delegations)
    if "load" in t and ("data" in t or "directory" in t or "dns" in t or "les" in t or "examples" in t):
        return None
    # Fortran: fortran, transpose
    if any(x in t for x in ["fortran", "transpose", "openacc", "fortran-written"]):
        return INTENT_APP_SETTINGS_HDF5_FORTRAN
    # Default: default, no transpose, python, standard
    if any(x in t for x in ["default", "no transpose", "python", "standard layout", "standard hdf5"]):
        return INTENT_APP_SETTINGS_HDF5_DEFAULT
    return None


# =============================================================================
# PAGE 01 — OVERVIEW (parameters, physics validation, data availability)
# =============================================================================

def _check_p01_overview(t: str) -> Optional[str]:
    """Check Page 01 (Overview) intents."""
    # Overview theory/equations — must check before generic overview
    if ("theory" in t or "equations" in t) and "overview" in t:
        return INTENT_OVERVIEW_THEORY
    if "physics validation equations" in t:
        return INTENT_OVERVIEW_THEORY
    if "overview" in t:
        return INTENT_OVERVIEW
    if "parameters" in t and ("simulation" in t or "metadata" in t or "input" in t):
        return INTENT_OVERVIEW
    if "physics validation" in t or "mach number" in t or "knudsen number" in t:
        return INTENT_OVERVIEW
    if "data availability" in t or "what files" in t or "which files" in t:
        return INTENT_OVERVIEW
    if "compressibility" in t and ("validation" in t or "divergence" in t):
        return INTENT_OVERVIEW
    return None


# =============================================================================
# PAGE 02 — THEORY & EQUATIONS (NS equations, LBM, D3Q19, MRT matrix)
# Check before real isotropy so "NS equations" etc. route here, not to real isotropy theory
# =============================================================================

def _check_p02_theory_equations(t: str) -> Optional[str]:
    """Check Page 02 (Theory & Equations) intents. Returns intent or None."""
    # NS equations — Navier-Stokes, filtered NS, LES
    if re.search(r"\bns\s*equation", t) or re.search(r"\bnavier[- ]?stokes\b", t):
        return INTENT_THEORY_NS_EQUATIONS
    if "filtered ns" in t or "les equation" in t or "continuity" in t and "momentum" in t:
        return INTENT_THEORY_NS_EQUATIONS

    # D3Q19 lattice — stencil, visualization
    if re.search(r"\bd3q19\s*lattice\b", t) or "lattice stencil" in t or "lattice visualization" in t:
        return INTENT_THEORY_D3Q19_LATTICE
    if "d3q19" in t and ("stencil" in t or "visualization" in t or "plot" in t or "show" in t):
        return INTENT_THEORY_D3Q19_LATTICE

    # MRT matrix — transformation matrix, M matrix
    if "mrt matrix" in t or "transformation matrix" in t or "m matrix" in t:
        return INTENT_THEORY_MRT_MATRIX
    if "relaxation rate" in t and ("mrt" in t or "d3q19" in t):
        return INTENT_THEORY_MRT_MATRIX

    # LBM formulation — MRT, BGK, SRT, equilibrium, forcing
    if "lbm formulation" in t or "lbm equation" in t:
        return INTENT_THEORY_LBM_FORMULATION
    if "mrt formulation" in t or "mrt equation" in t:
        return INTENT_THEORY_LBM_FORMULATION
    if re.search(r"\bbgk\b", t) or re.search(r"\bsrt\b", t) and "real" not in t:
        return INTENT_THEORY_LBM_FORMULATION
    if "equilibrium distribution" in t or "guo forcing" in t or "guo's forcing" in t:
        return INTENT_THEORY_LBM_FORMULATION

    # Generic theory equations page — "theory equations", "theory page"
    if "theory equation" in t or "theory & equation" in t:
        return INTENT_THEORY_EQUATIONS_FULL
    if "theory page" in t or "equations page" in t:
        return INTENT_THEORY_EQUATIONS_FULL
    # "theory" or "equations" alone, when NOT about real isotropy (no real/isotropy/lumley/b_ij)
    if ("theory" in t or "equations" in t) and not any(
        x in t for x in ["real", "isotropy", "lumley", "b_ij", "b_ii", "energy fraction", "reynolds stress"]
    ):
        # Prefer theory equations full when it's clearly the theory page
        if "d3q19" in t or "mrt" in t or "lbm" in t or "navier" in t:
            return INTENT_THEORY_EQUATIONS_FULL
        # "theory" or "equations" alone — could be theory page
        if t.strip() in ("theory", "equations", "show theory", "show equations"):
            return INTENT_THEORY_EQUATIONS_FULL

    return None


# =============================================================================
# PAGE 04 — REAL ISOTROPY (eps_real_validation*.csv)
# Subplot order: A=energy_fractions, B=lumley, C=diagonal_bii, D=cross, E=deviations, F=convergence
# Check more specific (subplot B/C/D/E/F) before generic (energy fractions, summary)
# =============================================================================

_P04_LUMLEY_PATTERNS = [
    r"\blumley\b", r"\blumely\b",
    r"\blumely\s*triange\b",
    r"\bsubplot\s*b\b",
    r"\bxi\s*eta\b", r"\bξ\s*η\b",
    r"\brealizability\b", r"\blumley\s*triangle\b",
    r"\blumley\s*invariants\b",
]

_P04_DIAGONAL_PATTERNS = [
    r"\bsubplot\s*c\b",
    r"\bsubplot\s*3\b",
    r"\bthird\s*subplot\b",
    r"\bthird\s*plot\b",
    r"\bdiagonal\s*b_?ii\b",
    r"\bdiagonal\s*anisotropy\b",
    r"\bb11\b.*\bb22\b.*\bb33\b",
    r"\bb_?11\b.*\bb_?22\b.*\bb_?33\b",
]

_P04_CROSS_PATTERNS = [
    r"\bsubplot\s*d\b",
    r"\bsubplot\s*4\b",
    r"\bfourth\s*subplot\b",
    r"\bcross[- ]?correlation",
    r"\bb12\b.*\bb13\b.*\bb23\b",
    r"\banisotropy\s*index\b",
]

_P04_DEVIATIONS_PATTERNS = [
    r"\bsubplot\s*e\b",
    r"\bsubplot\s*5\b",
    r"\bfifth\s*subplot\b",
    r"\bdeviations\b",
    r"\benergy\s*fraction\s*deviation",
    r"\bdevx\b|\bdevy\b|\bdevz\b",
]

_P04_CONVERGENCE_PATTERNS = [
    r"\bsubplot\s*f\b",
    r"\bsubplot\s*6\b",
    r"\bsixth\s*subplot\b",
    r"\bconvergence\b",
    r"\brunning\s*std\b",
    r"\brunning\s*standard\s*deviation\b",
]


def _check_p04_real_isotropy(t: str, has_spectrum: bool) -> Optional[str]:
    """Check Page 04 (Real Isotropy) intents. Returns intent or None."""
    # Theory ONLY when user is NOT asking for a plot — if "plot" or "show" (figure) in request, prefer plot intent
    theory_only = ("theory" in t or "equations" in t or "formulas" in t) and "plot" not in t and "chart" not in t
    # Lumley (subplot B) — most specific
    for p in _P04_LUMLEY_PATTERNS:
        if re.search(p, t) and not has_spectrum:
            return INTENT_LUMLEY_TRIANGLE

    # Diagonal b_ii (subplot C)
    for p in _P04_DIAGONAL_PATTERNS:
        if re.search(p, t) and not has_spectrum:
            return INTENT_DIAGONAL_BII

    # Cross-correlations (subplot D)
    for p in _P04_CROSS_PATTERNS:
        if re.search(p, t) and not has_spectrum:
            return INTENT_CROSS_CORRELATIONS

    # Deviations (subplot E)
    for p in _P04_DEVIATIONS_PATTERNS:
        if re.search(p, t) and not has_spectrum:
            return INTENT_DEVIATIONS

    # Convergence (subplot F)
    for p in _P04_CONVERGENCE_PATTERNS:
        if re.search(p, t) and not has_spectrum:
            return INTENT_CONVERGENCE

    # Energy fractions (subplot A) — generic real isotropy
    if "energy fraction" in t or "frac_x" in t or "frac_y" in t or "frac_z" in t:
        return INTENT_ENERGY_FRACTIONS
    # Explicit real-space ask must win even when the same sentence also mentions spectra.
    if "real isotropy" in t or "real-space isotropy" in t or "real space isotropy" in t:
        return INTENT_ENERGY_FRACTIONS

    # Real isotropy summary (table) — check before spectral summary
    if ("summary" in t or "table" in t or "statistics" in t) and ("real" in t or "final isotropy" in t):
        return INTENT_REAL_ISOTROPY_SUMMARY
    if ("summary" in t or "table" in t) and "isotropy" in t and "spectral" not in t and "ic(k)" not in t:
        return INTENT_REAL_ISOTROPY_SUMMARY

    # General isotropy (ambiguous) — default to energy fractions.
    # "spectral isotropy" belongs to Page 05, not real-space isotropy.
    if "isotropy" in t and not has_spectrum and "spectral" not in t and "ic(k)" not in t:
        return INTENT_ENERGY_FRACTIONS

    # Theory & Equations — only when user asks for theory WITHOUT a plot (plot+theory handled by plot intents above)
    # Exclude "spectral" so "theory for spectral isotropy" routes to spectral, not real
    if theory_only and "spectral" not in t and ("real" in t or "isotropy" in t or "lumley" in t or "b_ij" in t or "b_ii" in t or "energy fraction" in t or "reynolds stress" in t or "anisotropy tensor" in t):
        return INTENT_REAL_ISOTROPY_THEORY

    return None


# =============================================================================
# PAGE 05 — SPECTRAL ISOTROPY (isotropy_coeff_*.dat)
# =============================================================================

def _check_p05_spectral_isotropy(t: str) -> Optional[str]:
    """Check Page 05 (Spectral Isotropy) intents. Returns intent or None."""
    # Spectral isotropy theory/equations — must check before summary/plot
    if ("theory" in t or "equations" in t) and "spectral" in t:
        return INTENT_SPECTRAL_ISOTROPY_THEORY
    if "spectral isotropy" in t and ("theory" in t or "equations" in t):
        return INTENT_SPECTRAL_ISOTROPY_THEORY
    if "ic(k) theory" in t:
        return INTENT_SPECTRAL_ISOTROPY_THEORY

    # Spectral isotropy summary (table) — check before generic spectral
    if ("summary" in t or "table" in t or "statistics" in t) and ("spectral" in t or "ic(k)" in t or "isotropy" in t):
        return INTENT_SPECTRAL_ISOTROPY_SUMMARY

    # Component spectra (E11, E22, E33)
    if "component spectra" in t or "e_ii" in t or "eii" in t:
        return INTENT_COMPONENT_SPECTRA
    if "e11" in t and "e22" in t and "e33" in t:
        return INTENT_COMPONENT_SPECTRA

    # Spectral isotropy (IC(k))
    if "spectral isotropy" in t or "ic(k)" in t or "isotropy coefficient" in t:
        return INTENT_SPECTRAL_ISOTROPY

    return None


# =============================================================================
# PAGE 06 — ENERGY SPECTRA (spectrum*.dat)
# =============================================================================

def _check_p06_energy_spectra(t: str, has_spectrum: bool) -> Optional[str]:
    """Check Page 06 (Energy Spectra) intents. Returns intent or None."""
    # Theory/equations — check before plot; exclude "spectral" so "spectral isotropy theory" stays in P05
    theory_only = ("theory" in t or "equations" in t or "formulas" in t) and "plot" not in t and "chart" not in t
    if theory_only and "spectral" not in t:
        if "spectra theory" in t or "energy spectra theory" in t or "theory for spectra" in t:
            return INTENT_ENERGY_SPECTRA_THEORY
        if ("e(k)" in t or "kolmogorov" in t or "spectra" in t or "spectrum" in t) and ("theory" in t or "equations" in t):
            return INTENT_ENERGY_SPECTRA_THEORY

    if has_spectrum or "kolmogorov" in t or "e(k)" in t or "-5/3" in t or "5/3" in t:
        return INTENT_ENERGY_SPECTRA
    return None


# =============================================================================
# PAGE 07 — FLATNESS (flatness_data*_*.txt)
# =============================================================================

def _check_p07_flatness(t: str) -> Optional[str]:
    """Check Page 07 (Flatness) intents. Returns intent or None."""
    if "flatness" not in t:
        return None
    # Theory/equations — must check before generic flatness
    if ("theory" in t or "equations" in t) and "flatness" in t:
        return INTENT_FLATNESS_THEORY
    if "flatness theory" in t or "flatness equations" in t or "theory for flatness" in t:
        return INTENT_FLATNESS_THEORY
    if "f(r) theory" in t or "kurtosis theory" in t:
        return INTENT_FLATNESS_THEORY
    return INTENT_FLATNESS


# =============================================================================
# PAGE 08 — STRUCTURE FUNCTIONS (structure_functions_*.txt)
# =============================================================================

def _check_p08_structure_functions(t: str) -> Optional[str]:
    """Check Page 08 (Structure Functions) intents. Returns intent or None."""
    if "structure function" not in t:
        return None
    if ("theory" in t or "equations" in t) and "structure" in t:
        return INTENT_STRUCTURE_FUNCTIONS_THEORY
    if "structure functions theory" in t or "theory for structure functions" in t:
        return INTENT_STRUCTURE_FUNCTIONS_THEORY
    if "she-leveque" in t or "she leveque" in t or "scaling exponent" in t:
        if "theory" in t or "equations" in t:
            return INTENT_STRUCTURE_FUNCTIONS_THEORY
    return INTENT_STRUCTURE_FUNCTIONS


# =============================================================================
# PAGE 09 — PDFs (*.vti, *.h5, *.hdf5 — velocity-based PDFs)
# =============================================================================

def _check_p09_pdf(t: str) -> Optional[str]:
    """Check Page 09 (PDFs) intents. Returns intent or None.
    PDFs page = probability density functions (turbulence). Exclude document/file PDF operations.
    """
    turb = ("vorticity", "enstrophy", "dissipation", "velocity", "joint pdf", "r-q", "probability density")
    has_turb = any(k in t for k in turb)
    # Exclude: document/file PDF operations (not probability density functions)
    if "pdf report" in t or "export report" in t:
        return None  # report generator
    if ("compile" in t or "latex" in t) and "pdf" in t:
        return None  # compile LaTeX to PDF
    if ("export figure" in t or "save figure" in t or "save as pdf" in t or "export to pdf" in t) and not has_turb:
        return None  # export_figure(format=pdf), not PDFs page
    if ("run " in t or "shell " in t or "command " in t) and "pdf" in t and not has_turb:
        return None  # shell command producing PDF (e.g. pdflatex), not PDFs page
    # PDFs page: velocity-based probability density functions
    pdf_triggers = (
        "pdfs page" in t or "pdf page" in t
        or ("pdf" in t and (
            "vorticity" in t or "enstrophy" in t or "dissipation" in t
            or "velocity" in t or "velocity magnitude" in t
            or "joint pdf" in t or "r-q" in t or "q-r" in t
            or "vti" in t or "hdf5" in t or "h5" in t
            or "turbulence" in t or "probability density" in t
        ))
    )
    if pdf_triggers:
        return INTENT_PDF
    return None


# =============================================================================
# PAGE 10 — OTHER TURBULENCE STATS (turbulence_stats*.csv, eps_real_validation*.csv)
# =============================================================================

def _check_p10_other_turbulence_stats(t: str) -> Optional[str]:
    """Check Page 10 (Other Turbulence Stats) intents. Returns intent or None."""
    from agents.remote_document import is_remote_document_request

    # Exclude: spectral isotropy, real isotropy, PDFs (handled by P04, P05, P09)
    if "spectral isotropy" in t or "ic(k)" in t:
        return None
    if "real isotropy" in t or "lumley" in t or "energy fraction" in t:
        return None
    if "pdf" in t and "turbulence" in t:
        return None  # P09
    if is_remote_document_request(t):
        return None
    # Summary/table — check before generic plot
    if ("summary" in t or "table" in t or "latest values" in t) and (
        "turbulence stats" in t or "other stats" in t or "turbulence stats page" in t
    ):
        return INTENT_OTHER_TURBULENCE_STATS_SUMMARY
    if "turbulence stats" in t and ("summary" in t or "table" in t):
        return INTENT_OTHER_TURBULENCE_STATS_SUMMARY
    # Plot or generic other turbulence stats
    if "other stats" in t or "other turbulence stats" in t or "turbulence stats page" in t:
        return INTENT_OTHER_TURBULENCE_STATS
    if "turbulence stats" in t and ("plot" in t or "show" in t or "custom plot" in t):
        return INTENT_OTHER_TURBULENCE_STATS
    if "energy balance" in t or "eps validation" in t:
        return INTENT_OTHER_TURBULENCE_STATS
    # Columns/quantities from turbulence_stats or eps_validation CSVs
    if any(
        x in t
        for x in [
            "eps_spectral",
            "eps_real",
            "dissipation rate",
            "dissipation",
            "tke_real",
            "energy_balance",
            "tke",
        ]
    ) and ("plot" in t or "compute" in t or "show" in t or "visual" in t or "from" in t or "dns" in t or "les" in t or "openlb" in t):
        # Dissipation PDF is Page 09, not a turbulence-stats time series.
        if "dissipation" in t and "pdf" in t:
            return None
        return INTENT_OTHER_TURBULENCE_STATS
    if "time series" in t and "turbulence" in t:
        return INTENT_OTHER_TURBULENCE_STATS
    return None


# =============================================================================
# PAGE 12 — REPORT GENERATOR (add to report, generate report)
# =============================================================================

def _check_p12_report_generator(t: str) -> Optional[str]:
    """Check Page 12 (Report Generator) intents. Returns intent or None."""
    # Preview compiled report — full HTML with figures, tables, text (check before structure)
    if any(x in t for x in ["show me the report", "show report in chat", "preview report",
                            "let me see the report", "display report", "show the compiled report",
                            "compiled report", "complete compiled report", "full report",
                            "show report", "see the report", "current report",
                            "how it looks", "how the report looks", "report with figures",
                            "report with tables", "report with all"]):
        return INTENT_REPORT_PREVIEW
    # "What's in my report", "report structure", "list sections" -> preview (full report has TOC)
    if any(x in t for x in ["what's in my report", "whats in my report", "show report structure",
                            "list report sections", "report outline", "report structure",
                            "what sections", "section list"]):
        return INTENT_REPORT_PREVIEW
    if "report" in t and ("what" in t or "outline" in t or "structure" in t or "list" in t):
        return INTENT_REPORT_PREVIEW
    # Generate/export report — check before add (more specific)
    if any(x in t for x in ["generate report", "export report", "create pdf report", "create html report",
                            "save report", "build report", "write report", "produce report"]):
        return INTENT_REPORT_GENERATE
    if "report" in t and ("generate" in t or "export" in t or "create" in t or "pdf" in t or "html" in t):
        return INTENT_REPORT_GENERATE
    # Remove/delete section
    if any(x in t for x in ["delete section", "remove section"]):
        return INTENT_REPORT_REMOVE
    if ("delete the" in t or "remove the" in t) and "section" in t:
        return INTENT_REPORT_REMOVE
    if "section" in t and ("delete" in t or "remove" in t):
        return INTENT_REPORT_REMOVE
    # Move/reorder section (require explicit report-section context; not plot elements)
    if any(x in t for x in ["move section", "reorder section", "swap section"]):
        return INTENT_REPORT_REORDER
    if ("move the" in t or "reorder the" in t) and "section" in t:
        return INTENT_REPORT_REORDER
    if "section" in t and ("move" in t or "reorder" in t or "swap" in t):
        return INTENT_REPORT_REORDER
    # Edit section
    if any(x in t for x in ["edit section", "change section", "update section", "modify section"]):
        return INTENT_REPORT_EDIT
    if "section" in t and ("edit" in t or "change" in t or "update" in t or "modify" in t):
        return INTENT_REPORT_EDIT
    # Add/capture to report
    if any(x in t for x in ["add to report", "capture to report", "add this to report", "include in report",
                            "add figure to report", "add plot to report", "capture for report", "add table"]):
        return INTENT_REPORT_ADD_SECTION
    if "report" in t and ("add" in t or "capture" in t or "include" in t):
        return INTENT_REPORT_ADD_SECTION
    if "report builder" in t or "report page" in t:
        # Generic report page — prefer add if context suggests (e.g. "add to report builder")
        if "add" in t or "capture" in t:
            return INTENT_REPORT_ADD_SECTION
        return INTENT_REPORT_GENERATE  # Default to generate for "report page"
    return None


# =============================================================================
# PAGE 11 — 3D VOLUME VIEWER (*.vti, *.h5, *.hdf5)
# =============================================================================

def _check_p11_volume_viewer_3d(t: str) -> Optional[str]:
    """Check Page 11 (3D Volume Viewer) intents. Returns intent or None."""
    # Theory first
    if ("theory" in t or "equations" in t) and (
        "volume viewer" in t or "3d volume" in t or "vorticity" in t and "equation" in t
        or "q invariant" in t or "q_s^s" in t or "r invariant" in t
    ):
        return INTENT_VOLUME_VIEWER_3D_THEORY
    if "volume viewer theory" in t or "3d volume equations" in t:
        return INTENT_VOLUME_VIEWER_3D_THEORY
    # Plot
    if "3d volume" in t or "volume viewer" in t or "volume visualization" in t:
        return INTENT_VOLUME_VIEWER_3D
    if "velocity field 3d" in t or "vti visualization" in t:
        return INTENT_VOLUME_VIEWER_3D
    if "vti" in t and ("plot" in t or "3d" in t or "volume" in t or "visualize" in t or "show" in t):
        return INTENT_VOLUME_VIEWER_3D
    if "vorticity 3d" in t or "isosurface" in t and "velocity" in t:
        return INTENT_VOLUME_VIEWER_3D
    if "q invariant" in t or "r invariant" in t or "q_s^s" in t:
        if "plot" in t or "visualize" in t or "show" in t:
            return INTENT_VOLUME_VIEWER_3D
    return None


# =============================================================================
# MAIN: get_analysis_intent — order: P04 -> P05 -> P06 -> P07+ -> P10 -> P11 -> fallback
# =============================================================================

def get_analysis_intent(user_input: str) -> Optional[str]:
    """
    Detect the primary analysis/plot intent from user input.

    Returns:
        Intent constant (e.g. INTENT_LUMLEY_TRIANGLE) or None if no clear intent.
        Order matters: more specific intents checked first; page 04 before 05/06 to
        disambiguate isotropy (real vs spectral).
    """
    if not user_input or not user_input.strip():
        return None

    t = user_input.strip().lower()
    from agents.remote_document import is_remote_document_request
    if is_remote_document_request(user_input):
        return None

    has_spectrum = _has_spectrum_keywords(t)

    intent = _check_app_settings_hdf5(t)
    if intent is not None:
        return intent

    intent = _check_p01_overview(t)
    if intent is not None:
        return intent

    intent = _check_p02_theory_equations(t)
    if intent is not None:
        return intent

    # Prefer spectral isotropy over bare "isotropy" when both signals exist.
    intent = _check_p05_spectral_isotropy(t)
    if intent is not None:
        return intent

    intent = _check_p04_real_isotropy(t, has_spectrum)
    if intent is not None:
        return intent

    intent = _check_p06_energy_spectra(t, has_spectrum)
    if intent is not None:
        return intent

    intent = _check_p07_flatness(t)
    if intent is not None:
        return intent

    intent = _check_p08_structure_functions(t)
    if intent is not None:
        return intent

    intent = _check_p09_pdf(t)
    if intent is not None:
        return intent

    intent = _check_p10_other_turbulence_stats(t)
    if intent is not None:
        return intent

    intent = _check_p11_volume_viewer_3d(t)
    if intent is not None:
        return intent

    intent = _check_p12_report_generator(t)
    if intent is not None:
        return intent

    return None


def collect_analysis_intents(user_input: str) -> List[str]:
    """
    Collect all post-run analysis intents present in one request (multi-ask safe).

    Used by OpenLB run+analyse pipelines to chain multiple page tools. Theory-only
    intents are omitted.
    """
    if not user_input or not user_input.strip():
        return []

    t = user_input.strip().lower()
    has_spectrum = _has_spectrum_keywords(t)
    found: List[str] = []

    def _add(intent: Optional[str]) -> None:
        if not intent or intent in found:
            return
        if intent.endswith("_theory"):
            return
        found.append(intent)

    _add(_check_p05_spectral_isotropy(t))
    _add(_check_p04_real_isotropy(t, has_spectrum))

    energy = _check_p06_energy_spectra(t, has_spectrum)
    if energy == INTENT_ENERGY_SPECTRA:
        if any(i in found for i in (INTENT_COMPONENT_SPECTRA, INTENT_SPECTRAL_ISOTROPY)):
            if not (
                "energy spectra" in t
                or "energy spectrum" in t
                or "e(k)" in t
                or "kolmogorov" in t
            ):
                energy = None
    _add(energy)

    _add(_check_p07_flatness(t))
    _add(_check_p08_structure_functions(t))
    _add(_check_p09_pdf(t))
    _add(_check_p10_other_turbulence_stats(t))
    _add(_check_p11_volume_viewer_3d(t))
    return found


def get_plot_routing(user_input: str) -> Dict[str, Any]:
    """
    Route plot request to tool and return structured hints.
    Tool, file_pattern, skip_analyst, intent_override, prevent_tools derived from page_schema.
    """
    result: Dict[str, Any] = {
        "intent": None,
        "tool": None,
        "file_pattern": None,
        "skip_analyst": False,
        "intent_override_text": None,
        "prevent_tools": [],
    }

    intent = get_analysis_intent(user_input)
    if not intent:
        return result

    result["intent"] = intent
    routing = get_routing_for_intent(intent, user_input)
    if routing:
        result["tool"] = routing.get("tool")
        result["file_pattern"] = routing.get("file_pattern")
        result["skip_analyst"] = routing.get("skip_analyst", False)
        result["intent_override_text"] = routing.get("intent_override_text")
        result["prevent_tools"] = routing.get("prevent_tools", [])

    return result
