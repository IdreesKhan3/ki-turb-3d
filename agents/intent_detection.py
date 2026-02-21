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
from typing import Any, Dict, Optional

from .page_schema import (
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
    INTENT_LUMLEY_TRIANGLE,
    INTENT_PDF,
    INTENT_SPECTRAL_ISOTROPY,
    INTENT_SPECTRAL_ISOTROPY_SUMMARY,
    INTENT_SPECTRAL_ISOTROPY_THEORY,
    INTENT_STRUCTURE_FUNCTIONS,
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
    "INTENT_STRUCTURE_FUNCTIONS",
    "INTENT_PDF",
    "INTENT_UNKNOWN",
    "get_analysis_intent",
    "get_plot_routing",
]

def _has_spectrum_keywords(t: str) -> bool:
    """Exclude spectrum/spectra when user clearly wants isotropy (not energy spectra)."""
    return (
        "spectrum" in t or "spectra" in t or "e(k)" in t
        or "evolution" in t  # "time evolution", "evolution spectra" -> energy spectra
    )



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

    # Real isotropy summary (table) — check before spectral summary
    if ("summary" in t or "table" in t or "statistics" in t) and ("real" in t or "final isotropy" in t):
        return INTENT_REAL_ISOTROPY_SUMMARY
    if ("summary" in t or "table" in t) and "isotropy" in t and "spectral" not in t and "ic(k)" not in t:
        return INTENT_REAL_ISOTROPY_SUMMARY

    # General isotropy (ambiguous) — default to energy fractions
    if "isotropy" in t and not has_spectrum:
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

    if has_spectrum or "kolmogorov" in t or "e(k)" in t:
        return INTENT_ENERGY_SPECTRA
    return None


# =============================================================================
# PAGE 07 — FLATNESS (flatness_data*_*.txt)
# =============================================================================

def _check_p07_flatness(t: str) -> Optional[str]:
    """Check Page 07 (Flatness) intents. Returns intent or None."""
    if "flatness" in t:
        return INTENT_FLATNESS
    return None


# =============================================================================
# PAGE 08 — STRUCTURE FUNCTIONS (structure_functions_*.txt)
# =============================================================================

def _check_p08_structure_functions(t: str) -> Optional[str]:
    """Check Page 08 (Structure Functions) intents. Returns intent or None."""
    if "structure function" in t:
        return INTENT_STRUCTURE_FUNCTIONS
    return None


# =============================================================================
# PAGE 09 — PDFs (turbulence_stats*.csv)
# =============================================================================

def _check_p09_pdf(t: str) -> Optional[str]:
    """Check Page 09 (PDFs) intents. Returns intent or None."""
    if "pdf" in t and ("turbulence" in t or "velocity" in t or "dissipation" in t):
        return INTENT_PDF
    return None


# =============================================================================
# MAIN: get_analysis_intent — order: P04 -> P05 -> P06 -> P07+ -> P06 fallback
# =============================================================================

def get_analysis_intent(user_input: str) -> Optional[str]:
    """
    Detect analysis/plot intent from user input.

    Returns:
        Intent constant (e.g. INTENT_LUMLEY_TRIANGLE) or None if no clear intent.
        Order matters: more specific intents checked first; page 04 before 05/06 to
        disambiguate isotropy (real vs spectral).
    """
    if not user_input or not user_input.strip():
        return None

    t = user_input.strip().lower()
    has_spectrum = _has_spectrum_keywords(t)

    # Page 01 — Overview
    intent = _check_p01_overview(t)
    if intent is not None:
        return intent

    # Page 02 — Theory & Equations (NS, LBM, D3Q19, MRT matrix)
    intent = _check_p02_theory_equations(t)
    if intent is not None:
        return intent

    # Page 04 — Real Isotropy (includes generic "isotropy" -> energy fractions)
    intent = _check_p04_real_isotropy(t, has_spectrum)
    if intent is not None:
        return intent

    # Page 05 — Spectral Isotropy
    intent = _check_p05_spectral_isotropy(t)
    if intent is not None:
        return intent

    # Page 06 — Energy Spectra
    intent = _check_p06_energy_spectra(t, has_spectrum)
    if intent is not None:
        return intent

    # Page 07 — Flatness
    intent = _check_p07_flatness(t)
    if intent is not None:
        return intent

    # Page 08 — Structure Functions
    intent = _check_p08_structure_functions(t)
    if intent is not None:
        return intent

    # Page 09 — PDFs
    intent = _check_p09_pdf(t)
    if intent is not None:
        return intent

    return None


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
