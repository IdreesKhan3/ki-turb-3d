"""
Page Schema — Single source of truth for all analysis pages.

Defines the workflow for each page: file patterns, compute step (optional), plot/summary tools.
Orchestrator, intent detection, and prompts derive from this schema.

ORGANIZATION: Content is sectionized by page. When adding a new page:
  1. Add schema entry in PAGE_SCHEMA (in the page's section)
  2. Add intent constants (if page has agent tools)
  3. Add routing entries in INTENT_ROUTING
  4. Implement tools in agents/tools/physics/
  5. Add patterns in intent_detection.py
  6. Add prompts in team_manager/prompts.py
"""

from typing import Any, Dict, List, Optional

# Schema fields: file_patterns, compute_tool, plot_tools, summary_tool, skip_analyst, keywords, page_type
# - file_patterns: glob patterns steward uses to find data
# - compute_tool: analyst tool to run first (None = skip analyst)
# - plot_tools: visualizer tools that produce figures
# - summary_tool: visualizer tool that produces table (None = no table)
# - skip_analyst: True = steward -> visualizer directly
# - keywords: user phrases that map to this page
# - page_type: "analysis" | "informational" | "chat"


# =============================================================================
# PAGE_SCHEMA — one entry per page
# =============================================================================

PAGE_SCHEMA: Dict[str, Dict[str, Any]] = {
    # --- PAGES 00-03: Chat, Overview, Theory, Multi-Method (no agent tools) ---
    "autonomous_lab": {
        "file_patterns": [],
        "compute_tool": None,
        "plot_tools": [],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": ["autonomous lab", "chat", "ai assistant"],
        "page_type": "chat",
    },
    "app_settings": {
        "file_patterns": [],
        "compute_tool": None,
        "plot_tools": [],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": [
            "hdf5 fortran", "hdf5 default", "hdf5 format", "hdf5 layout",
            "fortran hdf5", "default hdf5", "data format", "set hdf5",
            "load hdf5 fortran", "use hdf5 fortran", "hdf5 fortran option",
            "hdf5 no transpose", "python hdf5", "standard hdf5",
        ],
        "page_type": "chat",
    },
    "overview": {
        "file_patterns": ["simulation.input", "simulation.json", "turbulence_stats*.csv", "tau_analysis_*.bin", "eps_real_validation*.csv"],
        "compute_tool": None,
        "plot_tools": ["get_overview_theory"],
        "summary_tool": "get_overview_summary",
        "skip_analyst": True,
        "keywords": ["overview", "metadata", "parameters", "time series", "re number", "physics validation", "mach number", "knudsen number", "data availability", "what files", "overview theory", "overview equations", "theory for overview", "equations for overview", "physics validation equations"],
        "page_type": "analysis",
    },
    "theory_equations": {
        "file_patterns": [],
        "compute_tool": None,
        "plot_tools": ["get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice", "get_theory_mrt_matrix"],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": ["theory", "equations", "d3q19", "mrt", "navier-stokes", "lbm formulation", "lattice visualization", "mrt matrix", "ns equations", "lbm equations", "lattice stencil", "transformation matrix"],
        "page_type": "informational",
    },
    "multi_method_support": {
        "file_patterns": [],
        "compute_tool": None,
        "plot_tools": [],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": ["multi method", "mrt", "srt", "bgk", "trt"],
        "page_type": "informational",
    },

    # =========================================================================
    # PAGE 04 — REAL ISOTROPY (eps_real_validation*.csv or turbulence_validation*.csv, LBM/NS)
    # Subplots: A=energy_fractions, B=lumley, C=diagonal_bii, D=cross, E=deviations, F=convergence
    # =========================================================================
    "real_isotropy": {
        "file_patterns": ["eps_real_validation*.csv", "turbulence_validation*.csv"],
        "compute_tool": None,
        "plot_tools": ["plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations", "plot_convergence", "get_real_isotropy_theory"],
        "summary_tool": "get_real_isotropy_summary",
        "skip_analyst": True,
        "keywords": ["real isotropy", "energy fractions", "lumley", "lumely", "subplot b", "subplot c", "subplot d", "subplot e", "subplot f", "cross-correlations", "deviations", "convergence", "running std", "xi eta", "diagonal b_ii", "b11 b22 b33", "b12 b13 b23", "anisotropy index", "energy fraction deviations", "real isotropy summary", "real isotropy table", "real isotropy page", "real isotropy theory", "real isotropy equations"],
        "page_type": "analysis",
    },

    # =========================================================================
    # PAGE 05 — SPECTRAL ISOTROPY (isotropy_coeff_*.dat)
    # =========================================================================
    "spectral_isotropy": {
        "file_patterns": ["isotropy_coeff_*.dat"],
        "compute_tool": "compute_spectral_isotropy",
        "plot_tools": ["plot_spectral_isotropy", "plot_component_spectra", "get_spectral_isotropy_theory"],
        "summary_tool": "get_spectral_isotropy_summary",
        "skip_analyst": False,
        "keywords": ["spectral isotropy", "ic(k)", "e11", "e22", "e33", "component spectra", "spectral isotropy page", "spectral isotropy theory", "spectral isotropy equations", "theory for spectral isotropy", "equations for spectral isotropy", "ic(k) theory"],
        "page_type": "analysis",
    },

    # =========================================================================
    # PAGE 06 — ENERGY SPECTRA (spectrum*.dat, norm*.dat)
    # =========================================================================
    "energy_spectra": {
        "file_patterns": ["spectrum*.dat", "norm*.dat"],
        "compute_tool": "compute_spectra",
        "plot_tools": ["plot_spectrum", "get_energy_spectra_theory"],
        "summary_tool": None,
        "skip_analyst": False,
        "keywords": ["spectra", "spectrum", "e(k)", "evolution", "kolmogorov", "spectra page", "from spectra", "time evolution", "spectra theory", "energy spectra theory", "theory for spectra", "equations for spectra", "e(k) theory", "kolmogorov theory"],
        "data_refs": {"raw": "current_spectra_data", "normalized": "current_spectra_norm", "evolution": "current_spectra_evolution"},
        "page_type": "analysis",
    },

    # =========================================================================
    # PAGE 07 — FLATNESS (flatness_data*_*.txt)
    # =========================================================================
    "flatness": {
        "file_patterns": ["flatness_data*_*.txt"],
        "compute_tool": "compute_flatness",
        "plot_tools": ["plot_flatness", "get_flatness_theory"],
        "summary_tool": "get_flatness_summary",
        "skip_analyst": False,
        "keywords": ["flatness", "kurtosis", "intermittency", "flatness page", "flatness data", "flatness theory", "flatness equations", "theory for flatness", "equations for flatness", "F(r) theory", "kurtosis theory"],
        "page_type": "analysis",
    },

    # =========================================================================
    # PAGE 08 — STRUCTURE FUNCTIONS (structure_functions_*.txt, structure_funcs*_t*.bin)
    # =========================================================================
    "structure_functions": {
        "file_patterns": ["structure_functions_*.txt", "structure_funcs*_t*.bin"],
        "compute_tool": "compute_structure_functions",
        "plot_tools": ["plot_structure_functions", "get_structure_functions_theory"],
        "summary_tool": None,
        "skip_analyst": False,
        "keywords": ["structure function", "S_p", "ESS", "structure functions page", "structure functions theory", "theory for structure functions", "equations for structure functions", "She-Leveque", "scaling exponent"],
        "data_refs": {"structure": "current_structure_functions_data"},
        "page_type": "analysis",
    },

    # =========================================================================
    # PAGE 09 — PDFs (*.vti, *.h5, *.hdf5) — Phase 1: schema wired, tools TBD
    # Velocity-based PDFs: vorticity, enstrophy, dissipation, velocity magnitude, joint PDFs (R-Q, velocity-dissipation, etc.)
    # =========================================================================
    "pdfs": {
        "file_patterns": ["*.vti", "*.h5", "*.hdf5"],
        "compute_tool": None,
        "plot_tools": ["plot_pdf"],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": [
            "pdf", "pdfs", "pdfs page", "probability density",
            "velocity pdf", "velocity components pdf", "u v w pdf", "vorticity pdf", "enstrophy pdf", "dissipation pdf", "dissipation rate pdf",
            "velocity magnitude pdf", "velocity magnitude",
            "joint pdf", "velocity-dissipation joint", "velocity-enstrophy joint",
            "dissipation-enstrophy joint", "r-q joint", "r-q topological", "q-r joint",
        ],
        "page_type": "analysis",
    },

    # =========================================================================
    # PAGE 10 — OTHER TURBULENCE STATS (turbulence_stats*.csv, eps_real_validation*.csv)
    # Custom x-y plots and data tables from turbulence_stats and eps_validation CSVs
    # =========================================================================
    "other_turbulence_stats": {
        "file_patterns": ["turbulence_stats*.csv", "eps_real_validation*.csv", "turbulence_validation*.csv"],
        "compute_tool": None,
        "plot_tools": ["plot_turbulence_stats"],
        "summary_tool": "get_turbulence_stats_summary",
        "skip_analyst": True,
        "keywords": [
            "other stats",
            "other turbulence stats",
            "turbulence stats",
            "turbulence stats page",
            "time series",
            "custom plot",
            "plot turbulence stats",
            "turbulence stats table",
            "turbulence stats summary",
            "energy balance",
            "eps validation",
        ],
        "page_type": "analysis",
    },
    # =========================================================================
    # PAGE 11 — 3D VOLUME VIEWER (*.vti, *.h5, *.hdf5)
    # Velocity field visualization: volume rendering, slices, isosurface, Q/R invariants
    # =========================================================================
    "volume_viewer_3d": {
        "file_patterns": ["*.vti", "*.h5", "*.hdf5"],
        "compute_tool": None,
        "plot_tools": ["plot_volume_3d"],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": [
            "3d volume",
            "3d volume viewer",
            "volume viewer",
            "volume visualization",
            "velocity field 3d",
            "vti",
            "hdf5",
            "vorticity",
            "isosurface",
            "q invariant",
            "r invariant",
            "q_s^s",
        ],
        "page_type": "analysis",
    },
    # =========================================================================
    # PAGE 12 — REPORT GENERATOR (report_config.json, lab_artifact_history)
    # Phase 2: add_report_section, generate_report wired.
    # =========================================================================
    "report_generator": {
        "file_patterns": ["report_config.json"],
        "compute_tool": None,
        "plot_tools": ["preview_report", "add_report_section", "remove_report_section",
                      "reorder_report_section", "edit_report_section", "generate_report"],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": [
            "report", "export report", "pdf report", "report builder",
            "scientific report", "generate report", "report page",
            "add to report", "capture to report", "report sections",
            "what's in my report", "show report structure", "report outline",
            "show me the report", "preview report", "compiled report", "display report",
            "delete section", "remove section", "move section", "reorder section",
            "edit section", "change section", "update section",
        ],
        "page_type": "analysis",
    },
    "citation": {
        "file_patterns": [],
        "compute_tool": None,
        "plot_tools": [],
        "summary_tool": None,
        "skip_analyst": True,
        "keywords": ["citation", "cite", "references", "bibtex"],
        "page_type": "informational",
    },
}

# --- PAGE 02 — Theory & Equations ---
INTENT_THEORY_NS_EQUATIONS = "theory_ns_equations"
INTENT_THEORY_LBM_FORMULATION = "theory_lbm_formulation"
INTENT_THEORY_D3Q19_LATTICE = "theory_d3q19_lattice"
INTENT_THEORY_MRT_MATRIX = "theory_mrt_matrix"
INTENT_THEORY_EQUATIONS_FULL = "theory_equations_full"

# =============================================================================
# INTENT CONSTANTS — used by intent_detection
# =============================================================================

# --- PAGE 01 — Overview ---
INTENT_OVERVIEW = "overview"
INTENT_OVERVIEW_THEORY = "overview_theory"


# --- PAGE 04 — Real Isotropy ---
INTENT_LUMLEY_TRIANGLE = "lumley_triangle"
INTENT_DIAGONAL_BII = "diagonal_bii"
INTENT_CROSS_CORRELATIONS = "cross_correlations"
INTENT_DEVIATIONS = "deviations"
INTENT_CONVERGENCE = "convergence"
INTENT_REAL_ISOTROPY_SUMMARY = "real_isotropy_summary"
INTENT_REAL_ISOTROPY_THEORY = "real_isotropy_theory"
INTENT_ENERGY_FRACTIONS = "energy_fractions"

# --- PAGE 05 — Spectral Isotropy ---
INTENT_SPECTRAL_ISOTROPY = "spectral_isotropy"
INTENT_SPECTRAL_ISOTROPY_SUMMARY = "spectral_isotropy_summary"
INTENT_SPECTRAL_ISOTROPY_THEORY = "spectral_isotropy_theory"
INTENT_COMPONENT_SPECTRA = "component_spectra"

# --- PAGE 06 — Energy Spectra ---
INTENT_ENERGY_SPECTRA = "energy_spectra"
INTENT_ENERGY_SPECTRA_THEORY = "energy_spectra_theory"

# --- PAGES 07-09 — Flatness, Structure Functions, PDFs ---
INTENT_FLATNESS = "flatness"
INTENT_FLATNESS_THEORY = "flatness_theory"
INTENT_STRUCTURE_FUNCTIONS = "structure_functions"
INTENT_STRUCTURE_FUNCTIONS_THEORY = "structure_functions_theory"
INTENT_PDF = "pdf"

# --- PAGE 10 — Other Turbulence Stats ---
INTENT_OTHER_TURBULENCE_STATS = "other_turbulence_stats"
INTENT_OTHER_TURBULENCE_STATS_SUMMARY = "other_turbulence_stats_summary"

# --- PAGE 11 — 3D Volume Viewer ---
INTENT_VOLUME_VIEWER_3D = "volume_viewer_3d"
INTENT_VOLUME_VIEWER_3D_THEORY = "volume_viewer_3d_theory"

# --- PAGE 12 — Report Generator ---
INTENT_REPORT_ADD_SECTION = "report_add_section"
INTENT_REPORT_REMOVE = "report_remove"
INTENT_REPORT_REORDER = "report_reorder"
INTENT_REPORT_EDIT = "report_edit"
INTENT_REPORT_GENERATE = "report_generate"
INTENT_REPORT_PREVIEW = "report_preview"

# --- APP SETTINGS — HDF5 format ---
INTENT_APP_SETTINGS_HDF5_FORTRAN = "app_settings_hdf5_fortran"
INTENT_APP_SETTINGS_HDF5_DEFAULT = "app_settings_hdf5_default"

# --- Fallback ---
INTENT_UNKNOWN = "unknown"


# =============================================================================
# INTENT_ROUTING — intent -> (primary_tool, prevent_tools, intent_override)
# file_pattern and skip_analyst come from PAGE_SCHEMA[page_id]
# =============================================================================

INTENT_ROUTING: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # PAGE 01 — OVERVIEW
    # -------------------------------------------------------------------------
    INTENT_OVERVIEW: {
        "page_id": "overview",
        "primary_tool": "get_overview_summary",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested OVERVIEW (Page 01): parameters, metadata, physics validation (Mach, Knudsen, compressibility), or data availability. "
            "Use get_overview_summary ONLY. Delegate: steward (find simulation.input or simulation.json or verify data dir) -> visualizer (get_overview_summary). "
            "Uses session data_directory/data_directories if no path given. Skip analyst. STOP after the overview table.\n\n"
        ),
    },
    INTENT_OVERVIEW_THEORY: {
        "page_id": "overview",
        "primary_tool": "get_overview_theory",
        "prevent_tools": ["get_overview_summary"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested OVERVIEW THEORY/EQUATIONS (Page 01): Physics Validation Equations (Mach, Knudsen, compressibility). "
            "Use get_overview_theory ONLY. Delegate: visualizer (get_overview_theory()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGE 02 — THEORY & EQUATIONS
    # -------------------------------------------------------------------------
    INTENT_THEORY_NS_EQUATIONS: {
        "page_id": "theory_equations",
        "primary_tool": "get_theory_ns_equations",
        "prevent_tools": ["get_theory_lbm_formulation", "plot_d3q19_lattice", "get_theory_mrt_matrix"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested NS EQUATIONS (Navier-Stokes, filtered NS, LES) from Theory & Equations page. "
            "Use get_theory_ns_equations ONLY. Delegate: visualizer (get_theory_ns_equations()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },
    INTENT_THEORY_LBM_FORMULATION: {
        "page_id": "theory_equations",
        "primary_tool": "get_theory_lbm_formulation",
        "prevent_tools": ["get_theory_ns_equations", "plot_d3q19_lattice", "get_theory_mrt_matrix"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested LBM FORMULATION (MRT, BGK/SRT, equilibrium, forcing, validation) from Theory & Equations page. "
            "Use get_theory_lbm_formulation ONLY. Delegate: visualizer (get_theory_lbm_formulation()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },
    INTENT_THEORY_D3Q19_LATTICE: {
        "page_id": "theory_equations",
        "primary_tool": "plot_d3q19_lattice",
        "prevent_tools": ["get_theory_ns_equations", "get_theory_lbm_formulation", "get_theory_mrt_matrix"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested D3Q19 LATTICE VISUALIZATION from Theory & Equations page. "
            "Use plot_d3q19_lattice ONLY. Delegate: visualizer (plot_d3q19_lattice()). No data needed. Skip steward and analyst. "
            "When user asks for custom appearance (longer vectors, dark background, front view, bigger nodes, etc.), pass the matching params (vector_scale, background_color, camera_elevation, camera_azimuth, node_size, etc.) to the visualizer. STOP after producing the plot.\n\n"
        ),
    },
    INTENT_THEORY_MRT_MATRIX: {
        "page_id": "theory_equations",
        "primary_tool": "get_theory_mrt_matrix",
        "prevent_tools": ["get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested MRT MATRIX (transformation matrix M, M⁻¹, relaxation vector S) from Theory & Equations page. "
            "Use get_theory_mrt_matrix ONLY. Delegate: visualizer (get_theory_mrt_matrix()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },
    INTENT_THEORY_EQUATIONS_FULL: {
        "page_id": "theory_equations",
        "primary_tool": "get_theory_ns_equations",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested THEORY & EQUATIONS page content (generic). "
            "Start with get_theory_ns_equations, then get_theory_lbm_formulation. Optionally add plot_d3q19_lattice and get_theory_mrt_matrix if user wants full page. "
            "Delegate: visualizer. No data needed. Skip steward and analyst. STOP after producing content.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGE 04 — REAL ISOTROPY
    # -------------------------------------------------------------------------
    INTENT_LUMLEY_TRIANGLE: {
        "page_id": "real_isotropy",
        "primary_tool": "plot_lumley_triangle",
        "prevent_tools": ["plot_real_isotropy", "plot_diagonal_bii"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested LUMLEY TRIANGLE (ξ, η trajectory). "
            "Use plot_lumley_triangle ONLY—NOT plot_real_isotropy or plot_diagonal_bii. "
            "Delegate: steward (find eps_real_validation*.csv or turbulence_validation*.csv in path) -> visualizer (plot_lumley_triangle(data_dir=...)). "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again. "
            "If user ALSO asked for theory/equations: after plot, call get_real_isotropy_theory(subplot='B'). Skip analyst.\n\n"
        ),
    },
    INTENT_DIAGONAL_BII: {
        "page_id": "real_isotropy",
        "primary_tool": "plot_diagonal_bii",
        "prevent_tools": ["plot_real_isotropy", "plot_lumley_triangle", "plot_cross_correlations", "plot_deviations", "plot_convergence"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested DIAGONAL b_ii (subplot C): b11, b22, b33 vs t/t0. "
            "Use plot_diagonal_bii ONLY—NOT plot_real_isotropy (energy fractions) or plot_lumley_triangle or plot_cross_correlations. "
            "Delegate: steward (find eps_real_validation*.csv or turbulence_validation*.csv in path) -> visualizer (plot_diagonal_bii(data_dir=..., palette='Dark2' if user asked for different/non-default colors)). "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again. "
            "If user ALSO asked for theory/equations: after plot, call get_real_isotropy_theory(subplot='C'). "
            "When user asks for 'different colors' or 'non-default colors': pass palette='Dark2' or palette='Set1' or custom_colors in style_updates. Skip analyst.\n\n"
        ),
    },
    INTENT_CROSS_CORRELATIONS: {
        "page_id": "real_isotropy",
        "primary_tool": "plot_cross_correlations",
        "prevent_tools": ["plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_deviations", "plot_convergence"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested CROSS-CORRELATIONS (subplot D): |b12|, |b13|, |b23|, anisotropy index vs t/t0. "
            "Use plot_cross_correlations ONLY—NOT plot_real_isotropy, plot_lumley_triangle, plot_diagonal_bii, or plot_deviations. "
            "Delegate: steward (find eps_real_validation*.csv or turbulence_validation*.csv) -> visualizer (plot_cross_correlations(data_dir=...)). "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again. "
            "If user ALSO asked for theory/equations: after plot, call get_real_isotropy_theory(subplot='D'). "
            "tol_list defaults to [0.001, 0.01]; user can request [0.001, 0.005, 0.01]. Skip analyst.\n\n"
        ),
    },
    INTENT_DEVIATIONS: {
        "page_id": "real_isotropy",
        "primary_tool": "plot_deviations",
        "prevent_tools": ["plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_convergence"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested DEVIATIONS (subplot E): |E_x−1/3|, |E_y−1/3|, |E_z−1/3|, max dev vs t/t0. "
            "Use plot_deviations ONLY—NOT plot_real_isotropy, plot_lumley_triangle, plot_diagonal_bii, plot_cross_correlations, or plot_convergence. "
            "Delegate: steward (find eps_real_validation*.csv or turbulence_validation*.csv) -> visualizer (plot_deviations(data_dir=...)). "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again. "
            "If user ALSO asked for theory/equations: after plot, call get_real_isotropy_theory(subplot='E'). "
            "tol_list defaults to [0.01, 0.02]; optional stationary_t for statistical stationarity line. Skip analyst.\n\n"
        ),
    },
    INTENT_CONVERGENCE: {
        "page_id": "real_isotropy",
        "primary_tool": "plot_convergence",
        "prevent_tools": ["plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested CONVERGENCE (subplot F): running std of E_x, E_y, E_z vs t/t0. "
            "Use plot_convergence ONLY—NOT plot_real_isotropy, plot_deviations, or other real isotropy plots. "
            "Delegate: steward (find eps_real_validation*.csv or turbulence_validation*.csv) -> visualizer (plot_convergence(data_dir=...)). "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again. "
            "If user ALSO asked for theory/equations: after plot, call get_real_isotropy_theory(subplot='F'). "
            "conv_windows derived from data length by default. Skip analyst.\n\n"
        ),
    },
    INTENT_REAL_ISOTROPY_SUMMARY: {
        "page_id": "real_isotropy",
        "primary_tool": "get_real_isotropy_summary",
        "prevent_tools": ["plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations", "plot_convergence", "compute_spectral_isotropy", "get_spectral_isotropy_summary"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested REAL ISOTROPY SUMMARY TABLE (Final Ex, Ey, Ez, anisotropy index). "
            "Use get_real_isotropy_summary ONLY—NOT plots or spectral isotropy summary. "
            "Delegate: steward (find eps_real_validation*.csv or turbulence_validation*.csv) -> visualizer (get_real_isotropy_summary(data_dir=...)). "
            "Skip analyst. STOP after the table.\n\n"
        ),
    },
    INTENT_REAL_ISOTROPY_THEORY: {
        "page_id": "real_isotropy",
        "primary_tool": "get_real_isotropy_theory",
        "prevent_tools": ["plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations", "plot_convergence", "get_real_isotropy_summary"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested THEORY & EQUATIONS for real-space isotropy (Reynolds stress, anisotropy tensor, Lumley coordinates, etc.). "
            "Use get_real_isotropy_theory ONLY—no data needed, no plots. "
            "Delegate: visualizer (get_real_isotropy_theory()). Skip steward and analyst.\n\n"
        ),
    },
    INTENT_ENERGY_FRACTIONS: {
        "page_id": "real_isotropy",
        "primary_tool": "plot_real_isotropy",
        "prevent_tools": ["compute_spectra", "plot_spectrum", "compute_spectral_isotropy", "plot_spectral_isotropy", "plot_component_spectra", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations", "plot_convergence"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested REAL ISOTROPY (Page 04). "
            "Lumley/subplot B -> plot_lumley_triangle. Energy fractions or just isotropy -> plot_real_isotropy. "
            "Data: eps_real_validation*.csv or turbulence_validation*.csv. Skip analyst. Delegate: steward (find validation CSV) -> visualizer. "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again. "
            "If user ALSO asked for 'theory' or 'equations' for this subplot: after plot_real_isotropy, call get_real_isotropy_theory(subplot='A') to show only the equations used in the energy fractions subplot.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGE 05 — SPECTRAL ISOTROPY
    # -------------------------------------------------------------------------
    INTENT_SPECTRAL_ISOTROPY: {
        "page_id": "spectral_isotropy",
        "primary_tool": "plot_spectral_isotropy",
        "prevent_tools": ["compute_spectra", "plot_spectrum", "plot_real_isotropy", "plot_lumley_triangle"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested SPECTRAL ISOTROPY (Page 05). "
            "IC(k) -> compute_spectral_isotropy + plot_spectral_isotropy. "
            "Data: isotropy_coeff_*.dat. Delegate: steward (find isotropy_coeff_*.dat) -> analyst (compute_spectral_isotropy) -> visualizer. "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again.\n\n"
        ),
    },
    INTENT_SPECTRAL_ISOTROPY_SUMMARY: {
        "page_id": "spectral_isotropy",
        "primary_tool": "get_spectral_isotropy_summary",
        "prevent_tools": ["compute_spectra", "plot_spectrum", "plot_real_isotropy", "plot_lumley_triangle", "plot_spectral_isotropy", "plot_component_spectra"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested SPECTRAL ISOTROPY SUMMARY TABLE ONLY (Page 05, Tab 3). "
            "Do NOT plot. Delegate: steward (find isotropy_coeff_*.dat) -> analyst (compute_spectral_isotropy) -> visualizer (get_spectral_isotropy_summary). STOP after the table. "
            "Never delegate to plot_spectral_isotropy or plot_component_spectra for this request.\n\n"
        ),
    },
    INTENT_COMPONENT_SPECTRA: {
        "page_id": "spectral_isotropy",
        "primary_tool": "plot_component_spectra",
        "prevent_tools": ["compute_spectra", "plot_spectrum", "plot_spectral_isotropy"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested COMPONENT SPECTRA (E11, E22, E33). "
            "Use compute_spectral_isotropy first, then plot_component_spectra. "
            "Delegate: steward (find isotropy_coeff_*.dat) -> analyst (compute_spectral_isotropy) -> visualizer (plot_component_spectra). "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again.\n\n"
        ),
    },
    INTENT_SPECTRAL_ISOTROPY_THEORY: {
        "page_id": "spectral_isotropy",
        "primary_tool": "get_spectral_isotropy_theory",
        "prevent_tools": ["compute_spectral_isotropy", "plot_spectral_isotropy", "plot_component_spectra", "get_spectral_isotropy_summary"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested SPECTRAL ISOTROPY THEORY/EQUATIONS (Page 05): E11/E22/E33, IC(k), isotropic turbulence. "
            "Use get_spectral_isotropy_theory ONLY. Delegate: visualizer (get_spectral_isotropy_theory()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGE 06 — ENERGY SPECTRA
    # -------------------------------------------------------------------------
    INTENT_ENERGY_SPECTRA: {
        "page_id": "energy_spectra",
        "primary_tool": "plot_spectrum",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested ENERGY SPECTRA (raw E(k) or normalized). "
            "Exactly 3 steps: steward (find spectrum*.dat) -> analyst (compute_spectra) -> visualizer (plot_spectrum). "
            "ONE plot only. When Context has 'Computed spectra' or data_reference=current_spectra_data, delegate directly to visualizer—do NOT re-delegate to analyst. "
            "STOP after producing the plot. Do not delegate plot_spectrum again."
        ),
        "intent_override_evolution": (
            "INTENT_OVERRIDE: User requested TIME EVOLUTION spectra. "
            "Exactly 3 steps: steward (find spectrum*.dat) -> analyst (compute_spectra mode=evolution) -> visualizer (plot_spectrum mode=evolution). "
            "ONE plot only. When Context has 'Computed spectra' or data_reference=current_spectra_evolution, delegate directly to visualizer—do NOT re-delegate to analyst. "
            "STOP after producing the plot. Do not delegate plot_spectrum again."
        ),
    },
    INTENT_ENERGY_SPECTRA_THEORY: {
        "page_id": "energy_spectra",
        "primary_tool": "get_energy_spectra_theory",
        "prevent_tools": ["compute_spectra", "plot_spectrum"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested ENERGY SPECTRA THEORY/EQUATIONS (Page 06): E(k), Kolmogorov, Pope model, normalized spectrum. "
            "Use get_energy_spectra_theory ONLY. Delegate: visualizer (get_energy_spectra_theory()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGES 07-09 — Flatness, Structure Functions, PDFs
    # -------------------------------------------------------------------------
    INTENT_FLATNESS: {
        "page_id": "flatness",
        "primary_tool": "plot_flatness",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested FLATNESS (Page 07). "
            "Delegate: steward (find flatness_data*_*.txt) -> analyst (compute_flatness) -> visualizer (plot_flatness). "
            "ONE plot only. STOP after producing the plot. Do not delegate to visualizer again. "
            "If user ALSO asked for summary: after plot, call get_flatness_summary()."
        ),
    },
    INTENT_FLATNESS_THEORY: {
        "page_id": "flatness",
        "primary_tool": "get_flatness_theory",
        "prevent_tools": ["compute_flatness", "plot_flatness"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested FLATNESS THEORY/EQUATIONS (Page 07): F_L(r), longitudinal velocity increment, Gaussian reference (F=3), intermittency. "
            "Use get_flatness_theory ONLY. Delegate: visualizer (get_flatness_theory()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },
    INTENT_STRUCTURE_FUNCTIONS: {
        "page_id": "structure_functions",
        "primary_tool": "plot_structure_functions",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested STRUCTURE FUNCTIONS (Page 08): S_p(r), ESS, or anomalies. "
            "Exactly 3 steps: steward (find structure_functions_*.txt or structure_funcs*_t*.bin) -> analyst (compute_structure_functions) -> visualizer (plot_structure_functions). "
            "ONE plot only. When Context has 'Computed structure functions' or data_reference=current_structure_functions_data, delegate directly to visualizer—do NOT re-delegate to analyst. "
            "STOP after producing the plot. Do not delegate plot_structure_functions again."
        ),
    },
    INTENT_STRUCTURE_FUNCTIONS_THEORY: {
        "page_id": "structure_functions",
        "primary_tool": "get_structure_functions_theory",
        "prevent_tools": ["compute_structure_functions", "plot_structure_functions"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested STRUCTURE FUNCTIONS THEORY/EQUATIONS (Page 08): S_p(r), ESS, She-Leveque scaling. "
            "Use get_structure_functions_theory ONLY. Delegate: visualizer (get_structure_functions_theory()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },
    INTENT_PDF: {
        "page_id": "pdfs",
        "primary_tool": "plot_pdf",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested PDFs (Page 09): probability density functions from velocity fields—vorticity, enstrophy, dissipation, velocity magnitude, or joint PDFs (R-Q, velocity-dissipation, etc.). "
            "Delegate: steward (find *.vti or *.h5 or *.hdf5 in path) -> visualizer (plot_pdf). "
            "Skip analyst. Data: velocity fields from VTI/HDF5. ONE plot only. STOP after producing the plot.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGE 10 — OTHER TURBULENCE STATS
    # -------------------------------------------------------------------------
    INTENT_OTHER_TURBULENCE_STATS: {
        "page_id": "other_turbulence_stats",
        "primary_tool": "plot_turbulence_stats",
        "prevent_tools": ["get_turbulence_stats_summary"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested OTHER TURBULENCE STATS (Page 10): custom x-y plot from turbulence_stats*.csv or eps_real_validation*.csv. "
            "Delegate: steward (find turbulence_stats*.csv or eps_real_validation*.csv in path) -> visualizer (plot_turbulence_stats). "
            "Skip analyst. ONE plot only. STOP after producing the plot.\n\n"
        ),
    },
    INTENT_OTHER_TURBULENCE_STATS_SUMMARY: {
        "page_id": "other_turbulence_stats",
        "primary_tool": "get_turbulence_stats_summary",
        "prevent_tools": ["plot_turbulence_stats"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested OTHER TURBULENCE STATS SUMMARY/TABLE (Page 10). "
            "Delegate: steward (find turbulence_stats*.csv or eps_real_validation*.csv) -> visualizer (get_turbulence_stats_summary). "
            "Skip analyst. STOP after the table.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGE 11 — 3D VOLUME VIEWER
    # -------------------------------------------------------------------------
    INTENT_VOLUME_VIEWER_3D: {
        "page_id": "volume_viewer_3d",
        "primary_tool": "plot_volume_3d",
        "prevent_tools": ["get_volume_viewer_theory"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested 3D VOLUME VIEWER (Page 11): 3D visualization of velocity fields from *.vti, *.h5, *.hdf5. "
            "Delegate: steward (find *.vti or *.h5 or *.hdf5 in path) -> visualizer (plot_volume_3d). "
            "Skip analyst. Pass data_dir when task specifies path (e.g. examples/DNS/512). ONE plot only. STOP after producing the plot.\n\n"
        ),
    },
    INTENT_VOLUME_VIEWER_3D_THEORY: {
        "page_id": "volume_viewer_3d",
        "primary_tool": "get_volume_viewer_theory",
        "prevent_tools": ["plot_volume_3d"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested 3D VOLUME VIEWER THEORY/EQUATIONS (Page 11): velocity magnitude, vorticity, Q_S^S, Q/R invariants. "
            "Use get_volume_viewer_theory ONLY. Delegate: visualizer (get_volume_viewer_theory()). No data needed. Skip steward and analyst. STOP after the markdown.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # PAGE 12 — REPORT GENERATOR
    # -------------------------------------------------------------------------
    INTENT_REPORT_ADD_SECTION: {
        "page_id": "report_generator",
        "primary_tool": "add_report_section",
        "prevent_tools": ["generate_report"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested to ADD TO REPORT (Page 12): add section (plot, text, or table). "
            "Use add_report_section. For plot: section_type='plot', title from figure. For text: section_type='text', content=markdown. "
            "For table: section_type='table', table_data=list of dicts (each dict=row). Skip steward and analyst. Delegate to visualizer. STOP after adding.\n\n"
        ),
    },
    INTENT_REPORT_REMOVE: {
        "page_id": "report_generator",
        "primary_tool": "remove_report_section",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested to REMOVE/DELETE a report section. "
            "Use remove_report_section(index=1-based). Extract section number from user (e.g. 'section 2' -> index=2). "
            "Delegate to visualizer. STOP after removing.\n\n"
        ),
    },
    INTENT_REPORT_REORDER: {
        "page_id": "report_generator",
        "primary_tool": "reorder_report_section",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested to MOVE/REORDER a report section. "
            "Use reorder_report_section(from_index, to_index). Extract indices from user (1-based). "
            "E.g. 'move section 2 up' -> from_index=2, to_index=1. 'move section 1 down' -> from_index=1, to_index=2. "
            "Delegate to visualizer. STOP after reordering.\n\n"
        ),
    },
    INTENT_REPORT_EDIT: {
        "page_id": "report_generator",
        "primary_tool": "edit_report_section",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested to EDIT a report section (title, content, caption, header_level). "
            "Use edit_report_section(index=1-based, title=..., content=..., caption=..., header_level=...). "
            "Only pass fields user asked to change. Delegate to visualizer. STOP after editing.\n\n"
        ),
    },
    INTENT_REPORT_GENERATE: {
        "page_id": "report_generator",
        "primary_tool": "generate_report",
        "prevent_tools": ["add_report_section"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested to GENERATE REPORT (Page 12): export the scientific report as HTML or PDF. "
            "Use generate_report. format='pdf' or 'html' based on user preference. Uses report_sections from session. "
            "Skip steward and analyst. Delegate directly to visualizer (generate_report). STOP after generating.\n\n"
        ),
    },
    INTENT_REPORT_PREVIEW: {
        "page_id": "report_generator",
        "primary_tool": "preview_report",
        "prevent_tools": ["generate_report"],
        "intent_override": (
            "INTENT_OVERRIDE: User requested to SEE THE FULL COMPILED REPORT (Page 12): figures, tables, text, sections—everything rendered in chat. "
            "Use preview_report ONLY. Do NOT use generate_report or read_file. "
            "Delegate directly to visualizer (preview_report). STOP after showing the compiled report.\n\n"
        ),
    },

    # -------------------------------------------------------------------------
    # APP SETTINGS — HDF5 format (steward: set_hdf5_format)
    # -------------------------------------------------------------------------
    INTENT_APP_SETTINGS_HDF5_FORTRAN: {
        "page_id": "app_settings",
        "primary_tool": "set_hdf5_format",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested HDF5 format: Fortran (transpose for Fortran-written velocity files). "
            "Delegate to steward: set_hdf5_format(format='fortran'). Steward calls the tool. STOP after steward confirms.\n\n"
        ),
    },
    INTENT_APP_SETTINGS_HDF5_DEFAULT: {
        "page_id": "app_settings",
        "primary_tool": "set_hdf5_format",
        "prevent_tools": [],
        "intent_override": (
            "INTENT_OVERRIDE: User requested HDF5 format: Default (no transpose, Python/standard layout). "
            "Delegate to steward: set_hdf5_format(format='default'). Steward calls the tool. STOP after steward confirms.\n\n"
        ),
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_page_for_intent(intent: Optional[str]) -> Optional[str]:
    """Map intent constant to page_id."""
    route = INTENT_ROUTING.get(intent) if intent else None
    return route.get("page_id") if route else None


def get_routing_for_intent(intent: str, user_input: Optional[str] = None) -> Dict[str, Any]:
    """
    Get full routing config for an intent. Used by intent_detection.get_plot_routing.
    Returns: tool, file_pattern, skip_analyst, intent_override_text, prevent_tools
    """
    route = INTENT_ROUTING.get(intent)
    if not route:
        return {}
    page_id = route.get("page_id")
    cfg = PAGE_SCHEMA.get(page_id, {})
    file_pattern = (cfg.get("file_patterns") or [None])[0]
    skip_analyst = cfg.get("skip_analyst", True)
    primary_tool = route.get("primary_tool")
    prevent_tools = route.get("prevent_tools", [])
    intent_override = route.get("intent_override")
    if intent == INTENT_ENERGY_SPECTRA and user_input and "evolution" in user_input.lower():
        intent_override = route.get("intent_override_evolution") or intent_override
    return {
        "tool": primary_tool,
        "file_pattern": file_pattern,
        "skip_analyst": skip_analyst,
        "intent_override_text": intent_override,
        "prevent_tools": prevent_tools,
    }


def get_workflow_for_page(page_id: str) -> Dict[str, Any]:
    """Return workflow config for a page. Empty dict if unknown."""
    return PAGE_SCHEMA.get(page_id, {})


def format_catalog() -> str:
    """Generate PAGE_CATALOG string from schema for orchestrator prompt."""
    lines = [
        "AVAILABLE PAGES & TOOLS (match user intent to the right tool).",
        "Where you see 'X(...) -> Y | Z', step X must be completed before Y or Z.",
        "",
    ]
    for i, (pid, cfg) in enumerate(PAGE_SCHEMA.items(), 1):
        patterns = ", ".join(cfg.get("file_patterns", []))
        compute = cfg.get("compute_tool")
        plots = cfg.get("plot_tools", [])
        summary = cfg.get("summary_tool")
        skip = cfg.get("skip_analyst", False)
        name = pid.replace("_", " ").title()
        if compute:
            plot_str = " | ".join(plots) if plots else ""
            if summary:
                plot_str = f"{plot_str} | {summary}" if plot_str else summary
            lines.append(f"{i}. {name}: {patterns}")
            lines.append(f"   - {compute}(...) -> {plot_str}")
        else:
            tool_str = " | ".join(plots) if plots else "(tools not yet in chat)"
            if summary:
                tool_str = f"{tool_str} | {summary}" if tool_str else summary
            lines.append(f"{i}. {name}: {patterns}")
            lines.append(f"   - {tool_str}")
        lines.append("")
    return "\n".join(lines).strip()


def get_file_pattern_for_page(page_id: str) -> Optional[str]:
    """First file pattern for a page (for steward)."""
    cfg = PAGE_SCHEMA.get(page_id, {})
    patterns = cfg.get("file_patterns", [])
    return patterns[0] if patterns else None


def get_all_file_patterns() -> Dict[str, str]:
    """Return page_id -> first file pattern for steward prompt."""
    return {
        pid: (cfg.get("file_patterns") or [""])[0]
        for pid, cfg in PAGE_SCHEMA.items()
        if cfg.get("file_patterns")
    }


def get_tool_for_request(page_id: str, request_type: str) -> Optional[str]:
    """
    request_type: "lumley" | "energy_fractions" | "ic_plot" | "component" | "summary" | "spectrum"
    Returns the tool name to use.
    """
    cfg = PAGE_SCHEMA.get(page_id, {})
    if page_id == "overview":
        if request_type in ("theory", "equations"):
            return "get_overview_theory"
        return cfg.get("summary_tool") or "get_overview_summary"
    if page_id == "real_isotropy":
        if request_type == "summary":
            return cfg.get("summary_tool") or "get_real_isotropy_summary"
        if request_type == "lumley":
            return "plot_lumley_triangle"
        if request_type in ("diagonal_bii", "subplot_c"):
            return "plot_diagonal_bii"
        if request_type in ("cross_correlations", "subplot_d"):
            return "plot_cross_correlations"
        if request_type in ("deviations", "subplot_e"):
            return "plot_deviations"
        if request_type in ("convergence", "subplot_f"):
            return "plot_convergence"
        return "plot_real_isotropy"
    if page_id == "spectral_isotropy":
        if request_type == "summary":
            return cfg.get("summary_tool")
        if request_type in ("theory", "equations"):
            return "get_spectral_isotropy_theory"
        if request_type == "component":
            return "plot_component_spectra"
        return "plot_spectral_isotropy"
    if page_id == "energy_spectra":
        if request_type in ("theory", "equations"):
            return "get_energy_spectra_theory"
        return "plot_spectrum"
    if page_id == "flatness":
        if request_type in ("theory", "equations"):
            return "get_flatness_theory"
        if request_type == "summary":
            return "get_flatness_summary"
        return "plot_flatness"
    if page_id == "structure_functions":
        if request_type in ("theory", "equations"):
            return "get_structure_functions_theory"
        return "plot_structure_functions"
    if page_id == "theory_equations":
        if request_type in ("ns_equations", "navier-stokes", "ns"):
            return "get_theory_ns_equations"
        if request_type in ("lbm_formulation", "lbm", "mrt", "bgk", "srt"):
            return "get_theory_lbm_formulation"
        if request_type in ("d3q19_lattice", "lattice", "lattice visualization", "stencil"):
            return "plot_d3q19_lattice"
        if request_type in ("mrt_matrix", "matrix", "transformation matrix"):
            return "get_theory_mrt_matrix"
        return "get_theory_ns_equations"  # default for generic theory request
    if page_id == "other_turbulence_stats":
        if request_type == "summary":
            return "get_turbulence_stats_summary"
        return "plot_turbulence_stats"
    if page_id == "pdfs":
        return "plot_pdf"
    if page_id == "volume_viewer_3d":
        if request_type in ("theory", "equations"):
            return "get_volume_viewer_theory"
        return "plot_volume_3d"
    if page_id == "report_generator":
        if request_type == "generate":
            return "generate_report"
        if request_type == "preview":
            return "preview_report"
        if request_type == "structure":
            return "preview_report"  # structure/structure request -> preview (full report has TOC)
        if request_type == "remove":
            return "remove_report_section"
        if request_type == "reorder":
            return "reorder_report_section"
        if request_type == "edit":
            return "edit_report_section"
        return "add_report_section"
    return (cfg.get("plot_tools") or [None])[0]
