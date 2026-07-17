"""Tests that lock per-agent tool permissions against an expected reference set."""

from agents.runtime import tool_registry
from agents.runtime.models import AgentName


# Expected per-agent permissions. Changes to agent capabilities must be reflected
# here, causing these tests to fail if permissions drift unintentionally.
GOLDEN_STEWARD = {
    "list_directory", "find_file", "read_file", "read_document",
    "set_app_theme", "load_data", "load_dataset_manifest", "list_simulation_jobs",
    "load_analysis_products", "get_analysis_product_summary",
    "set_selection_mode", "set_hdf5_format",
    "search_codebase", "extract_section", "regex_search",
    "run_shell_command", "git_operation",
    "delete_file", "modify_file", "rename_file", "write_file",
    "web_search", "search_research_papers", "browse_web", "download_file",
    "run_pytest", "run_import_check", "run_verify_command",
}
GOLDEN_ANALYST = {
    "list_directory", "find_file", "read_file", "read_document",
    "list_simulation_jobs",
    "load_analysis_products", "get_analysis_product_summary",
    "compute_overview_validation", "compute_pdfs", "compute_volume_field",
    "compute_spectra", "compute_spectral_isotropy", "compute_isotropy",
    "compute_flatness", "compute_structure_functions", "export_data",
    "web_search", "search_research_papers", "browse_web", "download_file",
    "semantic_search", "find_symbol_definitions", "find_symbol_references",
    "write_file",
    "generate_content", "generate_code", "compile_latex",
}
GOLDEN_VISUALIZER = {
    "plot_spectrum", "get_energy_spectra_theory", "plot_spectral_isotropy",
    "plot_component_spectra", "get_spectral_isotropy_summary", "get_spectral_isotropy_theory",
    "plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii",
    "plot_cross_correlations", "plot_deviations", "plot_convergence",
    "get_real_isotropy_summary", "get_real_isotropy_theory", "export_figure",
    "export_data", "export_isotropy_data",
    "get_overview_summary", "get_overview_theory", "get_analysis_product_summary",
    "get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice",
    "get_theory_mrt_matrix",
    "plot_flatness", "get_flatness_summary", "get_flatness_theory", "export_flatness_data",
    "plot_structure_functions", "get_structure_functions_theory",
    "plot_turbulence_stats", "get_turbulence_stats_columns", "get_turbulence_stats_summary",
    "plot_volume_3d", "get_volume_viewer_theory",
    "plot_pdf",
    "add_report_section", "generate_report", "preview_report",
    "remove_report_section", "reorder_report_section", "edit_report_section",
}
GOLDEN_SIMULATION = {
    "build_simulation_case", "compile_simulation", "start_simulation",
    "check_simulation_status", "cancel_simulation", "supervise_simulation",
    "fetch_simulation_outputs",
    "postprocess_simulation_outputs", "read_dataset_manifest", "load_dataset_manifest",
    "list_simulation_jobs",
    "load_analysis_products", "get_analysis_product_summary",
    "web_search", "browse_web", "search_research_papers",
}
GOLDEN_ORCHESTRATOR = {
    "web_search", "search_research_papers", "browse_web",
}
GOLDEN_REVIEWER = {
    "web_search", "search_research_papers", "browse_web",
}
GOLDEN_ENGINEER = {
    "list_directory", "find_file", "read_file",
    "search_codebase", "extract_section", "regex_search",
    "semantic_search", "find_symbol_definitions", "find_symbol_references",
    "git_operation", "delete_file", "modify_file", "rename_file", "write_file",
    "run_shell_command",
    "web_search", "search_research_papers", "browse_web", "download_file",
    "run_pytest", "run_import_check", "run_verify_command",
}
GOLDEN_CONFIRMABLE = {
    "delete_file", "rename_file", "write_file", "modify_file", "download_file",
    "build_simulation_case", "compile_simulation", "start_simulation",
    "cancel_simulation", "fetch_simulation_outputs", "postprocess_simulation_outputs",
}


def test_steward_permissions_match_golden():
    assert set(tool_registry.tools_for_agent("steward")) == GOLDEN_STEWARD


def test_analyst_permissions_match_golden():
    assert set(tool_registry.tools_for_agent("analyst")) == GOLDEN_ANALYST


def test_visualizer_permissions_match_golden():
    assert set(tool_registry.tools_for_agent("visualizer")) == GOLDEN_VISUALIZER


def test_simulation_permissions_match_golden():
    assert set(tool_registry.tools_for_agent("simulation")) == GOLDEN_SIMULATION


def test_orchestrator_and_reviewer_have_web_research_tools():
    assert set(tool_registry.tools_for_agent("orchestrator")) == GOLDEN_ORCHESTRATOR
    assert set(tool_registry.tools_for_agent("reviewer")) == GOLDEN_REVIEWER


def test_engineer_permissions_match_golden():
    assert set(tool_registry.tools_for_agent("engineer")) == GOLDEN_ENGINEER


def test_orchestrator_and_reviewer_receive_web_tool_definitions():
    from agents import tools
    orch = {d["name"] for d in tools.get_tools_for_agent("orchestrator")}
    rev = {d["name"] for d in tools.get_tools_for_agent("reviewer")}
    assert orch == GOLDEN_ORCHESTRATOR
    assert rev == GOLDEN_REVIEWER


def test_engineer_receives_verify_tool_definitions():
    from agents import tools
    names = {d["name"] for d in tools.get_tools_for_agent("engineer")}
    assert names == GOLDEN_ENGINEER
    assert {"run_pytest", "run_import_check", "run_verify_command"} <= names


def test_confirmable_tools_match_golden():
    assert set(tool_registry.confirmable_tools()) == GOLDEN_CONFIRMABLE


def test_confirmable_tools_are_all_registered():
    registered = tool_registry.all_registered_tools()
    for name in GOLDEN_CONFIRMABLE:
        assert name in registered
        assert tool_registry.requires_confirmation(name) is True


def test_is_agent_allowed_matches_membership():
    assert tool_registry.is_agent_allowed("plot_spectrum", "visualizer") is True
    assert tool_registry.is_agent_allowed("plot_spectrum", "steward") is False
    assert tool_registry.is_agent_allowed("run_shell_command", "steward") is True
    assert tool_registry.is_agent_allowed("web_search", "steward") is True
    assert tool_registry.is_agent_allowed("web_search", "orchestrator") is True
    assert tool_registry.is_agent_allowed("web_search", "simulation") is True
    # Unknown tools and agents are disallowed.
    assert tool_registry.is_agent_allowed("nonexistent_tool", "steward") is False
    assert tool_registry.is_agent_allowed("plot_spectrum", "nobody") is False


def test_export_data_shared_between_analyst_and_visualizer():
    spec = tool_registry.get_spec("export_data")
    assert spec is not None
    assert AgentName.ANALYST in spec.allowed_agents
    assert AgentName.VISUALIZER in spec.allowed_agents


def test_tools_derived_in_tools_package_equal_registry():
    from agents import tools
    assert set(tools.STEWARD_TOOL_NAMES) == GOLDEN_STEWARD
    assert set(tools.ANALYST_TOOL_NAMES) == GOLDEN_ANALYST
    assert set(tools.VISUALIZER_TOOL_NAMES) == GOLDEN_VISUALIZER
    assert set(tools.CONFIRMABLE_TOOLS) == GOLDEN_CONFIRMABLE
