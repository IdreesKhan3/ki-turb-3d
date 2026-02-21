"""
Agent Tools — Capabilities each LLM agent can call.

Aggregates tool definitions from core, execution, search, physics modules.
Per-agent tool sets prevent scope creep (e.g. Steward cannot plot).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from . import app_control, core, execution, search, physics, generation

STEWARD_TOOL_NAMES = frozenset({
    "list_directory", "find_file", "read_file",
    "set_app_theme", "load_data", "set_selection_mode",
    "search_codebase", "extract_section", "regex_search",
    "run_shell_command", "git_operation",
    "delete_file", "modify_file", "rename_file", "write_file",
})
ANALYST_TOOL_NAMES = frozenset({
    "list_directory", "find_file", "read_file",
    "compute_spectra", "compute_spectral_isotropy", "compute_isotropy", "export_data",
    "web_search", "search_research_papers", "browse_web", "download_file",
    "semantic_search", "find_symbol_definitions", "find_symbol_references",
    "write_file",
    "generate_content", "generate_code", "compile_latex",
})
VISUALIZER_TOOL_NAMES = frozenset({
    "plot_spectrum", "get_energy_spectra_theory", "plot_spectral_isotropy", "plot_component_spectra", "get_spectral_isotropy_summary", "get_spectral_isotropy_theory",
    "plot_real_isotropy", "plot_lumley_triangle", "plot_diagonal_bii", "plot_cross_correlations", "plot_deviations",
    "plot_convergence", "get_real_isotropy_summary", "get_real_isotropy_theory", "export_figure", "export_data", "export_isotropy_data",
    "get_overview_summary", "get_overview_theory",
    "get_theory_ns_equations", "get_theory_lbm_formulation", "plot_d3q19_lattice", "get_theory_mrt_matrix",
})
REVIEWER_TOOL_NAMES = frozenset()
ORCHESTRATOR_TOOL_NAMES = frozenset()

CONFIRMABLE_TOOLS = frozenset({
    "delete_file", "rename_file", "create_file", "write_file", "modify_file", "download_file",
    "run_shell_command",
})


def _format_confirmation_message(tool: str, args: Dict[str, Any]) -> str:
    """Human-readable summary of the pending action."""
    if tool == "delete_file":
        return f"Delete file: {args.get('filepath', '?')}"
    if tool == "rename_file":
        return f"Rename {args.get('filepath', '?')} → {args.get('new_filepath', '?')}"
    if tool == "create_file":
        return f"Create file: {args.get('filepath', '?')}"
    if tool == "write_file":
        return f"Write/overwrite file: {args.get('filepath', '?')}"
    if tool == "modify_file":
        return f"Modify file: {args.get('filepath', '?')}"
    if tool == "download_file":
        return f"Download to: {args.get('save_path', args.get('url', '?')[:50])}"
    if tool == "run_shell_command":
        return f"Run shell command: {args.get('cmd', '?')}"
    return f"{tool}: {args}"


def get_tools_definition() -> List[Dict[str, Any]]:
    """Tool definitions for the LLM (function-calling format)."""
    tools = []
    tools.extend(app_control.get_tool_definitions())
    tools.extend(core.get_tool_definitions())
    tools.extend(execution.get_tool_definitions())
    tools.extend(search.get_tool_definitions())
    tools.extend(physics.get_tool_definitions())
    tools.extend(generation.get_tool_definitions())
    return tools


def get_tools_for_agent(agent_name: str) -> List[Dict[str, Any]]:
    """Return only the tools this agent is allowed to use. Prevents scope creep."""
    all_tools = get_tools_definition()
    allowed: Set[str] = {
        "orchestrator": ORCHESTRATOR_TOOL_NAMES,
        "steward": STEWARD_TOOL_NAMES,
        "analyst": ANALYST_TOOL_NAMES,
        "visualizer": VISUALIZER_TOOL_NAMES,
        "reviewer": REVIEWER_TOOL_NAMES,
    }.get(agent_name.lower(), set())
    if not allowed:
        return all_tools
    return [t for t in all_tools if t.get("name") in allowed]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
    allowed_tool_names: Optional[Set[str]] = None,
) -> Union[str, Dict[str, Any]]:
    """Execute a tool and return result (string or dict for plot artifact).
    session_context: Optional dict with spectra_plot_styles, axis_labels, etc. for plot_spectrum.
    allowed_tool_names: If set, reject tool calls not in this set (prevents scope creep).
    For confirmable tools: session_context may contain tool_confirmation_approved or
    tool_confirmation_rejected to skip the confirmation prompt.
    """
    session_context = session_context or {}
    if allowed_tool_names is not None and name not in allowed_tool_names:
        return f"Error: Tool '{name}' is not available for this agent. Use only your assigned tools."

    if name in CONFIRMABLE_TOOLS:
        approved = session_context.get("tool_confirmation_approved")
        rejected = session_context.get("tool_confirmation_rejected")
        if rejected:
            return "User rejected the operation."
        if not approved:
            return {
                "status": "pending_confirmation",
                "tool": name,
                "args": args,
                "message": _format_confirmation_message(name, args),
            }

    try:
        if name in core.CORE_TOOL_NAMES:
            return core.execute_tool(name, args, project_root)
        if name in execution.EXECUTION_TOOL_NAMES:
            return execution.execute_tool(name, args, project_root)
        if name in search.SEARCH_TOOL_NAMES:
            return search.execute_tool(name, args, project_root)
        if name in physics.PHYSICS_TOOL_NAMES:
            return physics.execute_tool(name, args, project_root, session_context)
        if name in app_control.APP_CONTROL_TOOL_NAMES:
            return app_control.execute_tool(name, args, project_root, session_context)
        if name in generation.GENERATION_TOOL_NAMES:
            return generation.execute_tool(name, args, project_root, session_context)
        return f"Error: Unknown tool '{name}'"
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"
