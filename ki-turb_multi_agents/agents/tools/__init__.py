"""Agent tool facade with role-scoped lazy imports.

The simulation and physics-control agents never receive the unrestricted shell
executor.  Importing the simulation tool set on a headless compute node also
avoids importing Streamlit/Plotly-heavy visualization modules.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from ..runtime import tool_registry
from ..security import approval_policy

STEWARD_TOOL_NAMES = tool_registry.tools_for_agent("steward")
SIMULATION_TOOL_NAMES = tool_registry.tools_for_agent("simulation")
ANALYST_TOOL_NAMES = tool_registry.tools_for_agent("analyst")
VISUALIZER_TOOL_NAMES = tool_registry.tools_for_agent("visualizer")
REVIEWER_TOOL_NAMES = tool_registry.tools_for_agent("reviewer")
ORCHESTRATOR_TOOL_NAMES = tool_registry.tools_for_agent("orchestrator")
ENGINEER_TOOL_NAMES = tool_registry.tools_for_agent("engineer")
CONFIRMABLE_TOOLS = tool_registry.confirmable_tools()
_format_confirmation_message = approval_policy.confirmation_message

_ROLE_MODULES = {
    "orchestrator": ("search",),
    "reviewer": ("search",),
    "simulation": ("simulation", "search"),
    "steward": ("app_control", "core", "documents", "execution", "search", "simulation", "verify"),
    "analyst": ("core", "documents", "search", "physics", "generation", "simulation"),
    "visualizer": ("physics",),
    "engineer": ("core", "execution", "search", "verify"),
}

# Exact routing keeps execution lazy and prevents accidental shell exposure.
_APP_CONTROL_NAMES = {
    "set_app_theme", "load_data", "load_dataset_manifest", "set_selection_mode",
    "set_hdf5_format",
}
_EXECUTION_NAMES = {"run_shell_command", "git_operation"}
_DOCUMENT_NAMES = {"read_document"}
_GENERATION_NAMES = {"generate_content", "generate_code", "compile_latex"}
_VERIFY_NAMES = {"run_pytest", "run_import_check", "run_verify_command"}
_SEARCH_NAMES = {
    "web_search", "search_research_papers", "browse_web", "download_file",
    "semantic_search", "find_symbol_definitions", "find_symbol_references",
    "search_codebase", "extract_section", "regex_search",
}
_PHYSICS_NAMES = {
    "load_analysis_products", "get_analysis_product_summary",
    "compute_spectra", "compute_spectral_isotropy", "compute_isotropy",
    "compute_flatness", "compute_structure_functions", "compute_pdfs",
    "compute_volume_field", "compute_overview_validation",
    "export_data",
}


def _load_module(key: str):
    return import_module(f"{__package__}.{key}")


def _module_keys_for_role(agent_name: str) -> Iterable[str]:
    return _ROLE_MODULES.get((agent_name or "").strip().lower(), ())


def _module_key_for_tool(name: str) -> str:
    # Specialized executors first. Role permission sets overlap (e.g. simulation
    # may *use* web_search, but the implementation lives in the search module).
    if name in _APP_CONTROL_NAMES:
        return "app_control"
    if name in _EXECUTION_NAMES:
        return "execution"
    if name in _VERIFY_NAMES:
        return "verify"
    if name in _DOCUMENT_NAMES:
        return "documents"
    if name in _GENERATION_NAMES:
        return "generation"
    if name in _SEARCH_NAMES:
        return "search"
    if name in _PHYSICS_NAMES or name in VISUALIZER_TOOL_NAMES:
        return "physics"
    if name in SIMULATION_TOOL_NAMES:
        return "simulation"
    return "core"


def _definitions_for_modules(keys: Iterable[str]) -> List[Dict[str, Any]]:
    definitions: List[Dict[str, Any]] = []
    for key in keys:
        module = _load_module(key)
        definitions.extend(module.get_tool_definitions())
    return definitions


def get_tools_definition() -> List[Dict[str, Any]]:
    """Return all tool definitions; mainly used by UI/introspection code."""
    keys = ("app_control", "core", "documents", "execution", "search", "physics", "generation", "simulation", "verify")
    definitions = _definitions_for_modules(keys)
    tool_registry.enrich_from_definitions(definitions)
    return definitions


def get_tools_for_agent(agent_name: str) -> List[Dict[str, Any]]:
    permission_map = {
        "orchestrator": ORCHESTRATOR_TOOL_NAMES,
        "steward": STEWARD_TOOL_NAMES,
        "simulation": SIMULATION_TOOL_NAMES,
        "analyst": ANALYST_TOOL_NAMES,
        "visualizer": VISUALIZER_TOOL_NAMES,
        "reviewer": REVIEWER_TOOL_NAMES,
        "engineer": ENGINEER_TOOL_NAMES,
    }
    role = (agent_name or "").strip().lower()
    allowed = permission_map.get(role)
    if not allowed:
        return []
    definitions = _definitions_for_modules(_module_keys_for_role(role))
    tool_registry.enrich_from_definitions(definitions)
    return [item for item in definitions if item.get("name") in allowed]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
    allowed_tool_names: Optional[Set[str]] = None,
) -> Union[str, Dict[str, Any]]:
    if session_context is None:
        session_context = {}
    if allowed_tool_names is not None and name not in allowed_tool_names:
        if name.lower() in ("steward", "simulation", "analyst", "visualizer", "orchestrator", "reviewer", "engineer"):
            return f"Error: '{name}' is an agent, not a tool. The orchestrator must delegate to it."
        return f"Error: Tool '{name}' is not available for this agent."

    # Fail closed before confirmation for shell commands that can never run.
    # Otherwise Accept → blocked → retry → Accept loops forever in the UI.
    if name == "run_shell_command":
        from ..security.shell_policy import ShellPolicyError, to_argv

        try:
            to_argv(str((args or {}).get("cmd") or ""))
        except ShellPolicyError as exc:
            return (
                f"Error: Blocked command ({exc}). "
                "Only allowlisted, non-chained commands are permitted. "
                "For deleting files/directories use delete_file (recursive=true for dirs), "
                "not shell rm/python."
            )

    if name in CONFIRMABLE_TOOLS:
        if session_context.get("tool_confirmation_rejected"):
            return "User rejected the operation."
        if not session_context.get("tool_confirmation_approved"):
            return {
                "status": "pending_confirmation",
                "tool": name,
                "args": args,
                "message": _format_confirmation_message(name, args),
            }

    try:
        key = _module_key_for_tool(name)
        module = _load_module(key)
        names_attr = {
            "core": "CORE_TOOL_NAMES",
            "execution": "EXECUTION_TOOL_NAMES",
            "search": "SEARCH_TOOL_NAMES",
            "physics": "PHYSICS_TOOL_NAMES",
            "app_control": "APP_CONTROL_TOOL_NAMES",
            "generation": "GENERATION_TOOL_NAMES",
            "documents": "DOCUMENT_TOOL_NAMES",
            "simulation": "SIMULATION_TOOL_NAMES",
            "verify": "VERIFY_TOOL_NAMES",
        }[key]
        if name not in getattr(module, names_attr):
            return f"Error: Unknown tool '{name}'"
        with_context = key in {
            "physics", "app_control", "generation", "documents", "simulation", "core",
        }
        return (
            module.execute_tool(name, args, project_root, session_context)
            if with_context
            else module.execute_tool(name, args, project_root)
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        return f"Tool error: {type(exc).__name__}: {exc}"


__all__ = [
    "get_tools_definition", "get_tools_for_agent", "execute_tool",
    "STEWARD_TOOL_NAMES", "SIMULATION_TOOL_NAMES", "ANALYST_TOOL_NAMES",
    "VISUALIZER_TOOL_NAMES", "REVIEWER_TOOL_NAMES", "ORCHESTRATOR_TOOL_NAMES",
    "ENGINEER_TOOL_NAMES",
]
