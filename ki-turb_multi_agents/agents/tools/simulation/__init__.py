"""Simulation tools: prepare, run, monitor, and fetch CFD simulation results.

These tools bridge the agent runtime to the CFD backends in the ``integrations``
package. Case files and job records are stored under ``<project>/simulations``.
"""

from pathlib import Path
from typing import Any, Dict, List

from . import case_builder, run_control, data_fetch, manifest, postprocess
from . import compile as compile_tool

CASE_BUILDER_TOOL_NAMES = case_builder.CASE_BUILDER_TOOL_NAMES
RUN_CONTROL_TOOL_NAMES = run_control.RUN_CONTROL_TOOL_NAMES
DATA_FETCH_TOOL_NAMES = data_fetch.DATA_FETCH_TOOL_NAMES
MANIFEST_TOOL_NAMES = manifest.MANIFEST_TOOL_NAMES
COMPILE_TOOL_NAMES = compile_tool.COMPILE_TOOL_NAMES
POSTPROCESS_TOOL_NAMES = postprocess.POSTPROCESS_TOOL_NAMES

SIMULATION_TOOL_NAMES = (
    CASE_BUILDER_TOOL_NAMES
    | COMPILE_TOOL_NAMES
    | RUN_CONTROL_TOOL_NAMES
    | DATA_FETCH_TOOL_NAMES
    | POSTPROCESS_TOOL_NAMES
    | MANIFEST_TOOL_NAMES
)


def get_tool_definitions() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    tools.extend(case_builder.get_tool_definitions())
    tools.extend(compile_tool.get_tool_definitions())
    tools.extend(run_control.get_tool_definitions())
    tools.extend(data_fetch.get_tool_definitions())
    tools.extend(postprocess.get_tool_definitions())
    tools.extend(manifest.get_tool_definitions())
    return tools


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Dict[str, Any] | None = None,
) -> str:
    if session_context is None:
        session_context = {}
    if name in CASE_BUILDER_TOOL_NAMES:
        return case_builder.execute_tool(name, args, project_root)
    if name in COMPILE_TOOL_NAMES:
        return compile_tool.execute_tool(name, args, project_root, session_context=session_context)
    if name in RUN_CONTROL_TOOL_NAMES:
        return run_control.execute_tool(name, args, project_root, session_context=session_context)
    if name in DATA_FETCH_TOOL_NAMES:
        return data_fetch.execute_tool(name, args, project_root)
    if name in POSTPROCESS_TOOL_NAMES:
        return postprocess.execute_tool(name, args, project_root)
    if name in MANIFEST_TOOL_NAMES:
        return manifest.execute_tool(name, args, project_root)
    return f"Error: Unknown simulation tool '{name}'"
