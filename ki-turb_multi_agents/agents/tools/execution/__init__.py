"""
Execution tools: shell commands, git operations.
"""

from pathlib import Path
from typing import Any, Dict, List

from . import shell_commands
from . import git_operations as git_ops


SHELL_TOOL_NAMES = shell_commands.SHELL_TOOL_NAMES
GIT_TOOL_NAMES = git_ops.GIT_TOOL_NAMES
EXECUTION_TOOL_NAMES = SHELL_TOOL_NAMES | GIT_TOOL_NAMES


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for execution tools."""
    tools = []
    tools.extend(shell_commands.get_tool_definitions())
    tools.extend(git_ops.get_tool_definitions())
    return tools


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    """Execute an execution tool."""
    if name in SHELL_TOOL_NAMES:
        return shell_commands.execute_tool(name, args, project_root)
    if name in GIT_TOOL_NAMES:
        return git_ops.execute_tool(name, args, project_root)
    return f"Error: Unknown execution tool '{name}'"
