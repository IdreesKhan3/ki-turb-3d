"""
Shell command execution with dangerous command blocking.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ...shared.config import SHELL_COMMAND_TIMEOUT


SHELL_TOOL_NAMES = frozenset({"run_shell_command",})

DANGEROUS_COMMANDS = [
    "rm -rf /",
    "sudo ",
    "mkfs",
    "shutdown",
    "reboot",
    "chmod 777 /",
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for shell commands."""
    return [
        {
            "name": "run_shell_command",
            "description": "Execute a shell command (git status, ls, etc). Use for Git and file system operations.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Shell command to run"}},
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    """Execute shell command with dangerous command blocking."""
    if name != "run_shell_command":
        return f"Error: Unknown shell tool '{name}'"

    cmd = args.get("cmd", "")
    if not cmd:
        return "Error: cmd required"

    # Block dangerous commands
    if any(dangerous in cmd for dangerous in DANGEROUS_COMMANDS):
        return "Error: Blocked dangerous system command."

    try:
        res = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=SHELL_COMMAND_TIMEOUT,
        )
        return res.stdout if res.returncode == 0 else f"Error: {res.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {SHELL_COMMAND_TIMEOUT} seconds."
    except Exception as e:
        return f"Error: {e}"
