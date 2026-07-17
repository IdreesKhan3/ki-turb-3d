"""Shell command execution restricted by the shell allowlist policy.

Commands are validated by ``agents/security/shell_policy.py``, tokenized, and run
with ``shell=False`` to prevent shell metacharacter injection.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ...shared.config import SHELL_COMMAND_TIMEOUT
from ...security.shell_policy import ShellPolicyError, to_argv


SHELL_TOOL_NAMES = frozenset({"run_shell_command",})


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for shell commands."""
    return [
        {
            "name": "run_shell_command",
            "description": "Execute an allowlisted, read-only shell command (git, ls, cat, grep, find, etc.). No pipes, redirects, or command chaining.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Shell command to run (single allowlisted program)"}},
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    """Execute an allowlisted shell command with no shell interpolation."""
    if name != "run_shell_command":
        return f"Error: Unknown shell tool '{name}'"

    cmd = args.get("cmd", "")
    if not cmd:
        return "Error: cmd required"

    # Validate against the allowlist and tokenize into argv (raises on violation).
    try:
        argv = to_argv(cmd)
    except ShellPolicyError as e:
        return f"Error: Blocked command ({e}). Only allowlisted, non-chained commands are permitted."

    try:
        res = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=SHELL_COMMAND_TIMEOUT,
        )
        return res.stdout if res.returncode == 0 else f"Error: {res.stderr}"
    except FileNotFoundError:
        return f"Error: Command not found: {argv[0]}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {SHELL_COMMAND_TIMEOUT} seconds."
    except Exception as e:
        return f"Error: {e}"
