"""Determine which actions require user confirmation.

The set of confirmable tools is sourced from the tool registry
(``confirmation_required`` on each tool spec), and confirmation messages are
formatted here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..runtime import tool_registry


def requires_confirmation(tool: str, args: Optional[Dict[str, Any]] = None) -> bool:
    """Return whether an action must be confirmed before execution.

    ``args`` is accepted so future policies can decide per-invocation; the current
    decision is per-tool.
    """
    return tool_registry.requires_confirmation(tool)


def confirmation_message(tool: str, args: Optional[Dict[str, Any]] = None) -> str:
    """Return a human-readable summary of a pending action for the user."""
    args = args or {}
    if tool == "delete_file":
        path = args.get("filepath", "?")
        if args.get("recursive"):
            return f"Delete directory recursively: {path}"
        return f"Delete file or empty directory: {path}"
    if tool == "rename_file":
        return f"Rename {args.get('filepath', '?')} → {args.get('new_filepath', '?')}"
    if tool == "write_file":
        return f"Write/overwrite file: {args.get('filepath', '?')}"
    if tool == "modify_file":
        return f"Modify file: {args.get('filepath', '?')}"
    if tool == "download_file":
        return f"Download to: {args.get('save_path', args.get('url', '?')[:50])}"
    if tool == "run_shell_command":
        return f"Run shell command: {args.get('cmd', '?')}"
    return f"{tool}: {args}"
