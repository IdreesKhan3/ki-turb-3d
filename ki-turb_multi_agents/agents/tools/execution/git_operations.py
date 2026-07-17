"""
Git operations tool — wraps agents.data_steward.git_operations for agent tools.
"""

from pathlib import Path
from typing import Any, Dict, List


GIT_TOOL_NAMES = frozenset({"git_operation",})


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for git operations."""
    return [
        {
            "name": "git_operation",
            "description": "Git operations. Use for status, log, diff, add, commit, push, pull, branch, stash, tag, remote. Prefer over run_shell_command for git.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "status|log|diff|add|commit|push|pull|branch|branch_create|branch_switch|branch_delete|branch_merge|stash|stash_list|stash_pop|stash_apply|stash_drop|tag_list|tag_create|tag_push|tag_delete|remote_list|remote_add|remote_remove|restore|reset_soft|conflict_detect|suggest|cherry_pick|rebase",
                    },
                    "files": {"type": "array", "description": "For add, restore: list of file paths"},
                    "message": {"type": "string", "description": "For commit, stash: message"},
                    "branch": {"type": "string", "description": "Branch name for branch_*, push, pull"},
                    "remote": {"type": "string", "description": "Remote name (default origin)"},
                    "limit": {"type": "integer", "description": "For log: max entries"},
                    "staged": {"type": "boolean", "description": "For diff: staged changes"},
                    "confirmed": {"type": "boolean", "description": "True to skip confirmation for write ops"},
                    "tag": {"type": "string", "description": "Tag name for tag_*"},
                    "name": {"type": "string", "description": "For remote_add: remote name; for branch: branch name"},
                    "url": {"type": "string", "description": "For remote_add: remote URL"},
                    "commit": {"type": "string", "description": "For cherry_pick: commit hash"},
                    "commits": {"type": "integer", "description": "For reset_soft: number of commits to undo"},
                    "index": {"type": "integer", "description": "For stash_pop/apply/drop: stash index"},
                    "force": {"type": "boolean", "description": "For branch_delete: force delete"},
                    "switch": {"type": "boolean", "description": "For branch_create: switch to new branch"},
                },
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    """Execute git operation by delegating to data_steward.git_operations."""
    if name != "git_operation":
        return f"Error: Unknown git tool '{name}'"

    from ...data_steward.git_operations import execute_git_operation

    result = execute_git_operation(args, project_root)

    if result.get("success"):
        output = result.get("data", {}).get("output", result.get("message", ""))
        return str(output) if output else result.get("message", "Done.")

    if result.get("requires_confirmation"):
        return (
            f"{result.get('message', 'Confirmation required')}. "
            "For write operations (add, commit, push, etc.), user must confirm. "
            "Reply with the confirmation request or use run_shell_command for read-only git."
        )

    return f"Error: {result.get('message', 'Git operation failed')}"
