"""Restrict shell execution to an allowlist of commands.

Validation proceeds in three steps:

    1. Reject shell metacharacters that enable chaining, piping, redirection, or
       substitution.
    2. Tokenize the command with ``shlex``; reject if it cannot be tokenized.
    3. Require the program (first token) to be in an explicit allowlist.

Validated commands are returned as an ``argv`` list for execution with
``shell=False``.
"""

from __future__ import annotations

import shlex
from typing import List, Tuple

# Read-only inspection and version-control commands. Interpreters and package
# managers are excluded because they permit arbitrary code execution.
DEFAULT_ALLOWED_COMMANDS = frozenset({
    "git",
    "ls", "cat", "head", "tail", "wc", "pwd", "echo",
    "grep", "rg", "find", "which", "tree", "stat", "file",
    "du", "df", "date", "whoami", "uname", "env",
    "diff", "sort", "uniq", "cut", "basename", "dirname",
})

# Any of these in the raw command means "trying to do more than one thing" or
# "invoke the shell" — reject before tokenizing.
_FORBIDDEN_SUBSTRINGS = (
    ";", "|", "&", ">", "<", "`", "$(", "${", "\n", "\r",
    "&&", "||", ">>", "<<",
)


class ShellPolicyError(ValueError):
    """Raised when a shell command violates the allowlist policy."""


def is_command_allowed(cmd: str, allowed=DEFAULT_ALLOWED_COMMANDS) -> Tuple[bool, str]:
    """Return (allowed, reason). Reason is empty when allowed."""
    if not cmd or not cmd.strip():
        return False, "empty command"

    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in cmd:
            return False, f"forbidden shell operator: {bad!r}"

    try:
        tokens = shlex.split(cmd)
    except ValueError as e:
        return False, f"could not parse command: {e}"

    if not tokens:
        return False, "no command found"

    program = tokens[0]
    # Reject path-qualified programs (e.g. ./run, /usr/bin/python) — only bare
    # allowlisted names are permitted so intent is unambiguous.
    if "/" in program or "\\" in program:
        return False, f"path-qualified programs are not allowed: {program!r}"

    if program not in allowed:
        return False, f"command not in allowlist: {program!r}"

    return True, ""


def to_argv(cmd: str, allowed=DEFAULT_ALLOWED_COMMANDS) -> List[str]:
    """Validate ``cmd`` and return its argv list (for subprocess with shell=False).

    Raises :class:`ShellPolicyError` if the command is not allowed.
    """
    ok, reason = is_command_allowed(cmd, allowed=allowed)
    if not ok:
        raise ShellPolicyError(reason)
    return shlex.split(cmd)


def check_command(cmd: str, allowed=DEFAULT_ALLOWED_COMMANDS) -> None:
    """Raise :class:`ShellPolicyError` if ``cmd`` is not allowed; else return None."""
    ok, reason = is_command_allowed(cmd, allowed=allowed)
    if not ok:
        raise ShellPolicyError(reason)
