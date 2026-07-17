"""Security policies for agent actions.

Modules:
    path_policy      Restrict file writes to allowlisted directories.
    shell_policy     Restrict shell execution to an allowlist of commands.
    approval_policy  Determine which actions require user confirmation.

These modules contain pure policy logic with no side effects.
"""

from .path_policy import (
    PathPolicy,
    PathPolicyError,
    default_policy,
)
from .shell_policy import (
    ShellPolicyError,
    check_command,
    is_command_allowed,
    to_argv,
    DEFAULT_ALLOWED_COMMANDS,
)
from .approval_policy import (
    requires_confirmation,
    confirmation_message,
)

__all__ = [
    "PathPolicy",
    "PathPolicyError",
    "default_policy",
    "ShellPolicyError",
    "check_command",
    "is_command_allowed",
    "to_argv",
    "DEFAULT_ALLOWED_COMMANDS",
    "requires_confirmation",
    "confirmation_message",
]

from .openlb_permissions import OpenLBPermission, ROLE_PERMISSIONS, has_openlb_permission, require_openlb_permission
