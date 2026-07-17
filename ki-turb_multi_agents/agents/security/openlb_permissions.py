"""Explicit OpenLB HIT capabilities assigned to agent roles."""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class OpenLBPermission(str, Enum):
    READ_SOURCE = "OPENLB_READ_SOURCE"
    WRITE_CASE = "OPENLB_WRITE_CASE"
    MODIFY_HIT_APP = "OPENLB_MODIFY_HIT_APP"
    CONFIGURE_BUILD = "OPENLB_CONFIGURE_BUILD"
    COMPILE = "OPENLB_COMPILE"
    RUN = "OPENLB_RUN"
    CANCEL = "OPENLB_CANCEL"
    CHECKPOINT = "OPENLB_CHECKPOINT"
    FETCH_DATA = "OPENLB_FETCH_DATA"
    ANALYSE = "OPENLB_ANALYSE"
    VISUALIZE = "OPENLB_VISUALIZE"
    REVIEW = "OPENLB_REVIEW"


ROLE_PERMISSIONS: Dict[str, FrozenSet[OpenLBPermission]] = {
    "orchestrator": frozenset(),
    "physics": frozenset({OpenLBPermission.READ_SOURCE}),
    "simulation": frozenset({
        OpenLBPermission.READ_SOURCE,
        OpenLBPermission.WRITE_CASE,
        OpenLBPermission.MODIFY_HIT_APP,
        OpenLBPermission.CONFIGURE_BUILD,
        OpenLBPermission.COMPILE,
        OpenLBPermission.RUN,
        OpenLBPermission.CANCEL,
        OpenLBPermission.CHECKPOINT,
        OpenLBPermission.FETCH_DATA,
    }),
    "steward": frozenset({OpenLBPermission.FETCH_DATA}),
    "analyst": frozenset({OpenLBPermission.ANALYSE}),
    "visualizer": frozenset({OpenLBPermission.VISUALIZE}),
    "reviewer": frozenset({OpenLBPermission.REVIEW}),
}


def has_openlb_permission(role: str, permission: OpenLBPermission | str) -> bool:
    try:
        value = permission if isinstance(permission, OpenLBPermission) else OpenLBPermission(permission)
    except ValueError:
        return False
    return value in ROLE_PERMISSIONS.get(role.lower(), frozenset())


def require_openlb_permission(role: str, permission: OpenLBPermission | str) -> None:
    value = permission if isinstance(permission, OpenLBPermission) else OpenLBPermission(permission)
    if not has_openlb_permission(role, value):
        raise PermissionError(f"agent role '{role}' lacks {value.value}")


__all__ = ["OpenLBPermission", "ROLE_PERMISSIONS", "has_openlb_permission", "require_openlb_permission"]
