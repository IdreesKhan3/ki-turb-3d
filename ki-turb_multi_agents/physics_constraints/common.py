"""Constraint checks shared across all CFD cases."""

from __future__ import annotations

from typing import Sequence

from schemas import ConstraintCheck


def check_positive(name: str, value) -> ConstraintCheck:
    return ConstraintCheck(
        name=f"{name}_positive",
        passed=value is not None and value > 0,
        severity="error",
        message=f"{name} must be positive.",
        value=value,
    )


def check_resolution(resolution: Sequence[int]) -> ConstraintCheck:
    nx, ny, nz = resolution
    return ConstraintCheck(
        name="resolution_positive",
        passed=nx > 0 and ny > 0 and nz > 0,
        severity="error",
        message="Mesh resolution must be positive in all directions.",
        value=tuple(resolution),
    )
