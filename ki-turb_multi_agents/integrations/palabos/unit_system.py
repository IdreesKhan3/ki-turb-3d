"""Palabos → UnitSystem adapter (stub).

When Palabos cases are wired, implement ``unit_system_from_palabos_case`` to fill
the same ``schemas.unit_system.UnitSystem`` contract used by OpenLB. Agents and
postprocessing must not grow a Palabos-specific unit schema.
"""
from __future__ import annotations

from schemas.unit_system import UnitSystem


def unit_system_from_palabos_case(case) -> UnitSystem:
    raise NotImplementedError(
        "Palabos UnitSystem adapter is not implemented yet; "
        "fill schemas.unit_system.UnitSystem from the Palabos converter."
    )


__all__ = ["unit_system_from_palabos_case"]
