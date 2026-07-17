"""Physics-constraint validators for CFD cases.

Each validator inspects a :class:`~schemas.cfd_case.CFDCase` and returns a
:class:`~schemas.validation_report.ValidationReport`. :func:`validate_case`
combines the validators that apply to a given case so invalid simulations are
rejected before any solver runs.
"""

from .registry import validate_case

__all__ = ["validate_case"]
