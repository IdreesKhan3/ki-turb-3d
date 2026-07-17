"""Curated CFD case factories.

Factories build physics-validated :class:`~schemas.cfd_case.CFDCase` objects so
the agent starts from a sound configuration instead of inventing parameters.
"""

from .registry import make_case

__all__ = ["make_case"]
