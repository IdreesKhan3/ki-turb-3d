"""Solver-neutral analysis product loading for KI-TURB agents and pages."""

from .manifest_index import MANIFEST_KIND_TO_SESSION_KEY, SESSION_KEY_ALIASES
from .product_loader import AnalysisProductLoader

__all__ = ["AnalysisProductLoader", "MANIFEST_KIND_TO_SESSION_KEY", "SESSION_KEY_ALIASES"]
