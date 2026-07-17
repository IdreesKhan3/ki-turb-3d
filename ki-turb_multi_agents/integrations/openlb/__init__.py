"""OpenLB-specific services used by KI-TURB's solver-neutral workflow.

Imports are deliberately lightweight so configuration and capability tests do
not pull in the Streamlit-facing ``agents`` package.
"""

from .capability_validator import (
    CapabilityStatus,
    OpenLBHITCapabilities,
    OpenLBHITCapabilityValidator,
    UnsupportedOpenLBCapability,
)
from .config_translator import OpenLBHITConfigTranslator
from .provenance import OpenLBProvenanceCollector, ProvenanceRecord
from .unit_system import unit_system_from_openlb_hit

__all__ = [
    "CapabilityStatus",
    "OpenLBHITCapabilities",
    "OpenLBHITCapabilityValidator",
    "UnsupportedOpenLBCapability",
    "OpenLBHITConfigTranslator",
    "OpenLBProvenanceCollector",
    "ProvenanceRecord",
    "unit_system_from_openlb_hit",
]
