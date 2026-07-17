"""Minimal framework-independent role identifiers used by the tool registry."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    STEWARD = "steward"
    ANALYST = "analyst"
    VISUALIZER = "visualizer"
    REVIEWER = "reviewer"
    SIMULATION = "simulation"
    ENGINEER = "engineer"

    @classmethod
    def coerce(cls, value: Any) -> Optional["AgentName"]:
        if isinstance(value, cls):
            return value
        if not value:
            return None
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return None


__all__ = ["AgentName"]
