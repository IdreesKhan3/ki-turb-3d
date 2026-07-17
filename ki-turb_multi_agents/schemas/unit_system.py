"""Solver-neutral unit contract for CFD cases, manifests, and analysis products.

Backends (OpenLB, Palabos, …) fill this schema; agents and postprocessing consume
it without importing solver-specific converters.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UnitFrame(str, Enum):
    """Coordinate frame of a stored field or scalar."""

    PHYSICAL = "physical"
    LATTICE = "lattice"
    DIMENSIONLESS = "dimensionless"
    UNKNOWN = "unknown"


class FieldKind(str, Enum):
    VELOCITY = "velocity_field"
    PRESSURE = "pressure_field"
    DENSITY = "density_field"
    VORTICITY = "vorticity_field"
    FORCING = "forcing_field"
    TIME = "time"
    LENGTH = "length"
    VISCOSITY = "viscosity"


class FieldUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    frame: UnitFrame = UnitFrame.UNKNOWN
    # Human / machine label, e.g. physical_velocity, lattice_density
    label: str = ""
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _default_label(self):
        if not self.label:
            frame = self.frame.value if isinstance(self.frame, UnitFrame) else str(self.frame)
            self.label = f"{frame}_{self.kind.removesuffix('_field')}"
        return self


class UnitSystem(BaseModel):
    """Authoritative unit bridge attached to a case / dataset / analysis product."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    # Which backend filled this contract (openlb, palabos, …) — not the schema name.
    source_backend: Optional[str] = None
    # Preferred frame for analysis (spectra, stats, …).
    analysis_frame: UnitFrame = UnitFrame.PHYSICAL

    length_ref: Optional[float] = None  # characteristic length (physical)
    velocity_ref: Optional[float] = None  # characteristic velocity (physical)
    density_ref: Optional[float] = None
    viscosity: Optional[float] = None  # kinematic viscosity in analysis_frame
    reynolds_number: Optional[float] = None

    dx: Optional[float] = None  # grid spacing (physical if analysis_frame=physical)
    dt: Optional[float] = None  # time step (physical if analysis_frame=physical)
    lattice_sound_speed: Optional[float] = None
    relaxation_time: Optional[float] = None  # lattice τ when applicable
    mach: Optional[float] = None  # lattice Mach when applicable

    fields: Dict[str, FieldUnit] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def field_labels(self) -> Dict[str, str]:
        """Legacy flat map used by older collectors / provenance."""
        return {key: fu.label for key, fu in self.fields.items()}

    def require_analysis_ready(self) -> None:
        if self.dx is None or float(self.dx) <= 0:
            raise ValueError("UnitSystem.dx must be positive for analysis")
        if self.viscosity is None or float(self.viscosity) <= 0:
            raise ValueError("UnitSystem.viscosity must be positive for analysis")
        if self.analysis_frame == UnitFrame.UNKNOWN:
            raise ValueError("UnitSystem.analysis_frame must be set for analysis")

    @classmethod
    def from_legacy_labels(
        cls,
        labels: Dict[str, str],
        *,
        source_backend: Optional[str] = None,
        dx: Optional[float] = None,
        viscosity: Optional[float] = None,
    ) -> "UnitSystem":
        fields: Dict[str, FieldUnit] = {}
        for kind, label in labels.items():
            frame = UnitFrame.UNKNOWN
            low = str(label).lower()
            if "physical" in low:
                frame = UnitFrame.PHYSICAL
            elif "lattice" in low:
                frame = UnitFrame.LATTICE
            fields[kind] = FieldUnit(kind=kind, frame=frame, label=str(label))
        return cls(
            source_backend=source_backend,
            analysis_frame=UnitFrame.PHYSICAL,
            dx=dx,
            viscosity=viscosity,
            fields=fields,
        )


__all__ = ["UnitFrame", "FieldKind", "FieldUnit", "UnitSystem"]
