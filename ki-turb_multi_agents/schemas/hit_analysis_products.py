"""Canonical, solver-neutral products of a HIT analysis workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .unit_system import UnitSystem


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProductProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: Optional[str] = None
    source_steps: List[int] = Field(default_factory=list)
    source_files: List[str] = Field(default_factory=list)
    algorithm: str = ""
    algorithm_version: str = "1"
    normalization: Optional[str] = None
    units: Dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StationarityAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")

    stationary: bool = False
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tke_cv: Optional[float] = None
    dissipation_cv: Optional[float] = None
    power_cv: Optional[float] = None
    normalized_tke_slope: Optional[float] = None
    reason: str = ""


class ResolutionAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")

    passed: bool = False
    kmax_eta_mean: Optional[float] = None
    kmax_eta_min: Optional[float] = None
    required_minimum: float = 1.0
    samples: int = 0
    reason: str = ""


class EnergyBalanceAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")

    tke_rate: List[float] = Field(default_factory=list)
    forcing_power: List[float] = Field(default_factory=list)
    dissipation_real: List[float] = Field(default_factory=list)
    dissipation_spectral: List[float] = Field(default_factory=list)
    residual: List[float] = Field(default_factory=list)
    relative_error_mean: Optional[float] = None
    passed: Optional[bool] = None
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class EnergySpectrumProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    time: Optional[float] = None
    wavenumber: List[float]
    energy: List[float]
    compensated_energy: Optional[List[float]] = None
    k_eta: Optional[List[float]] = None
    inertial_slope: Optional[float] = None
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class SpectralIsotropyProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    time: Optional[float] = None
    wavenumber: List[float]
    e11: List[float]
    e22: List[float]
    e33: List[float]
    maximum_component_deviation: Optional[float] = None
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class ReynoldsStressProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    time: Optional[float] = None
    r11: float
    r22: float
    r33: float
    r12: float
    r13: float
    r23: float
    b11: Optional[float] = None
    b22: Optional[float] = None
    b33: Optional[float] = None
    b12: Optional[float] = None
    b13: Optional[float] = None
    b23: Optional[float] = None
    invariant_ii: Optional[float] = None
    invariant_iii: Optional[float] = None
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class TimeHistoryProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    time: List[float] = Field(default_factory=list)
    step: List[int] = Field(default_factory=list)
    tke: List[float] = Field(default_factory=list)
    dissipation: List[float] = Field(default_factory=list)
    forcing_power: List[float] = Field(default_factory=list)
    re_lambda: List[float] = Field(default_factory=list)
    mach_max: List[float] = Field(default_factory=list)
    density_min: List[float] = Field(default_factory=list)
    density_max: List[float] = Field(default_factory=list)
    divergence_rms: List[float] = Field(default_factory=list)
    kmax_eta: List[float] = Field(default_factory=list)
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class StructureFunctionProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    time: Optional[float] = None
    separation: List[float]
    orders: List[int]
    longitudinal: Dict[str, List[float]] = Field(default_factory=dict)
    transverse: Dict[str, List[float]] = Field(default_factory=dict)
    signed_longitudinal_third: Optional[List[float]] = None
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class PDFProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    time: Optional[float] = None
    variable: str
    bin_center: List[float]
    density: List[float]
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class FlatnessProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    time: Optional[float] = None
    separation: List[float]
    flatness: List[float]
    provenance: ProductProvenance = Field(default_factory=ProductProvenance)


class StatisticalSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    metric: str
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_level: float = 0.95
    confidence_low: float
    confidence_high: float
    effective_sample_size: float
    sample_count: int


class HITAnalysisProducts(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    run_id: Optional[str] = None
    time_history: Optional[TimeHistoryProduct] = None
    spectra: List[EnergySpectrumProduct] = Field(default_factory=list)
    spectral_isotropy: List[SpectralIsotropyProduct] = Field(default_factory=list)
    reynolds_stress: List[ReynoldsStressProduct] = Field(default_factory=list)
    structure_functions: List[StructureFunctionProduct] = Field(default_factory=list)
    pdfs: List[PDFProduct] = Field(default_factory=list)
    flatness: List[FlatnessProduct] = Field(default_factory=list)
    stationarity: Optional[StationarityAssessment] = None
    resolution: Optional[ResolutionAssessment] = None
    energy_balance: Optional[EnergyBalanceAssessment] = None
    uncertainty: List[StatisticalSummary] = Field(default_factory=list)
    validation_status: str = "unvalidated"
    warnings: List[str] = Field(default_factory=list)
    unit_system: Optional[UnitSystem] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "HITAnalysisProducts":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ProductProvenance",
    "StationarityAssessment",
    "ResolutionAssessment",
    "EnergyBalanceAssessment",
    "EnergySpectrumProduct",
    "SpectralIsotropyProduct",
    "ReynoldsStressProduct",
    "TimeHistoryProduct",
    "StructureFunctionProduct",
    "PDFProduct",
    "FlatnessProduct",
    "StatisticalSummary",
    "HITAnalysisProducts",
]
