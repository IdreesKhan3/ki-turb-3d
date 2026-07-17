"""Typed, solver-neutral OpenLB HIT control contract.

Every parameter advertised by KI-TURB is represented here.  The OpenLB adapter
must either consume a field exactly or reject the configuration explicitly.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 2


class OpenLBLattice(str, Enum):
    D3Q19 = "D3Q19"
    D3Q27 = "D3Q27"


class HITCollisionModel(str, Enum):
    BGK = "BGK"
    TRT = "TRT"
    MRT = "MRT"
    RLB = "RLB"
    SMAGORINSKY_BGK = "SmagorinskyBGK"
    SMAGORINSKY_MRT = "SmagorinskyMRT"
    WALE = "WALE"
    CONSISTENT_STRAIN_SMAGORINSKY = "ConsistentStrainSmagorinsky"
    SHEAR_SMAGORINSKY = "ShearSmagorinsky"
    KRAUSE = "Krause"
    DYNAMIC_SMAGORINSKY = "DynamicSmagorinsky"


class HITInitialConditionType(str, Enum):
    SYNTHETIC_SPECTRUM = "synthetic_spectrum"
    RESTART = "restart"
    IMPORTED_FIELD = "imported_field"


class HITForcingType(str, Enum):
    NONE = "none"
    SPECTRAL_RANDOM = "spectral_random"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"
    CONSTANT_ENERGY_INPUT = "constant_energy_input"
    CONSTANT_TKE = "constant_tke"


class BuildProfile(str, Enum):
    SERIAL = "serial"
    OPENMP = "openmp"
    MPI = "mpi"
    MPI_OPENMP = "mpi_openmp"
    CUDA = "cuda"
    HIP = "hip"
    DEBUG = "debug"
    SANITIZER = "sanitizer"


class ExecutionMode(str, Enum):
    LOCAL = "local"
    MPI = "mpi"
    DOCKER = "docker"
    SLURM = "slurm"
    SSH = "ssh"
    CLOUD = "cloud"


class HITDomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution: Tuple[int, int, int] = (64, 64, 64)
    size: Tuple[float, float, float] = (2.0 * math.pi,) * 3
    periodic: Tuple[bool, bool, bool] = (True, True, True)
    lattice: OpenLBLattice = OpenLBLattice.D3Q19

    @model_validator(mode="after")
    def _valid(self):
        if any(n < 4 for n in self.resolution):
            raise ValueError("each HIT resolution must be at least 4")
        if any(v <= 0 for v in self.size):
            raise ValueError("all domain lengths must be positive")
        if not all(self.periodic):
            raise ValueError("HIT requires periodicity in x, y and z")
        dx = tuple(l / n for l, n in zip(self.size, self.resolution))
        if max(dx) - min(dx) > 1e-12 * max(dx):
            raise ValueError("the current OpenLB HIT application requires uniform grid spacing")
        return self


class HITScalingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    density: float = 1.0
    characteristic_length: Optional[float] = None
    characteristic_velocity: Optional[float] = None
    physical_viscosity: Optional[float] = None
    reynolds_number: Optional[float] = None
    target_re_lambda: Optional[float] = None
    relaxation_time: Optional[float] = None
    target_mach: float = 0.05
    max_mach: float = 0.1
    lattice_sound_speed: float = 1.0 / math.sqrt(3.0)

    @model_validator(mode="after")
    def _valid(self):
        for name in ("density", "target_mach", "max_mach", "lattice_sound_speed"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("characteristic_length", "characteristic_velocity", "physical_viscosity", "reynolds_number", "target_re_lambda"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.relaxation_time is not None and self.relaxation_time <= 0.5:
            raise ValueError("relaxation_time must be greater than 0.5")
        if self.target_mach > self.max_mach:
            raise ValueError("target_mach cannot exceed max_mach")
        return self


class HITCollisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: HITCollisionModel = HITCollisionModel.BGK
    smagorinsky_constant: Optional[float] = None
    trt_magic_parameter: Optional[float] = None
    mrt_relaxation_rates: Optional[List[float]] = None

    @model_validator(mode="after")
    def _valid(self):
        if self.smagorinsky_constant is not None and self.smagorinsky_constant <= 0:
            raise ValueError("smagorinsky_constant must be positive")
        if self.trt_magic_parameter is not None and self.trt_magic_parameter <= 0:
            raise ValueError("trt_magic_parameter must be positive")
        if self.mrt_relaxation_rates and any(not 0 < value < 2 for value in self.mrt_relaxation_rates):
            raise ValueError("MRT relaxation rates must be in (0,2)")
        return self


class HITInitialConditionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: HITInitialConditionType = HITInitialConditionType.SYNTHETIC_SPECTRUM
    seed: Optional[int] = 12345
    wavenumber_min: Optional[int] = 1
    wavenumber_peak: Optional[float] = 4.0
    wavenumber_max: Optional[int] = 8
    spectrum_model: Optional[Literal["von_karman_pao", "gaussian_k4", "power_law"]] = "gaussian_k4"
    spectrum_exponent: Optional[float] = -2.0
    target_urms: Optional[float] = None
    source_file: Optional[str] = None
    forcing_state_file: Optional[str] = None
    verify_divergence_tolerance: float = 1e-8

    @model_validator(mode="after")
    def _valid(self):
        if self.type != HITInitialConditionType.SYNTHETIC_SPECTRUM and not self.source_file:
            raise ValueError(f"source_file is required for {self.type.value}")
        if self.type == HITInitialConditionType.SYNTHETIC_SPECTRUM:
            if self.wavenumber_min is None or self.wavenumber_max is None:
                raise ValueError("synthetic spectrum requires a wavenumber range")
            if self.wavenumber_min < 1 or self.wavenumber_max < self.wavenumber_min:
                raise ValueError("invalid initial-condition wavenumber range")
        if self.target_urms is not None and self.target_urms <= 0:
            raise ValueError("target_urms must be positive")
        return self


class HITForcingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: HITForcingType = HITForcingType.NONE
    wavenumber_min: Optional[int] = None
    wavenumber_max: Optional[int] = None
    amplitude: Optional[float] = None
    target_injection_rate: Optional[float] = None
    target_tke: Optional[float] = None
    correlation_time: Optional[float] = None
    controller_gain: float = 0.1
    update_interval: int = 1
    seed: Optional[int] = 23456
    remove_mean_force: bool = True
    solenoidal_projection: bool = True
    units: Literal["lattice_acceleration", "physical_acceleration"] = "lattice_acceleration"

    @model_validator(mode="after")
    def _valid(self):
        if self.update_interval <= 0:
            raise ValueError("forcing update_interval must be positive")
        if self.type in (HITForcingType.SPECTRAL_RANDOM, HITForcingType.ORNSTEIN_UHLENBECK):
            if self.wavenumber_min is None or self.wavenumber_max is None:
                raise ValueError("spectral forcing requires a wavenumber range")
            if self.wavenumber_min < 1 or self.wavenumber_max < self.wavenumber_min:
                raise ValueError("invalid forcing wavenumber range")
        for name in ("amplitude", "target_injection_rate", "target_tke", "correlation_time", "controller_gain"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.type == HITForcingType.ORNSTEIN_UHLENBECK and not self.correlation_time:
            raise ValueError("OU forcing requires correlation_time")
        if self.type == HITForcingType.CONSTANT_ENERGY_INPUT and not self.target_injection_rate:
            raise ValueError("constant-energy-input forcing requires target_injection_rate")
        if self.type == HITForcingType.CONSTANT_TKE and not self.target_tke:
            raise ValueError("constant-TKE forcing requires target_tke")
        return self


class HITRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = 10000
    output_interval: int = 1000
    diagnostics_interval: int = 100
    checkpoint_interval: Optional[int] = None
    sample_start_step: int = 0

    @model_validator(mode="after")
    def _valid(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.output_interval <= 0 or self.diagnostics_interval <= 0:
            raise ValueError("output and diagnostics intervals must be positive")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if not 0 <= self.sample_start_step <= self.max_steps:
            raise ValueError("sample_start_step must be in the run")
        return self


class HITExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: ExecutionMode = ExecutionMode.LOCAL
    build_profile: BuildProfile = BuildProfile.SERIAL
    num_procs: int = 1
    num_threads: int = 1
    memory_gb: Optional[float] = None
    walltime: Optional[str] = None
    queue: Optional[str] = None
    host: Optional[str] = None
    container_image: Optional[str] = None
    extra_environment: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid(self):
        if self.num_procs <= 0 or self.num_threads <= 0:
            raise ValueError("num_procs and num_threads must be positive")
        if self.memory_gb is not None and self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.mode == ExecutionMode.SSH and not self.host:
            raise ValueError("SSH execution requires host")
        if self.mode == ExecutionMode.DOCKER and not self.container_image:
            raise ValueError("Docker execution requires container_image")
        return self


class HITCheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    interval: Optional[int] = None
    directory: str = "checkpoints"
    retain: int = 2
    include_forcing_state: bool = True


class HITOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: str = "output"
    write_velocity: bool = True
    write_pressure: bool = False
    write_density: bool = True
    write_vorticity: bool = True
    write_forcing: bool = False
    write_populations: bool = False
    format: Literal["vtm", "vti", "hdf5", "xdmf_hdf5"] = "vtm"
    compression: Optional[str] = None


class HITAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_spectra: bool = True
    spectral_isotropy: bool = True
    reynolds_stress: bool = True
    anisotropy_invariants: bool = True
    energy_balance: bool = True
    structure_functions: bool = True
    pdfs: bool = True
    flatness: bool = True
    stationarity: bool = True
    uncertainty: bool = True


class HITVisualizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_histories: bool = True
    spectra: bool = True
    isotropy: bool = True
    stresses: bool = True
    lumley: bool = True
    pdfs: bool = True
    structure_functions: bool = True
    flatness: bool = True
    volume_fields: bool = True
    formats: List[Literal["html", "png", "svg"]] = Field(default_factory=lambda: ["html", "png"])


class HITAcceptanceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_mach: float = 0.1
    minimum_tau_margin: float = 0.005
    minimum_kmax_eta: float = 1.0
    maximum_mass_drift_fraction: float = 1e-6
    maximum_density_deviation: float = 0.05
    maximum_divergence_rms: float = 1e-6
    maximum_component_energy_deviation: float = 0.1
    maximum_energy_balance_relative_error: float = 0.15
    stationary_cv_limit: float = 0.05


class HITDerivedScaling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dx: float
    dt: float
    lattice_viscosity: float
    lattice_velocity: float
    actual_mach: float
    reynolds_number: float
    relaxation_time: float
    characteristic_length: float
    characteristic_velocity: float
    physical_viscosity: float


class HITEffectiveOpenLBConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    openlb_version: Optional[str] = None
    openlb_commit: Optional[str] = None
    lattice: Optional[str] = None
    descriptor: Optional[str] = None
    collision: Optional[str] = None
    dynamics_class: Optional[str] = None
    forcing: Optional[str] = None
    build_profile: Optional[str] = None
    executable_sha256: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class HITMeasuredQuantities(BaseModel):
    model_config = ConfigDict(extra="allow")
    mass: Optional[float] = None
    density_min: Optional[float] = None
    density_max: Optional[float] = None
    mach_max: Optional[float] = None
    tke: Optional[float] = None
    dissipation: Optional[float] = None
    forcing_power: Optional[float] = None
    divergence_rms: Optional[float] = None
    re_lambda: Optional[float] = None
    kmax_eta: Optional[float] = None


class OpenLBHITConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION
    name: str = "openlb_hit"
    description: Optional[str] = None
    openlb_version: Optional[str] = None
    domain: HITDomainConfig = Field(default_factory=HITDomainConfig)
    scaling: HITScalingConfig = Field(default_factory=HITScalingConfig)
    collision: HITCollisionConfig = Field(default_factory=HITCollisionConfig)
    initial_condition: HITInitialConditionConfig = Field(default_factory=HITInitialConditionConfig)
    forcing: HITForcingConfig = Field(default_factory=HITForcingConfig)
    runtime: HITRuntimeConfig = Field(default_factory=HITRuntimeConfig)
    execution: HITExecutionConfig = Field(default_factory=HITExecutionConfig)
    checkpoint: HITCheckpointConfig = Field(default_factory=HITCheckpointConfig)
    outputs: HITOutputConfig = Field(default_factory=HITOutputConfig)
    analysis: HITAnalysisRequest = Field(default_factory=HITAnalysisRequest)
    visualization: HITVisualizationRequest = Field(default_factory=HITVisualizationRequest)
    acceptance: HITAcceptanceThresholds = Field(default_factory=HITAcceptanceThresholds)
    effective: Optional[HITEffectiveOpenLBConfig] = None
    measured: Optional[HITMeasuredQuantities] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def derive_scaling(self, *, relative_tolerance: float = 1e-6) -> HITDerivedScaling:
        nx, _, _ = self.domain.resolution
        lx, _, _ = self.domain.size
        s = self.scaling
        length = s.characteristic_length or lx
        velocity, viscosity, reynolds = s.characteristic_velocity, s.physical_viscosity, s.reynolds_number
        if sum(v is not None for v in (velocity, viscosity, reynolds)) < 2:
            raise ValueError("supply at least two of characteristic_velocity, physical_viscosity and reynolds_number")
        if velocity is None:
            velocity = float(reynolds) * float(viscosity) / length
        if viscosity is None:
            viscosity = velocity * length / float(reynolds)
        calculated_re = velocity * length / viscosity
        if reynolds is not None and not math.isclose(calculated_re, reynolds, rel_tol=relative_tolerance, abs_tol=relative_tolerance):
            raise ValueError(f"inconsistent Reynolds inputs: supplied {reynolds:g}, calculated {calculated_re:g}")
        dx = lx / nx
        if s.relaxation_time is None:
            lattice_velocity = s.target_mach * s.lattice_sound_speed
            dt = lattice_velocity * dx / velocity
            lattice_viscosity = viscosity * dt / dx**2
            tau = 0.5 + 3.0 * lattice_viscosity
        else:
            tau = s.relaxation_time
            lattice_viscosity = (tau - 0.5) / 3.0
            dt = lattice_viscosity * dx**2 / viscosity
            lattice_velocity = velocity * dt / dx
        actual_mach = lattice_velocity / s.lattice_sound_speed
        return HITDerivedScaling(dx=dx, dt=dt, lattice_viscosity=lattice_viscosity,
            lattice_velocity=lattice_velocity, actual_mach=actual_mach,
            reynolds_number=calculated_re, relaxation_time=tau,
            characteristic_length=length, characteristic_velocity=velocity,
            physical_viscosity=viscosity)

    @classmethod
    def from_cfd_case(cls, case: Any) -> "OpenLBHITConfig":
        if getattr(case, "hit", None) is not None:
            hit = case.hit
            return hit if isinstance(hit, cls) else cls.model_validate(hit)
        ex = dict(getattr(case.solver, "extra", {}) or {})
        fex = dict(getattr(case.flow, "extra", {}) or {})
        oex = dict(getattr(case.outputs, "extra", {}) or {})
        rex = dict(getattr(case.runtime, "extra", {}) or {})
        eex = dict(getattr(case.execution, "extra", {}) or {})
        forcing_alias = {"off":"none", "spectral_low_k":"spectral_random", "low_wavenumber":"spectral_random", "ou":"ornstein_uhlenbeck", "linear":"constant_energy_input"}
        forcing = forcing_alias.get((case.flow.forcing_type or "none").lower(), (case.flow.forcing_type or "none").lower())
        collision_alias = {"smagorinsky":"SmagorinskyBGK", "smagorinskimrt":"SmagorinskyMRT", "dynsmagorinsky":"DynamicSmagorinsky", "dns":"BGK"}
        collision = collision_alias.get((case.solver.scheme or "BGK").replace("_", "").lower(), case.solver.scheme or "BGK")
        initial_alias = {"divergence_free_random":"synthetic_spectrum", "synthetic":"synthetic_spectrum"}
        initial = initial_alias.get((case.flow.initial_condition or "synthetic_spectrum").lower(), (case.flow.initial_condition or "synthetic_spectrum").lower())

        # Legacy CFDCase objects allow all scaling and spectral controls to be omitted.
        # Convert those omissions into one documented, internally consistent HIT case
        # instead of passing None into the strict typed schema. Explicit user values
        # always win.
        nmin = min(int(value) for value in case.mesh.resolution)
        resolvable_kmax = max(1, nmin // 2 - 1)
        default_ic_kmax = min(8, resolvable_kmax)
        ic_kmin = case.flow.ic_wavenumber_min if case.flow.ic_wavenumber_min is not None else 1
        ic_kmax = case.flow.ic_wavenumber_max if case.flow.ic_wavenumber_max is not None else max(ic_kmin, default_ic_kmax)
        force_kmin = case.flow.forcing_wavenumber_min
        force_kmax = case.flow.forcing_wavenumber_max
        if forcing in {"spectral_random", "ornstein_uhlenbeck"}:
            force_kmin = 1 if force_kmin is None else force_kmin
            force_kmax = min(2, resolvable_kmax) if force_kmax is None else force_kmax

        supplied_velocity = ex.get("char_velocity")
        supplied_viscosity = case.solver.viscosity
        supplied_reynolds = case.solver.reynolds_number
        if sum(value is not None for value in (supplied_velocity, supplied_viscosity, supplied_reynolds)) < 2:
            if supplied_velocity is None:
                supplied_velocity = 0.1
            if supplied_viscosity is None and supplied_reynolds is None:
                supplied_reynolds = 100.0

        return cls(
            name=case.name, description=case.description,
            domain={"resolution": case.mesh.resolution, "size": case.geometry.size,
                    "lattice": ex.get("lattice", "D3Q19")},
            scaling={"density": ex.get("density", 1.0), "characteristic_length": case.geometry.size[0],
                     "characteristic_velocity": supplied_velocity, "physical_viscosity": supplied_viscosity,
                     "reynolds_number": supplied_reynolds, "target_re_lambda": case.flow.target_re_lambda,
                     "relaxation_time": ex.get("relaxation_time"), "target_mach": ex.get("mach_number", 0.05),
                     "max_mach": ex.get("max_mach", 0.1)},
            collision={"model": collision, "smagorinsky_constant": ex.get("smagorinsky_constant"),
                       "trt_magic_parameter": ex.get("trt_magic_parameter")},
            initial_condition={"type": initial, "seed": case.flow.ic_seed if case.flow.ic_seed is not None else 12345,
                               "wavenumber_min": ic_kmin,
                               "wavenumber_max": ic_kmax,
                               "spectrum_exponent": case.flow.ic_spectrum_exponent,
                               "target_urms": case.flow.target_urms, "source_file": fex.get("source_file")},
            forcing={"type": forcing, "wavenumber_min": force_kmin,
                     "wavenumber_max": force_kmax, "amplitude": case.flow.forcing_amplitude,
                     "target_injection_rate": fex.get("target_injection_rate") or (case.flow.forcing_amplitude if forcing == "constant_energy_input" else None),
                     "target_tke": fex.get("target_tke"), "correlation_time": fex.get("correlation_time"),
                     "update_interval": case.flow.forcing_update_interval or 1, "seed": fex.get("forcing_seed", 23456)},
            runtime={"max_steps": case.runtime.max_steps or 10000, "output_interval": case.runtime.output_interval,
                     "diagnostics_interval": rex.get("diagnostics_interval", 100),
                     "checkpoint_interval": case.runtime.checkpoint_interval,
                     "sample_start_step": case.outputs.sample_start_step or 0},
            execution={"mode": case.execution.mode, "build_profile": eex.get("build_profile", "serial"),
                       "num_procs": case.execution.num_procs, "num_threads": case.execution.num_threads,
                       "memory_gb": case.execution.memory_gb, "walltime": case.execution.walltime,
                       "queue": case.execution.queue, "host": eex.get("host"),
                       "container_image": case.execution.container_image,
                       "extra_environment": eex.get("environment", {})},
            checkpoint={"enabled": bool(case.runtime.checkpoint_interval), "interval": case.runtime.checkpoint_interval},
            outputs={"write_velocity": case.outputs.write_velocity, "write_pressure": case.outputs.write_pressure,
                     "write_density": oex.get("write_density", True), "write_vorticity": case.outputs.write_vorticity,
                     "write_forcing": oex.get("write_forcing", False), "format": oex.get("format", "vtm")},
            analysis={"energy_spectra": case.outputs.write_spectra, "spectral_isotropy": case.outputs.write_isotropy,
                      "flatness": case.outputs.write_flatness, "structure_functions": case.outputs.write_structure_functions,
                      "pdfs": case.outputs.write_pdfs}, metadata=dict(case.metadata or {}))


__all__ = [name for name in globals() if name.startswith("HIT") or name in {"OpenLBHITConfig", "OpenLBLattice", "BuildProfile", "ExecutionMode", "SCHEMA_VERSION"}]
