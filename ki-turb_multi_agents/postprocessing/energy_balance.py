"""Independent physical-space and spectral HIT energy-balance calculations."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from schemas.hit_analysis_products import EnergyBalanceAssessment, ProductProvenance

from .periodic_derivatives import strain_rate_tensor


def turbulent_kinetic_energy(velocity: np.ndarray, *, remove_mean: bool = True) -> float:
    u = np.asarray(velocity, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("velocity must have shape (nx, ny, nz, 3)")
    if remove_mean:
        u = u - np.mean(u, axis=(0, 1, 2), keepdims=True)
    return 0.5 * float(np.mean(np.sum(u * u, axis=-1)))


def physical_dissipation(
    velocity: np.ndarray,
    viscosity: float,
    spacing: float | Sequence[float],
    *,
    derivative_method: str = "spectral",
) -> float:
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    strain = strain_rate_tensor(velocity, spacing, method=derivative_method)  # type: ignore[arg-type]
    sij_sij = np.einsum("...ij,...ij->...", strain, strain)
    return 2.0 * viscosity * float(np.mean(sij_sij))


def spectral_dissipation(wavenumber: Sequence[float], energy: Sequence[float], viscosity: float) -> float:
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    k = np.asarray(wavenumber, dtype=float)
    e = np.asarray(energy, dtype=float)
    if k.shape != e.shape or k.ndim != 1:
        raise ValueError("wavenumber and energy must be one-dimensional arrays of equal shape")
    return 2.0 * viscosity * float(np.trapz(k * k * e, k))


def forcing_power(velocity: np.ndarray, force: np.ndarray) -> float:
    u = np.asarray(velocity, dtype=float)
    f = np.asarray(force, dtype=float)
    if u.shape != f.shape or u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("velocity and force must have matching shape (nx, ny, nz, 3)")
    return float(np.mean(np.sum(u * f, axis=-1)))


def energy_balance_history(
    time: Sequence[float],
    tke: Sequence[float],
    dissipation_real: Sequence[float],
    *,
    forcing: Optional[Sequence[float]] = None,
    dissipation_spectral_values: Optional[Sequence[float]] = None,
    relative_error_limit: float = 0.15,
    provenance: Optional[ProductProvenance] = None,
) -> EnergyBalanceAssessment:
    time_array = np.asarray(time, dtype=float)
    tke_array = np.asarray(tke, dtype=float)
    diss_real = np.asarray(dissipation_real, dtype=float)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("at least two time samples are required")
    if tke_array.shape != time_array.shape or diss_real.shape != time_array.shape:
        raise ValueError("tke and dissipation histories must match time")
    power = np.zeros_like(time_array) if forcing is None else np.asarray(forcing, dtype=float)
    if power.shape != time_array.shape:
        raise ValueError("forcing power history must match time")
    diss_spectral = (
        np.full_like(time_array, np.nan)
        if dissipation_spectral_values is None
        else np.asarray(dissipation_spectral_values, dtype=float)
    )
    if diss_spectral.shape != time_array.shape:
        raise ValueError("spectral dissipation history must match time")

    tke_rate = np.gradient(tke_array, time_array, edge_order=1)
    residual = tke_rate - (power - diss_real)
    scale = np.maximum(np.abs(power) + np.abs(diss_real), 1.0e-15)
    relative = np.abs(residual) / scale
    finite = relative[np.isfinite(relative)]
    relative_mean = float(np.mean(finite)) if finite.size else None
    passed = relative_mean <= relative_error_limit if relative_mean is not None else None
    return EnergyBalanceAssessment(
        tke_rate=tke_rate.tolist(),
        forcing_power=power.tolist(),
        dissipation_real=diss_real.tolist(),
        dissipation_spectral=diss_spectral.tolist(),
        residual=residual.tolist(),
        relative_error_mean=relative_mean,
        passed=passed,
        provenance=provenance or ProductProvenance(algorithm="energy_balance_history"),
    )


__all__ = [
    "turbulent_kinetic_energy",
    "physical_dissipation",
    "spectral_dissipation",
    "forcing_power",
    "energy_balance_history",
]
