"""Periodic finite-difference and spectral derivatives for HIT fields."""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _spacing_tuple(spacing: float | Sequence[float]) -> Tuple[float, float, float]:
    if np.isscalar(spacing):
        value = float(spacing)
        if value <= 0:
            raise ValueError("spacing must be positive")
        return value, value, value
    values = tuple(float(v) for v in spacing)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError("spacing must contain three positive values")
    return values  # type: ignore[return-value]


def periodic_central_difference(field: Array, axis: int, spacing: float) -> Array:
    """Second-order central derivative with periodic wrapping."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)


def spectral_derivative(field: Array, axis: int, spacing: float) -> Array:
    """Fourier derivative along one periodic axis."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    n = field.shape[axis]
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
    shape = [1] * field.ndim
    shape[axis] = n
    multiplier = (1j * wave_number).reshape(shape)
    transformed = np.fft.fft(field, axis=axis)
    derivative = np.fft.ifft(transformed * multiplier, axis=axis)
    return derivative.real if np.isrealobj(field) else derivative


def gradient_periodic(
    field: Array,
    spacing: float | Sequence[float] = 1.0,
    *,
    method: Literal["spectral", "central"] = "spectral",
) -> Tuple[Array, Array, Array]:
    dx = _spacing_tuple(spacing)
    derivative = spectral_derivative if method == "spectral" else periodic_central_difference
    return tuple(derivative(field, axis=i, spacing=dx[i]) for i in range(3))  # type: ignore[return-value]


def velocity_gradient_tensor(
    velocity: Array,
    spacing: float | Sequence[float] = 1.0,
    *,
    method: Literal["spectral", "central"] = "spectral",
) -> Array:
    """Return ``du_i/dx_j`` with output shape ``(..., 3, 3)``."""
    velocity = np.asarray(velocity)
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError("velocity must have shape (nx, ny, nz, 3)")
    grad = np.empty(velocity.shape[:-1] + (3, 3), dtype=np.float64)
    for component in range(3):
        derivatives = gradient_periodic(velocity[..., component], spacing, method=method)
        for axis in range(3):
            grad[..., component, axis] = derivatives[axis]
    return grad


def strain_rate_tensor(
    velocity: Array,
    spacing: float | Sequence[float] = 1.0,
    *,
    method: Literal["spectral", "central"] = "spectral",
) -> Array:
    grad = velocity_gradient_tensor(velocity, spacing, method=method)
    return 0.5 * (grad + np.swapaxes(grad, -1, -2))


def rotation_rate_tensor(
    velocity: Array,
    spacing: float | Sequence[float] = 1.0,
    *,
    method: Literal["spectral", "central"] = "spectral",
) -> Array:
    grad = velocity_gradient_tensor(velocity, spacing, method=method)
    return 0.5 * (grad - np.swapaxes(grad, -1, -2))


def divergence(
    velocity: Array,
    spacing: float | Sequence[float] = 1.0,
    *,
    method: Literal["spectral", "central"] = "spectral",
) -> Array:
    grad = velocity_gradient_tensor(velocity, spacing, method=method)
    return np.trace(grad, axis1=-2, axis2=-1)


def vorticity(
    velocity: Array,
    spacing: float | Sequence[float] = 1.0,
    *,
    method: Literal["spectral", "central"] = "spectral",
) -> Array:
    grad = velocity_gradient_tensor(velocity, spacing, method=method)
    omega = np.empty_like(velocity, dtype=np.float64)
    omega[..., 0] = grad[..., 2, 1] - grad[..., 1, 2]
    omega[..., 1] = grad[..., 0, 2] - grad[..., 2, 0]
    omega[..., 2] = grad[..., 1, 0] - grad[..., 0, 1]
    return omega


def q_criterion(
    velocity: Array,
    spacing: float | Sequence[float] = 1.0,
    *,
    method: Literal["spectral", "central"] = "spectral",
) -> Array:
    strain = strain_rate_tensor(velocity, spacing, method=method)
    rotation = rotation_rate_tensor(velocity, spacing, method=method)
    strain_norm_sq = np.einsum("...ij,...ij->...", strain, strain)
    rotation_norm_sq = np.einsum("...ij,...ij->...", rotation, rotation)
    return 0.5 * (rotation_norm_sq - strain_norm_sq)


__all__ = [
    "periodic_central_difference",
    "spectral_derivative",
    "gradient_periodic",
    "velocity_gradient_tensor",
    "strain_rate_tensor",
    "rotation_rate_tensor",
    "divergence",
    "vorticity",
    "q_criterion",
]

# Backward-compatible public name used by older KI-TURB analysis modules.
velocity_gradient = velocity_gradient_tensor
if "velocity_gradient" not in __all__:
    __all__.append("velocity_gradient")
