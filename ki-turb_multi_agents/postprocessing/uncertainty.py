"""Uncertainty estimates for temporally sampled HIT statistics."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

from schemas.hit_analysis_products import StatisticalSummary


def integrated_autocorrelation_time(values: Sequence[float] | np.ndarray) -> float:
    """Estimate integrated autocorrelation time using the initial-positive sequence."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        return 1.0
    x = x - np.mean(x)
    variance = float(np.dot(x, x) / x.size)
    if variance <= 0:
        return 1.0
    correlation = np.correlate(x, x, mode="full")[x.size - 1 :]
    correlation /= variance * np.arange(x.size, 0, -1)
    tau = 1.0
    for lag in range(1, correlation.size):
        rho = float(correlation[lag])
        if not np.isfinite(rho) or rho <= 0:
            break
        tau += 2.0 * rho
    return max(1.0, tau)


def summarize_uncertainty(
    values: Sequence[float] | np.ndarray,
    *,
    metric: str,
    confidence_level: float = 0.95,
) -> StatisticalSummary:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("cannot summarize an empty or non-finite sample")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    tau = integrated_autocorrelation_time(x)
    effective_n = max(1.0, x.size / tau)
    standard_error = std / math.sqrt(effective_n)
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    half_width = z * standard_error
    return StatisticalSummary(
        metric=metric,
        mean=mean,
        standard_deviation=std,
        standard_error=standard_error,
        confidence_level=confidence_level,
        confidence_low=mean - half_width,
        confidence_high=mean + half_width,
        effective_sample_size=effective_n,
        sample_count=int(x.size),
    )


def block_means(values: Sequence[float] | np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    block_count = x.size // block_size
    if block_count == 0:
        return np.asarray([], dtype=float)
    return x[: block_count * block_size].reshape(block_count, block_size).mean(axis=1)


__all__ = ["integrated_autocorrelation_time", "summarize_uncertainty", "block_means"]
