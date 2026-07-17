"""Detect statistically stationary windows in forced HIT time histories."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from schemas.hit_analysis_products import StationarityAssessment


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if abs(mean) < 1.0e-15:
        return float("inf")
    return float(np.std(values, ddof=1) / abs(mean)) if values.size > 1 else 0.0


def _normalized_slope(time: np.ndarray, values: np.ndarray) -> float:
    if time.size < 2:
        return float("inf")
    centered = time - np.mean(time)
    denominator = float(np.dot(centered, centered))
    slope = float(np.dot(centered, values - np.mean(values)) / denominator) if denominator else 0.0
    scale = max(abs(float(np.mean(values))), 1.0e-15)
    duration = max(float(time[-1] - time[0]), 1.0e-15)
    return slope * duration / scale


def detect_stationarity(
    time: Sequence[float] | np.ndarray,
    tke: Sequence[float] | np.ndarray,
    dissipation: Optional[Sequence[float] | np.ndarray] = None,
    forcing_power: Optional[Sequence[float] | np.ndarray] = None,
    *,
    minimum_samples: int = 8,
    window_fraction: float = 0.35,
    cv_limit: float = 0.05,
    normalized_slope_limit: float = 0.05,
    power_balance_limit: float = 0.15,
) -> StationarityAssessment:
    """Find the earliest tail window satisfying stationarity criteria.

    The detector scans candidate windows ending at the final sample.  It checks
    TKE variability and drift, and—when available—dissipation/forcing variability
    and their mean balance.
    """
    time_array = np.asarray(time, dtype=float)
    tke_array = np.asarray(tke, dtype=float)
    if time_array.ndim != 1 or tke_array.ndim != 1 or time_array.size != tke_array.size:
        raise ValueError("time and tke must be one-dimensional arrays of equal length")
    if time_array.size < minimum_samples:
        return StationarityAssessment(
            stationary=False,
            reason=f"insufficient samples: {time_array.size} < {minimum_samples}",
        )
    if np.any(np.diff(time_array) <= 0):
        return StationarityAssessment(
            stationary=False,
            reason="time values are not strictly increasing",
        )

    dissipation_array = None if dissipation is None else np.asarray(dissipation, dtype=float)
    power_array = None if forcing_power is None else np.asarray(forcing_power, dtype=float)
    for name, values in (("dissipation", dissipation_array), ("forcing_power", power_array)):
        if values is not None and values.shape != tke_array.shape:
            raise ValueError(f"{name} must match time and tke")

    largest_start = time_array.size - minimum_samples
    default_start = max(0, int(time_array.size * (1.0 - window_fraction)))
    candidate_starts = range(min(default_start, largest_start), largest_start + 1)
    best: Optional[StationarityAssessment] = None

    for start in candidate_starts:
        sl = slice(start, None)
        window_time = time_array[sl]
        window_tke = tke_array[sl]
        tke_cv = _coefficient_of_variation(window_tke)
        tke_slope = abs(_normalized_slope(window_time, window_tke))
        diss_cv = _coefficient_of_variation(dissipation_array[sl]) if dissipation_array is not None else None
        power_cv = _coefficient_of_variation(power_array[sl]) if power_array is not None else None

        checks = [tke_cv <= cv_limit, tke_slope <= normalized_slope_limit]
        if diss_cv is not None:
            checks.append(diss_cv <= cv_limit)
        if power_cv is not None:
            checks.append(power_cv <= cv_limit)
        if dissipation_array is not None and power_array is not None:
            mean_diss = abs(float(np.mean(dissipation_array[sl])))
            relative_balance = abs(float(np.mean(power_array[sl] - dissipation_array[sl]))) / max(mean_diss, 1.0e-15)
            checks.append(relative_balance <= power_balance_limit)

        assessment = StationarityAssessment(
            stationary=all(checks),
            start_index=start,
            end_index=time_array.size - 1,
            start_time=float(window_time[0]),
            end_time=float(window_time[-1]),
            tke_cv=tke_cv,
            dissipation_cv=diss_cv,
            power_cv=power_cv,
            normalized_tke_slope=tke_slope,
            reason="stationary tail window found" if all(checks) else "tail window exceeds stationarity thresholds",
        )
        best = assessment
        if assessment.stationary:
            return assessment

    return best or StationarityAssessment(stationary=False, reason="no candidate window")


__all__ = ["detect_stationarity"]
