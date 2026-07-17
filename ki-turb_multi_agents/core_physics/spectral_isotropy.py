"""Spectral isotropy from isotropy_coeff_*.dat (canonical HIT product format).

File columns (7):
  k, E11, E22, E33, dE11/dk, IC_standard, IC_derivative

IC_standard = E22/E11
IC_derivative = (2*E22 - k_phys * dE11/dk) / (2*E11)

The IC(k) time-average figure uses IC_standard (E22/E11), time-averaged across
snapshots. Under-resolved shells may store IC=0 when E11 is below threshold;
those zeros are treated as missing (NaN), not physical zeros.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

_EPS = 1e-15


def read_isotropy_coeff_file(filepath: Path) -> np.ndarray:
    d = np.loadtxt(str(filepath), comments="#", encoding="utf-8")
    return d.reshape(1, -1) if d.ndim == 1 else d


def _ic_standard(e11: np.ndarray, e22: np.ndarray) -> np.ndarray:
    out = np.full_like(e11, np.nan, dtype=float)
    np.divide(e22, e11, out=out, where=np.isfinite(e11) & (e11 > _EPS))
    return out


def _mask_invalid_ic(ic: np.ndarray, e11: np.ndarray) -> np.ndarray:
    """Treat IC=0 on under-resolved shells (tiny E11) as missing."""
    out = np.asarray(ic, dtype=float).copy()
    invalid = (
        ~np.isfinite(out)
        | (e11 <= _EPS)
        | ((np.abs(out) < _EPS) & (e11 <= 1e-12))
    )
    out[invalid] = np.nan
    return out


def _snapshot_ics(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (k, E11, IC_standard, IC_derivative) for one snapshot."""
    k = np.asarray(data[:, 0], dtype=float)
    e11 = np.asarray(data[:, 1], dtype=float)
    e22 = np.asarray(data[:, 2], dtype=float)
    ic_std = _ic_standard(e11, e22)
    if data.shape[1] >= 7:
        ic_file = _mask_invalid_ic(data[:, 5], e11)
        # Prefer recomputed E22/E11; fall back to file col when needed.
        ic_std = np.where(np.isfinite(ic_std), ic_std, ic_file)
        ic_deriv = _mask_invalid_ic(data[:, 6], e11)
        ic_deriv = np.where(np.isfinite(ic_deriv), ic_deriv, ic_std)
    else:
        ic_deriv = ic_std.copy()
    return k, e11, ic_std, ic_deriv


def avg_isotropy_coeff(data_arrays: Iterable[np.ndarray]) -> Optional[dict]:
    """
    Time-average spectral isotropy coefficients across snapshots.

    Primary curve (IC_mean) = temporal mean of IC_standard = E22/E11.
    IC_std = temporal std of that ratio.
    Zeros from under-resolved shells are ignored (NaN), not averaged in.
    """
    valid: List[np.ndarray] = [
        np.asarray(d, dtype=float)
        for d in data_arrays
        if getattr(d, "size", 0) and np.asarray(d).ndim == 2 and np.asarray(d).shape[1] >= 4
    ]
    if not valid:
        return None

    n = min(len(d) for d in valid)
    k = np.asarray(valid[0][:n, 0], dtype=float)
    e_stack = np.stack([d[:n, 1:4] for d in valid], axis=0)  # (T, K, 3)
    means = np.nanmean(e_stack, axis=0)
    e11_mean, e22_mean, e33_mean = means[:, 0], means[:, 1], means[:, 2]

    ic_std_rows = []
    ic_deriv_rows = []
    for d in valid:
        _, e11, ic_s, ic_d = _snapshot_ics(d[:n])
        ic_std_rows.append(ic_s)
        ic_deriv_rows.append(ic_d)
    ic_std_arr = np.stack(ic_std_rows, axis=0)
    ic_deriv_arr = np.stack(ic_deriv_rows, axis=0)

    with np.errstate(all="ignore"):
        ic_standard_mean = np.nanmean(ic_std_arr, axis=0)
        ic_standard_std = np.nanstd(ic_std_arr, axis=0)
        ic_deriv_mean = np.nanmean(ic_deriv_arr, axis=0)
        ic_deriv_std = np.nanstd(ic_deriv_arr, axis=0)
    ic_from_mean_e = _ic_standard(e11_mean, e22_mean)

    # Prefer ratio-of-means where E11 is trustworthy; else temporal mean of ratios.
    ic_mean = np.where(
        np.isfinite(ic_from_mean_e) & (e11_mean > _EPS),
        ic_from_mean_e,
        ic_standard_mean,
    )
    ic_std = ic_standard_std

    mean_component = means.mean(axis=1)
    ratios = np.divide(
        means,
        mean_component[:, None],
        out=np.full_like(means, np.nan),
        where=mean_component[:, None] > 0,
    )

    return {
        "k": k,
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "IC_standard_mean": ic_standard_mean,
        "IC_standard_std": ic_standard_std,
        "IC_deriv_mean": ic_deriv_mean,
        "IC_deriv_std": ic_deriv_std,
        "IC_from_mean_E": ic_from_mean_e,
        "E11_mean": e11_mean,
        "E22_mean": e22_mean,
        "E33_mean": e33_mean,
        "component_ratios": ratios,
        "isotropic_target": 1.0,
    }


def snapshot_ic_curve(data: np.ndarray, *, kind: str = "standard") -> tuple[np.ndarray, np.ndarray]:
    """Per-snapshot IC curve for overlay lines. kind: 'standard' | 'deriv'."""
    k, _e11, ic_std, ic_deriv = _snapshot_ics(np.asarray(data, dtype=float))
    return k, ic_std if kind != "deriv" else ic_deriv
