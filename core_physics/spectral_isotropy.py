"""
Spectral isotropy computations.

Time-averaging of IC(k), E11, E22, E33 from isotropy_coeff files.
No Streamlit or UI dependencies.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any


def read_isotropy_coeff_file(filepath: Path) -> np.ndarray:
    """
    Read isotropy coefficient file (isotropy_coeff_*.dat).

    Expected columns: k, E11, E22, E33, dE11/dk, IC_standard, IC_deriv (col 6).
    Returns 2D array (nrows, ncols).
    """
    data = np.loadtxt(str(filepath), comments="#", encoding="utf-8")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def avg_isotropy_coeff(data_arrays: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    Average IC(k), E11, E22, E33 across snapshots.

    Args:
        data_arrays: List of 2D arrays from read_isotropy_coeff_file (one per file)

    Returns:
        Dict with k, IC_mean, IC_std, E11_mean, E22_mean, E33_mean (or None if no valid data)
    """
    all_k, all_ic, all_e11, all_e22, all_e33 = [], [], [], [], []

    for d in data_arrays:
        if d.size == 0:
            continue

        k = d[:, 0]
        E11 = d[:, 1] if d.shape[1] > 1 else None
        E22 = d[:, 2] if d.shape[1] > 2 else None
        E33 = d[:, 3] if d.shape[1] > 3 else None

        if d.shape[1] >= 7:
            IC = d[:, 6]
        else:
            IC = np.divide(E11, E22, out=np.zeros_like(E11), where=E22 != 0)

        valid = (k > 0.5) & np.isfinite(IC) & (E11 > 1e-15)
        if np.any(valid):
            all_k.append(k[valid])
            all_ic.append(IC[valid])
            if E11 is not None:
                all_e11.append(E11[valid])
            if E22 is not None:
                all_e22.append(E22[valid])
            if E33 is not None:
                all_e33.append(E33[valid])

    if not all_ic:
        return None

    unique_k = np.unique(np.concatenate(all_k))
    ic_mean = np.zeros_like(unique_k)
    ic_std = np.zeros_like(unique_k)
    e11_mean = np.zeros_like(unique_k)
    e22_mean = np.zeros_like(unique_k)
    e33_mean = np.zeros_like(unique_k)
    counts = np.zeros_like(unique_k)

    for i, k0 in enumerate(unique_k):
        ic_vals, e11_vals, e22_vals, e33_vals = [], [], [], []
        for k, ic, e11, e22, e33 in zip(
            all_k, all_ic,
            all_e11 or [None] * len(all_ic),
            all_e22 or [None] * len(all_ic),
            all_e33 or [None] * len(all_ic),
        ):
            idx = np.argmin(np.abs(k - k0))
            if np.abs(k[idx] - k0) < 0.1:
                ic_vals.append(ic[idx])
                if e11 is not None:
                    e11_vals.append(e11[idx])
                if e22 is not None:
                    e22_vals.append(e22[idx])
                if e33 is not None:
                    e33_vals.append(e33[idx])

        if ic_vals:
            ic_mean[i] = np.mean(ic_vals)
            ic_std[i] = np.std(ic_vals)
            counts[i] = len(ic_vals)
            if e11_vals:
                e11_mean[i] = np.mean(e11_vals)
            if e22_vals:
                e22_mean[i] = np.mean(e22_vals)
            if e33_vals:
                e33_mean[i] = np.mean(e33_vals)

    min_samples = max(1, len(all_ic) // 2)
    mask = counts >= min_samples

    return {
        "k": unique_k[mask],
        "IC_mean": ic_mean[mask],
        "IC_std": ic_std[mask],
        "E11_mean": e11_mean[mask] if all_e11 else None,
        "E22_mean": e22_mean[mask] if all_e22 else None,
        "E33_mean": e33_mean[mask] if all_e33 else None,
    }
