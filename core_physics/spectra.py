"""
Energy spectrum time-averaging.

Pure averaging logic for (k, E) and normalized spectra.
No Streamlit or UI dependencies.
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_spectrum_time_avg(
    data_list: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Time-average spectrum data (k, E) over multiple snapshots.

    Args:
        data_list: List of (k, E) tuples from spectrum files

    Returns:
        (k_vals, E_avg, E_std) or (None, None, None) if no valid data
    """
    energy_accum = None
    energy_sq_accum = None
    count = 0
    k_vals = None

    for k, E in data_list:
        k = np.asarray(k, float)
        E = np.asarray(E, float)
        if k_vals is None:
            k_vals = k
            energy_accum = np.zeros_like(E)
            energy_sq_accum = np.zeros_like(E)

        if E.shape != energy_accum.shape:
            continue

        energy_accum += E
        energy_sq_accum += E**2
        count += 1

    if count == 0:
        return None, None, None

    E_avg = energy_accum / count
    E_var = (energy_sq_accum / count) - E_avg**2
    E_std = np.sqrt(np.maximum(E_var, 0.0))
    return k_vals, E_avg, E_std


def compute_spectrum_time_avg_norm(
    data_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Time-average normalized spectrum (keta, Enorm, Epope) over multiple snapshots.

    Args:
        data_list: List of (keta, Enorm, Epope) tuples from norm spectrum files

    Returns:
        (keta_vals, En_avg, En_std, Ep_avg) or (None, None, None, None) if no valid data
    """
    keta_vals = None
    En_accum = None
    En_sq_accum = None
    Ep_accum = None
    count = 0

    for keta, Enorm, Epope in data_list:
        keta = np.asarray(keta, float)
        Enorm = np.asarray(Enorm, float)
        Epope = np.asarray(Epope, float)
        if keta_vals is None:
            keta_vals = keta
            En_accum = np.zeros_like(Enorm)
            En_sq_accum = np.zeros_like(Enorm)
            Ep_accum = np.zeros_like(Epope)

        if Enorm.shape != En_accum.shape:
            continue

        En_accum += Enorm
        En_sq_accum += Enorm**2
        Ep_accum += Epope
        count += 1

    if count == 0:
        return None, None, None, None

    En_avg = En_accum / count
    En_var = (En_sq_accum / count) - En_avg**2
    En_std = np.sqrt(np.maximum(En_var, 0.0))
    Ep_avg = Ep_accum / count
    return keta_vals, En_avg, En_std, Ep_avg
