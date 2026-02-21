"""
Flatness factor time-averaging.

Time-average flatness F(r) over snapshots, with log-spaced r for error bars.
No Streamlit or UI dependencies.
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_flatness_time_avg(
    data_list: List[Tuple[np.ndarray, np.ndarray]],
    num_errorbars: int = 20,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Time-average flatness data and select log-spaced r values for error bars.

    Args:
        data_list: List of (r, F) tuples from flatness files
        num_errorbars: Number of log-spaced r positions for output

    Returns:
        (r_plot, F_mean, F_std) on selected r indices, or (None, None, None) if invalid
    """
    all_r = []
    all_flatness = []

    for r, F in data_list:
        r = np.asarray(r, float)
        F = np.asarray(F, float)
        if r.size == 0 or F.size == 0:
            continue
        all_r.append(r)
        all_flatness.append(F)

    if not all_r:
        return None, None, None

    r_full = all_r[0]
    flatness_array = np.array(all_flatness)

    if flatness_array.ndim != 2 or flatness_array.shape[1] != r_full.shape[0]:
        return None, None, None

    flatness_mean = np.mean(flatness_array, axis=0)
    flatness_std = np.std(flatness_array, axis=0)

    r_pos = r_full[r_full > 0]
    if r_pos.size < 2:
        return None, None, None

    log_r_vals = np.logspace(np.log10(r_pos[0]), np.log10(r_pos[-1]), num=num_errorbars)
    log_indices = sorted(set(int(np.argmin(np.abs(r_full - val))) for val in log_r_vals))

    r_plot = r_full[log_indices]
    F_mean = flatness_mean[log_indices]
    F_std = flatness_std[log_indices]
    return r_plot, F_mean, F_std
