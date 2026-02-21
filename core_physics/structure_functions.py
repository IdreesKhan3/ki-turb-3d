"""
Structure function time-averaging and theory curves.

She-Leveque scaling, experimental reference, time-averaging of S_p(r).
No Streamlit or UI dependencies.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


TABLE_P = [2, 3, 4, 5, 6]
EXP_ZETA = [0.71, 1.00, 1.28, 1.53, 1.78]


def zeta_p_she_leveque(p):
    """She-Leveque 1994 scaling exponent."""
    return p / 9 + 2 * (1 - (2 / 3) ** (p / 3))


def compute_structure_time_avg(
    data_list: List[Dict[str, Any]],
) -> Tuple[
    Optional[np.ndarray],
    Optional[Dict[int, np.ndarray]],
    Optional[Dict[int, np.ndarray]],
    float,
    Optional[List[int]],
]:
    """
    Time-average structure functions over snapshots.

    Args:
        data_list: List of dicts from structure function reader, each with r, S_p, u_rms

    Returns:
        (r_mean, Sp_mean_dict, Sp_std_dict, u_rms_mean, ps) or (None, None, None, 0.0, None)
    """
    sum_sp = None
    sum_r = None
    total_u_rms = 0.0
    num_files = 0
    ps = None
    max_dr = None

    for data in data_list:
        r = np.asarray(data.get("r", []), float)
        S_p = data.get("S_p", {})
        if r.size == 0 or not S_p:
            continue

        if sum_sp is None:
            max_dr = len(r)
            ps = sorted(S_p.keys())
            sum_sp = {p: np.zeros(max_dr, dtype=float) for p in ps}
            sum_r = np.zeros(max_dr, dtype=float)

        for p in ps:
            if p in S_p:
                sp_arr = np.asarray(S_p[p], float)
                min_len = min(len(sum_sp[p]), len(sp_arr))
                sum_sp[p][:min_len] += sp_arr[:min_len]

        min_r_len = min(len(sum_r), len(r))
        sum_r[:min_r_len] += r[:min_r_len]
        total_u_rms += float(data.get("u_rms", 0.0))
        num_files += 1

    if num_files == 0 or sum_sp is None:
        return None, None, None, 0.0, None

    r_mean = sum_r / num_files
    Sp_mean_dict = {p: sum_sp[p] / num_files for p in ps}
    u_rms_mean = total_u_rms / num_files

    Sp_list = []
    for data in data_list:
        S_p = data.get("S_p", {})
        if S_p:
            Sp_mat = np.vstack([np.asarray(S_p[p], float)[:max_dr] for p in ps])
            Sp_list.append(Sp_mat)

    if Sp_list:
        Sp_arr = np.stack(Sp_list, axis=0)
        Sp_std = np.std(Sp_arr, axis=0)
        Sp_std_dict = {p: Sp_std[idx, :] for idx, p in enumerate(ps)}
    else:
        Sp_std_dict = {p: np.zeros(max_dr) for p in ps}

    return r_mean, Sp_mean_dict, Sp_std_dict, u_rms_mean, list(ps)
