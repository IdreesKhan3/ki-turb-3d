"""Spectral isotropy products in the canonical isotropy_coeff layout."""
from __future__ import annotations

from typing import List

import numpy as np

from ._spectral import component_spectra_discrete, dE11_dk_shell
from .readers import VelocitySnapshot

_EPS = 1e-15


def compute_spectral_isotropy(snapshots: List[VelocitySnapshot]) -> List[dict]:
    """
    Per-snapshot isotropy_coeff files:

      k, E11, E22, E33, dE11/dk, IC_standard, IC_derivative

    with IC_standard = E22/E11 and
    IC_derivative = (2*E22 - k_phys * dE11/dk) / (2*E11),
    k_phys = k_discrete * 2π/nx.
    """
    out: List[dict] = []
    for snap in snapshots:
        q = component_spectra_discrete(snap.velocity)
        k = q["k"]
        e11, e22, e33 = q["E11"], q["E22"], q["E33"]
        count = q["shell_count"]
        nx = float(q["nx"])
        dE = dE11_dk_shell(e11, count, dk=float(q["dk"]))

        ic_std = np.zeros_like(e11)
        ic_deriv = np.zeros_like(e11)
        for bin_i in range(1, len(k)):
            if count[bin_i] <= 0 or e11[bin_i] <= _EPS:
                continue
            ic_std[bin_i] = e22[bin_i] / e11[bin_i]
            k_phys = k[bin_i] * (2.0 * np.pi / nx)
            ic_deriv[bin_i] = (2.0 * e22[bin_i] - k_phys * dE[bin_i]) / (2.0 * e11[bin_i])

        # Keep only shells that received at least one mode.
        keep = count > 0
        columns = np.column_stack(
            [k[keep], e11[keep], e22[keep], e33[keep], dE[keep], ic_std[keep], ic_deriv[keep]]
        )
        out.append(
            {
                "step": snap.step,
                "time": snap.time,
                "k": k[keep],
                "E11": e11[keep],
                "E22": e22[keep],
                "E33": e33[keep],
                "dE11_dk": dE[keep],
                "IC_standard": ic_std[keep],
                "IC_derivative": ic_deriv[keep],
                "columns": columns,
            }
        )
    return out
