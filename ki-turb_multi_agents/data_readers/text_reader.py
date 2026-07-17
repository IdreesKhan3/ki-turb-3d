"""
Text file reader for structure functions and flatness data
Reads structure_functions_*.txt and flatness_data*_*.txt files
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def read_structure_function_txt(filepath: str) -> Dict:
    """
    Read structure function text file into the same dict shape as the binary reader.

    Supported layouts:
    - OpenLB / KI-TURB postprocess:
      ``# r SL2 SL3 ... ST2 ... signed_SL3`` → uses longitudinal ``SLn`` as ``S_p[n]``
    - Legacy numeric columns after ``r`` → orders ``1..N``
    """
    path = Path(filepath)
    try:
        with path.open("r", encoding="utf-8") as fh:
            first = fh.readline().strip()
        header_cols: list[str] = []
        if first.startswith("#"):
            header_cols = first.lstrip("#").split()

        data = np.loadtxt(filepath, comments="#", encoding="utf-8")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.size == 0 or data.shape[1] < 2:
            raise ValueError("expected at least columns r and one S_p")

        r = np.asarray(data[:, 0], dtype=float)
        S_p: Dict[int, np.ndarray] = {}

        # Prefer explicit longitudinal columns from OpenLB/KI-TURB header.
        if header_cols and header_cols[0].lower() in {"r", "dr"}:
            for idx, name in enumerate(header_cols[1:], start=1):
                if idx >= data.shape[1]:
                    break
                match = re.fullmatch(r"SL(\d+)", str(name), flags=re.IGNORECASE)
                if match:
                    S_p[int(match.group(1))] = np.asarray(data[:, idx], dtype=float)

        # Legacy / unlabeled: remaining columns are successive orders starting at 1.
        if not S_p:
            for p in range(1, data.shape[1]):
                S_p[p] = np.asarray(data[:, p], dtype=float)

        if not S_p:
            raise ValueError("no structure-function columns found")

        # OpenLB/KI-TURB txt products do not store u_rms; estimate from large-r S_2
        # using the HIT limit ⟨(δu_L)²⟩ → 2 u_rms².
        u_rms = 0.0
        if 2 in S_p and len(S_p[2]) > 0:
            s2_plateau = float(np.nanmax(np.asarray(S_p[2], dtype=float)))
            if s2_plateau > 0.0:
                u_rms = float(np.sqrt(s2_plateau / 2.0))

        min_len = min(len(r), *(len(v) for v in S_p.values()))
        return {
            "r": r[:min_len],
            "S_p": {p: v[:min_len] for p, v in S_p.items()},
            "u_rms": u_rms,
            "norders": len(S_p),
            "max_dr": min_len,
            "source": str(path),
        }
    except Exception as e:
        raise ValueError(f"Error reading structure function file {filepath}: {e}") from e


def read_flatness_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read flatness file (flatness_data*_*.txt)

    Format: r, F(r) (two columns, skip comment/header rows)
    """
    try:
        data = np.loadtxt(filepath, comments="#", encoding="utf-8")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        r = data[:, 0]
        flatness = data[:, 1]
        return r, flatness
    except Exception as e:
        raise ValueError(f"Error reading flatness file {filepath}: {e}") from e
