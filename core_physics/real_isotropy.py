"""
Real-space isotropy computations.

Pure physics logic: Reynolds stress, anisotropy tensor, Lumley invariants.
No Streamlit or UI dependencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_turbulence_data(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load turbulence validation data from eps_real_validation.csv.
    Supports both LBM and NS formats. Accepts column aliases: u_rms_real/u_rms,
    TKE_real/TKE, frac_x/frac_y/frac_z or E_x/E_y/E_z.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dict with iter, iter_norm, TKE, u_rms, eps0, frac_x, frac_y, frac_z
    """
    df = pd.read_csv(csv_path)

    # Numeric parse (LBM and NS column names)
    for col in ["iter", "iter_norm", "TKE_real", "TKE", "u_rms_real", "u_rms",
                "eps_real", "frac_x", "frac_y", "frac_z", "E_x", "E_y", "E_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cols = df.columns.tolist()

    # Energy fractions: frac_x/y/z (LBM) or E_x/y/z (NS) or column indices
    if "frac_x" not in df.columns:
        if "E_x" in df.columns and "E_y" in df.columns and "E_z" in df.columns:
            df["frac_x"] = df["E_x"]
            df["frac_y"] = df["E_y"]
            df["frac_z"] = df["E_z"]
        elif len(cols) >= 20:
            df["frac_x"] = pd.to_numeric(df.iloc[:, 17], errors="coerce")
            df["frac_y"] = pd.to_numeric(df.iloc[:, 18], errors="coerce")
            df["frac_z"] = pd.to_numeric(df.iloc[:, 19], errors="coerce")

    # TKE: TKE_real (LBM) or TKE (NS)
    tke_col = df.get("TKE_real") if "TKE_real" in df.columns else df.get("TKE")
    if tke_col is None and len(df.columns) > 4:
        tke_col = df.iloc[:, 4]
    tke = tke_col.to_numpy() if tke_col is not None else np.zeros(len(df))

    # u_rms: u_rms_real (LBM) or u_rms (NS)
    u_rms_col = df.get("u_rms_real") if "u_rms_real" in df.columns else df.get("u_rms")
    if u_rms_col is None and len(df.columns) > 5:
        u_rms_col = df.iloc[:, 5]
    u_rms = u_rms_col.to_numpy() if u_rms_col is not None else np.zeros(len(df))

    data = {
        "iter": df["iter"].to_numpy(),
        "iter_norm": df.get("iter_norm", df["iter"]).to_numpy(),
        "TKE": tke,
        "u_rms": u_rms,
        "eps0": (df["eps_real"] if "eps_real" in df.columns else (df.iloc[:, 2] if len(df.columns) > 2 else df["iter"])).to_numpy(),
        "frac_x": df["frac_x"].to_numpy(),
        "frac_y": df["frac_y"].to_numpy(),
        "frac_z": df["frac_z"].to_numpy(),
    }
    return data


def load_reynolds_stress(stress_path: Path, turb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Load Reynolds stress from CSV, or compute from energy fractions if file missing.

    Args:
        stress_path: Path to reynolds_stress_validation.csv
        turb: Turbulence data from load_turbulence_data

    Returns:
        Dict with R11, R22, R33, R12, R13, R23, TKE
    """
    if not stress_path.exists():
        return compute_reynolds_from_fractions(turb)

    df = pd.read_csv(stress_path)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    n = min(len(df), len(turb["iter"]))

    R11, R22, R33 = df.iloc[:n, 1], df.iloc[:n, 2], df.iloc[:n, 3]
    R12, R13, R23 = df.iloc[:n, 4], df.iloc[:n, 5], df.iloc[:n, 6]
    TKE_from_R = 0.5 * (R11 + R22 + R33)

    return {
        "R11": R11.to_numpy(),
        "R22": R22.to_numpy(),
        "R33": R33.to_numpy(),
        "R12": R12.to_numpy(),
        "R13": R13.to_numpy(),
        "R23": R23.to_numpy(),
        "TKE": TKE_from_R.to_numpy(),
    }


def compute_reynolds_from_fractions(turb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute Reynolds stress from energy fractions when CSV not available.

    R_ii = frac_i * 2 * TKE, cross-terms zero for axis-aligned case.
    """
    TKE = turb["TKE"]
    R11 = turb["frac_x"] * 2 * TKE
    R22 = turb["frac_y"] * 2 * TKE
    R33 = turb["frac_z"] * 2 * TKE
    n = len(TKE)
    return dict(
        R11=R11, R22=R22, R33=R33,
        R12=np.zeros(n), R13=np.zeros(n), R23=np.zeros(n),
        TKE=TKE
    )


def anisotropy_tensor(R: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Reynolds stress anisotropy tensor: b_ij = R_ij/(2k) - delta_ij/3.

    Returns:
        Dict with b11, b22, b33, b12, b13, b23
    """
    k = R["TKE"]
    k_safe = np.where(k > 1e-10, k, 1e-10)

    b11 = R["R11"] / (2 * k_safe) - 1 / 3
    b22 = R["R22"] / (2 * k_safe) - 1 / 3
    b33 = R["R33"] / (2 * k_safe) - 1 / 3
    b12 = R["R12"] / (2 * k_safe)
    b13 = R["R13"] / (2 * k_safe)
    b23 = R["R23"] / (2 * k_safe)

    return dict(b11=b11, b22=b22, b33=b33, b12=b12, b13=b13, b23=b23)


def invariants(b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Lumley invariants: II_b, III_b, anisotropy index, xi, eta.

    eta = (-II_b/3)^(1/2), xi = (III_b/2)^(1/3)
    """
    II_b = -0.5 * (
        b["b11"]**2 + b["b22"]**2 + b["b33"]**2 +
        2 * (b["b12"]**2 + b["b13"]**2 + b["b23"]**2)
    )
    III_b = (1 / 3) * (
        b["b11"]**3 + b["b22"]**3 + b["b33"]**3 +
        3 * b["b11"] * (b["b12"]**2 + b["b13"]**2) +
        3 * b["b22"] * (b["b12"]**2 + b["b23"]**2) +
        3 * b["b33"] * (b["b13"]**2 + b["b23"]**2) +
        6 * b["b12"] * b["b13"] * b["b23"]
    )
    anis_index = np.sqrt(-2 * II_b)
    eta = np.sqrt(-II_b / 3)
    xi = np.cbrt(III_b / 2)
    return dict(II_b=II_b, III_b=III_b, anis_index=anis_index, xi=xi, eta=eta)
