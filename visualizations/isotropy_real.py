# visualizations/isotropy_real.py
"""
Real-space isotropy analysis core (Plotly-ready).

Reads:
- eps_real_validation.csv  (required)
- reynolds_stress_validation.csv (optional)

Outputs:
- Plotly figures:
    1) Energy fractions vs time (with optional moving averages, bands)
    2) Lumley triangle trajectory (xi-eta map)
    3) Anisotropy tensor diag components vs time
    4) Cross correlations + anisotropy index vs time
    5) Deviations from isotropy vs time
    6) Convergence metric (running std)

This module contains NO Streamlit code.
Streamlit page should import and call build_real_isotropy_figs(...).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import plotly.graph_objects as go


# ==========================================================
# Robust CSV loaders
# ==========================================================
def load_eps_real_validation(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Robust read of eps_real_validation.csv
    Expected columns (names may vary slightly):
    iter, iter_norm, eps_real, TKE_real, u_rms_real, frac_x, frac_y, frac_z, Re_Tp ...
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required file: {csv_path}")

    df = pd.read_csv(csv_path, comment="#", sep=r"[,\s]+", engine="python")
    col_map = {c.lower(): c for c in df.columns}

    def _col(*names, default=None):
        for n in names:
            if n.lower() in col_map:
                return df[col_map[n.lower()]].to_numpy(dtype=float)
        if default is not None:
            return df[default].to_numpy(dtype=float)
        raise KeyError(f"Could not find any of columns {names} in {csv_path.name}")

    data = {
        "iter": _col("iter"),
        "iter_norm": _col("iter_norm", default=col_map.get("iter")),
        "TKE": _col("tke_real", "tke"),
        "u_rms": _col("u_rms_real", "u_rms"),
        "eps0": _col("eps_real", "eps0"),
        "frac_x": _col("frac_x"),
        "frac_y": _col("frac_y"),
        "frac_z": _col("frac_z"),
    }

    return data


def load_reynolds_stress_validation(stress_path: Path, n_expected: int) -> Optional[Dict[str, np.ndarray]]:
    """
    Optional file read. Expected columns:
    iter, R11, R22, R33, R12, R13, R23
    Returns None if file missing.
    """
    if not stress_path.exists():
        return None

    df = pd.read_csv(stress_path, comment="#", sep=r"[,\s]+", engine="python")
    df = df.iloc[:n_expected]

    if df.shape[1] < 7:
        return None

    R11 = df.iloc[:, 1].to_numpy(float)
    R22 = df.iloc[:, 2].to_numpy(float)
    R33 = df.iloc[:, 3].to_numpy(float)
    R12 = df.iloc[:, 4].to_numpy(float)
    R13 = df.iloc[:, 5].to_numpy(float)
    R23 = df.iloc[:, 6].to_numpy(float)
    TKE_from_R = 0.5 * (R11 + R22 + R33)

    return {
        "R11": R11, "R22": R22, "R33": R33,
        "R12": R12, "R13": R13, "R23": R23,
        "TKE": TKE_from_R,
    }


# ==========================================================
# Derived quantities
# ==========================================================
def compute_reynolds_from_fractions(
    frac_x: np.ndarray, frac_y: np.ndarray, frac_z: np.ndarray, TKE: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    If reynolds_stress_validation.csv not present, estimate diagonal stresses from fractions.
    Cross terms assumed ~0.
    """
    R11 = frac_x * 2.0 * TKE
    R22 = frac_y * 2.0 * TKE
    R33 = frac_z * 2.0 * TKE
    n = len(TKE)
    return {
        "R11": R11, "R22": R22, "R33": R33,
        "R12": np.zeros(n), "R13": np.zeros(n), "R23": np.zeros(n),
        "TKE": TKE
    }


def compute_anisotropy_tensor(R: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    k = R["TKE"]
    k_safe = np.where(k > 1e-10, k, 1e-10)

    b11 = R["R11"] / (2*k_safe) - 1/3
    b22 = R["R22"] / (2*k_safe) - 1/3
    b33 = R["R33"] / (2*k_safe) - 1/3
    b12 = R["R12"] / (2*k_safe)
    b13 = R["R13"] / (2*k_safe)
    b23 = R["R23"] / (2*k_safe)

    return {"b11": b11, "b22": b22, "b33": b33, "b12": b12, "b13": b13, "b23": b23}


def compute_invariants(b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    II_b = -0.5 * (
        b["b11"]**2 + b["b22"]**2 + b["b33"]**2 +
        2*(b["b12"]**2 + b["b13"]**2 + b["b23"]**2)
    )
    III_b = (1/3) * (
        b["b11"]**3 + b["b22"]**3 + b["b33"]**3 +
        3*b["b11"]*(b["b12"]**2 + b["b13"]**2) +
        3*b["b22"]*(b["b12"]**2 + b["b23"]**2) +
        3*b["b33"]*(b["b13"]**2 + b["b23"]**2) +
        6*b["b12"]*b["b13"]*b["b23"]
    )

    # Pope (2000) invariants for Lumley triangle
    eta = np.sqrt(np.maximum(-II_b/3, 0.0))
    xi = np.cbrt(III_b/2)

    anisotropy_index = np.sqrt(np.maximum(-2*II_b, 0.0))
    return {"II_b": II_b, "III_b": III_b, "xi": xi, "eta": eta, "anisotropy_index": anisotropy_index}


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window >= len(x):
        return np.mean(x) * np.ones_like(x)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def running_std_energy_fractions(fracs: Tuple[np.ndarray, np.ndarray, np.ndarray], window: int) -> Tuple[np.ndarray, np.ndarray]:
    Ex, Ey, Ez = fracs
    n = len(Ex)
    if window < 2 or window > n:
        return np.arange(n), np.zeros(n)

    stds = []
    for i in range(window, n+1):
        sx = np.std(Ex[i-window:i])
        sy = np.std(Ey[i-window:i])
        sz = np.std(Ez[i-window:i])
        stds.append((sx+sy+sz)/3)
    t_idx = np.arange(window-1, n)
    return t_idx, np.array(stds)


# ==========================================================
# Plot builders (Plotly)
# ==========================================================
def fig_energy_fractions(
    t: np.ndarray, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
    show_ma: bool = True, ma_windows: List[int] = None,
    show_tolerance: bool = True, tolerances: List[float] = None,
    stationarity_t: Optional[float] = None,
    colors: Dict[str, str] = None,
) -> go.Figure:

    colors = colors or {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
    ma_windows = ma_windows or []
    tolerances = tolerances or [0.005, 0.01, 0.02]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=Ex, mode="markers", name="Ex/Etot (raw)",
                             marker=dict(size=3, color=colors["x"]), opacity=0.35))
    fig.add_trace(go.Scatter(x=t, y=Ey, mode="markers", name="Ey/Etot (raw)",
                             marker=dict(size=3, color=colors["y"]), opacity=0.35))
    fig.add_trace(go.Scatter(x=t, y=Ez, mode="markers", name="Ez/Etot (raw)",
                             marker=dict(size=3, color=colors["z"]), opacity=0.35))

    if show_ma and ma_windows:
        for w in ma_windows:
            Ex_ma = moving_average(Ex, w)
            Ey_ma = moving_average(Ey, w)
            Ez_ma = moving_average(Ez, w)
            # align time to center of window
            t_ma = t[w//2:w//2 + len(Ex_ma)]
            fig.add_trace(go.Scatter(x=t_ma, y=Ex_ma, mode="lines", name=f"Ex MA-{w}",
                                     line=dict(color=colors["x"], width=2)))
            fig.add_trace(go.Scatter(x=t_ma, y=Ey_ma, mode="lines", name=f"Ey MA-{w}",
                                     line=dict(color=colors["y"], width=2)))
            fig.add_trace(go.Scatter(x=t_ma, y=Ez_ma, mode="lines", name=f"Ez MA-{w}",
                                     line=dict(color=colors["z"], width=2)))

    fig.add_hline(y=1/3, line_dash="dash", line_color="red", annotation_text="Isotropic (1/3)")

    if show_tolerance:
        for tol in tolerances:
            fig.add_hrect(y0=1/3-tol, y1=1/3+tol, fillcolor="rgba(255,0,0,0.08)", line_width=0,
                          annotation_text=f"±{tol:.1%}", annotation_position="top left")

    if stationarity_t is not None:
        fig.add_vline(x=stationarity_t, line_dash="dash", line_color="purple",
                      annotation_text="Stationarity")

    fig.update_layout(
        xaxis_title="t / t0",
        yaxis_title="Energy fraction",
        height=520,
        legend_title="Real-space isotropy",
        hovermode="x unified",
    )
    return fig


def fig_lumley_triangle(
    xi: np.ndarray, eta: np.ndarray,
    show_boundaries: bool = True
) -> go.Figure:
    fig = go.Figure()

    if show_boundaries:
        xi_vals = np.linspace(-1/6, 1/3, 400)
        eta_two_comp = np.sqrt(1/27 + 2*xi_vals**3)
        eta_lower = np.where(xi_vals < 0, -xi_vals, xi_vals)

        fig.add_trace(go.Scatter(x=xi_vals[xi_vals <= 0], y=-xi_vals[xi_vals <= 0],
                                 mode="lines", name="Axisymmetric expansion",
                                 line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=xi_vals[xi_vals >= 0], y=xi_vals[xi_vals >= 0],
                                 mode="lines", name="Axisymmetric contraction",
                                 line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=xi_vals, y=eta_two_comp, mode="lines",
                                 name="Two-component limit", line=dict(color="red", width=2)))

        fig.add_trace(go.Scatter(
            x=np.concatenate([xi_vals, xi_vals[::-1]]),
            y=np.concatenate([eta_lower, eta_two_comp[::-1]]),
            fill="toself", fillcolor="rgba(200,200,200,0.25)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))

    # trajectory
    fig.add_trace(go.Scatter(x=xi, y=eta, mode="lines+markers",
                             name="DNS trajectory",
                             marker=dict(size=6, color=np.linspace(0, 1, len(xi)),
                                         colorscale="Viridis", line=dict(width=1, color="black"))))

    # special points
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Isotropic",
                             marker=dict(size=10, color="gold", symbol="star")))
    fig.add_trace(go.Scatter(x=[-1/6], y=[1/6], mode="markers", name="2-comp axisym",
                             marker=dict(size=9, color="magenta")))
    fig.add_trace(go.Scatter(x=[1/3], y=[1/3], mode="markers", name="1-comp",
                             marker=dict(size=9, color="blue")))

    fig.update_layout(
        xaxis_title=r"ξ = (III_b / 2)^{1/3}",
        yaxis_title=r"η = (-II_b / 3)^{1/2}",
        xaxis_range=[-0.2, 0.35],
        yaxis_range=[-0.01, 0.35],
        height=520,
        legend_title="Lumley triangle",
        hovermode="closest",
    )
    return fig


def fig_anisotropy_diagonal(
    t: np.ndarray, b: Dict[str, np.ndarray]
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=b["b11"], mode="lines+markers", name="b11"))
    fig.add_trace(go.Scatter(x=t, y=b["b22"], mode="lines+markers", name="b22"))
    fig.add_trace(go.Scatter(x=t, y=b["b33"], mode="lines+markers", name="b33"))
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.update_layout(
        xaxis_title="t / t0",
        yaxis_title="b_ii",
        height=420,
        legend_title="Anisotropy tensor",
        hovermode="x unified",
    )
    return fig


def fig_cross_corr_and_index(
    t: np.ndarray, b: Dict[str, np.ndarray], anis_idx: np.ndarray
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.abs(b["b12"]), mode="lines+markers", name="|b12|"))
    fig.add_trace(go.Scatter(x=t, y=np.abs(b["b13"]), mode="lines+markers", name="|b13|"))
    fig.add_trace(go.Scatter(x=t, y=np.abs(b["b23"]), mode="lines+markers", name="|b23|"))
    fig.add_trace(go.Scatter(x=t, y=anis_idx, mode="lines", name="anisotropy index",
                             line=dict(width=3, color="black")))

    fig.update_layout(
        xaxis_title="t / t0",
        yaxis_title="Cross-correlations / index",
        yaxis_type="log",
        height=420,
        hovermode="x unified",
    )
    return fig


def fig_deviations(
    t: np.ndarray, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray
) -> go.Figure:
    dev_x = np.abs(Ex - 1/3)
    dev_y = np.abs(Ey - 1/3)
    dev_z = np.abs(Ez - 1/3)
    dev_max = np.maximum.reduce([dev_x, dev_y, dev_z])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=dev_x, mode="lines+markers", name="|Ex-1/3|"))
    fig.add_trace(go.Scatter(x=t, y=dev_y, mode="lines+markers", name="|Ey-1/3|"))
    fig.add_trace(go.Scatter(x=t, y=dev_z, mode="lines+markers", name="|Ez-1/3|"))
    fig.add_trace(go.Scatter(x=t, y=dev_max, mode="lines", name="max deviation",
                             line=dict(width=2, color="black")))

    fig.update_layout(
        xaxis_title="t / t0",
        yaxis_title="Absolute deviation",
        yaxis_type="log",
        height=420,
        hovermode="x unified",
    )
    return fig


def fig_convergence(
    t: np.ndarray, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
    window: int = 50
) -> go.Figure:
    idx, run_std = running_std_energy_fractions((Ex, Ey, Ez), window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[idx], y=run_std, mode="lines", name=f"running σ (w={window})"))

    fig.update_layout(
        xaxis_title="t / t0",
        yaxis_title="Running std",
        yaxis_type="log",
        height=360,
        hovermode="x unified",
    )
    return fig


# ==========================================================
# Public API for Streamlit page
# ==========================================================
def build_real_isotropy_figs(
    data_dir: Path,
    ma_windows: List[int] = None,
    tolerances: List[float] = None,
    stationarity_iter: Optional[float] = None,
    convergence_window: int = 50,
) -> Dict[str, go.Figure]:
    """
    Main entry point:
    Returns dict of figures usable by Streamlit.
    """
    eps_path = data_dir / "eps_real_validation.csv"
    stress_path = data_dir / "reynolds_stress_validation.csv"

    data = load_eps_real_validation(eps_path)
    n = len(data["iter"])

    R = load_reynolds_stress_validation(stress_path, n)
    if R is None:
        R = compute_reynolds_from_fractions(data["frac_x"], data["frac_y"], data["frac_z"], data["TKE"])

    b = compute_anisotropy_tensor(R)
    inv = compute_invariants(b)

    iter0 = data["iter"][0] if data["iter"][0] != 0 else 1.0
    t_norm = data["iter"] / iter0

    ma_windows = ma_windows or []
    tolerances = tolerances or [0.005, 0.01, 0.02]
    stat_t = None if stationarity_iter is None else stationarity_iter / iter0

    figs = {
        "fractions": fig_energy_fractions(t_norm, data["frac_x"], data["frac_y"], data["frac_z"],
                                          show_ma=True, ma_windows=ma_windows,
                                          show_tolerance=True, tolerances=tolerances,
                                          stationarity_t=stat_t),
        "lumley": fig_lumley_triangle(inv["xi"], inv["eta"]),
        "bij_diag": fig_anisotropy_diagonal(t_norm, b),
        "cross_corr": fig_cross_corr_and_index(t_norm, b, inv["anisotropy_index"]),
        "deviations": fig_deviations(t_norm, data["frac_x"], data["frac_y"], data["frac_z"]),
        "convergence": fig_convergence(t_norm, data["frac_x"], data["frac_y"], data["frac_z"],
                                       window=convergence_window),
    }
    return figs


# ==========================================================
# Standalone run (quick check)
# ==========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-space isotropy quick plot check.")
    parser.add_argument("--data-dir", type=str, default=".", help="Folder with eps_real_validation.csv")
    args = parser.parse_args()

    figs = build_real_isotropy_figs(Path(args.data_dir), ma_windows=[200], stationarity_iter=50000)
    # write one html to check quickly
    out = Path(args.data_dir) / "real_isotropy_fractions.html"
    figs["fractions"].write_html(str(out))
    print(f"Saved quick check: {out}")
