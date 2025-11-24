# visualizations/isotropy_spectral.py
"""
Spectral isotropy analysis core (Plotly-ready).

Reads:
- isotropy_coeff_*.dat

For each simulation group:
- Time-average IC(k)
- Return Plotly figure

No Streamlit code here. Streamlit page imports and renders build_spectral_isotropy_fig(...).
"""

from __future__ import annotations
import numpy as np
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import plotly.graph_objects as go

from utils.file_detector import natural_sort_key, group_files_by_simulation


def read_ic_file(fname: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read isotropy_coeff_*.dat
    Expected columns:
      k, E11, E22, E33, dE11/dk, IC_standard, IC_deriv
    Uses IC_deriv if present.
    """
    data = np.loadtxt(fname, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    k = data[:, 0]
    E11 = data[:, 1]
    E22 = data[:, 2]

    if data.shape[1] >= 7:
        IC = data[:, 6]  # IC_deriv
    else:
        IC = np.divide(E22, E11, out=np.zeros_like(E22), where=E11 != 0)

    valid = (k > 0.5) & np.isfinite(IC) & (E11 > 1e-15)
    return k[valid], IC[valid]


def time_average_ic(files: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Interpolate onto common k grid by nearest neighbor tolerance and average.
    """
    all_k, all_ic = [], []
    for f in files:
        k, ic = read_ic_file(Path(f))
        if len(k):
            all_k.append(k)
            all_ic.append(ic)

    if not all_ic:
        return None, None

    unique_k = np.unique(np.concatenate(all_k))
    avg_ic = np.zeros_like(unique_k)
    counts = np.zeros_like(unique_k)

    for i, kk in enumerate(unique_k):
        vals = []
        for kf, icf in zip(all_k, all_ic):
            idx = np.argmin(np.abs(kf - kk))
            if abs(kf[idx] - kk) < 0.1:
                vals.append(icf[idx])
        if vals:
            avg_ic[i] = np.mean(vals)
            counts[i] = len(vals)

    min_samples = max(1, len(all_ic)//2)
    mask = counts >= min_samples
    return unique_k[mask], avg_ic[mask]


def load_spectral_groups(data_dir: Path) -> Dict[str, List[str]]:
    ic_files = sorted(data_dir.glob("isotropy_coeff_*.dat"),
                      key=lambda f: natural_sort_key(str(f)))
    if not ic_files:
        return {}

    groups = group_files_by_simulation(
        [str(f) for f in ic_files],
        r"(isotropy_coeff[_\w]*\d+)_\d+\.dat"
    )
    if not groups:
        groups = {"isotropy_coeff": [str(f) for f in ic_files]}
    return groups


def build_spectral_isotropy_fig(
    data_dir: Path,
    legend_names: Optional[Dict[str, str]] = None,
    colors: Optional[List[str]] = None,
    show_ref: bool = True
) -> go.Figure:
    """
    Main entry for Streamlit.
    Returns a Plotly figure of IC(k) time-averaged per group.
    """
    legend_names = legend_names or {}
    colors = colors or ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                        "#8c564b", "#e377c2", "#7f7f7f"]

    groups = load_spectral_groups(data_dir)
    if not groups:
        raise FileNotFoundError("No isotropy_coeff_*.dat found.")

    fig = go.Figure()
    plotted_any = False

    for i, (prefix, files) in enumerate(sorted(groups.items())):
        k_avg, ic_avg = time_average_ic(files)
        if k_avg is None:
            continue

        name = legend_names.get(prefix, prefix.replace("_", " "))
        color = colors[i % len(colors)]
        plotted_any = True

        fig.add_trace(go.Scatter(
            x=k_avg, y=ic_avg,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate="k=%{x:.3g}<br>IC=%{y:.3g}<extra></extra>"
        ))

    if not plotted_any:
        raise RuntimeError("No valid spectral isotropy data could be plotted.")

    if show_ref:
        fig.add_hline(y=1.0, line_dash="dash", line_color="black",
                      annotation_text="Isotropic IC=1", annotation_position="top right")

    fig.update_layout(
        xaxis_title="Wavenumber k",
        yaxis_title="Isotropy coefficient IC(k)",
        xaxis_type="log",
        height=520,
        legend_title="Spectral isotropy",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# ==========================================================
# Standalone run (quick check)
# ==========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spectral isotropy quick plot check.")
    parser.add_argument("--data-dir", type=str, default=".", help="Folder with isotropy_coeff_*.dat")
    args = parser.parse_args()

    fig = build_spectral_isotropy_fig(Path(args.data_dir))
    out = Path(args.data_dir) / "spectral_isotropy_IC.html"
    fig.write_html(str(out))
    print(f"Saved quick check: {out}")
