"""
PDF computations for turbulence quantities.

Velocity, vorticity, enstrophy, dissipation, joint PDFs.
Uses utils.iso_surfaces for gradient-derived quantities.
No Streamlit or UI dependencies.
"""

import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, Dict, Any

from utils.iso_surfaces import (
    compute_rotation_deformation_tensors,
    compute_vorticity_vector,
    compute_q_invariant,
    compute_r_invariant,
)


def compute_skewness_kurtosis(data: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute mean, RMS, skewness, and kurtosis."""
    data_clean = data[np.isfinite(data)]
    if len(data_clean) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = np.mean(data_clean)
    u_prime = data_clean - mean
    u2_mean = np.mean(u_prime**2)
    u3_mean = np.mean(u_prime**3)
    u4_mean = np.mean(u_prime**4)
    rms = np.sqrt(u2_mean) if u2_mean > 0 else 0.0
    skewness = u3_mean / (u2_mean**(3/2)) if u2_mean > 0 else 0.0
    kurtosis = u4_mean / (u2_mean**2) if u2_mean > 0 else 0.0
    return mean, rms, skewness, kurtosis


def _pdf_kde(data_flat: np.ndarray, bins: int, normalize: bool,
             norm_by_mean: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Common KDE-based PDF logic."""
    if len(data_flat) == 0:
        return np.array([]), np.array([])
    normalization_factor = 1.0
    if normalize:
        if norm_by_mean:
            m = np.mean(data_flat)
            if m > 0:
                data_flat = data_flat / m
                normalization_factor = m
        else:
            rms = np.sqrt(np.mean(data_flat**2))
            if rms > 0:
                data_flat = data_flat / rms
                normalization_factor = rms
    v_min, v_max = data_flat.min(), data_flat.max()
    v_range = v_max - v_min
    v_min -= 0.1 * v_range
    v_max += 0.1 * v_range
    grid = np.linspace(v_min, v_max, bins)
    try:
        kde = gaussian_kde(data_flat)
        pdf = kde(grid)
    except Exception:
        counts, edges = np.histogram(data_flat, bins=bins, range=(v_min, v_max), density=True)
        pdf = counts
        grid = (edges[:-1] + edges[1:]) / 2
    if normalize and normalization_factor > 0:
        pdf = pdf * normalization_factor
    return grid, pdf


def compute_velocity_magnitude_pdf(velocity: np.ndarray, bins: int = 100,
                                   normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smooth PDF for velocity magnitude using KDE."""
    u_mag = np.sqrt(
        velocity[:, :, :, 0]**2 + velocity[:, :, :, 1]**2 + velocity[:, :, :, 2]**2
    )
    u_flat = u_mag.flatten()[np.isfinite(u_mag.flatten())]
    return _pdf_kde(u_flat, bins, normalize, norm_by_mean=False)


def compute_velocity_magnitude_statistics(velocity: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute statistics for velocity magnitude."""
    u_mag = np.sqrt(
        velocity[:, :, :, 0]**2 + velocity[:, :, :, 1]**2 + velocity[:, :, :, 2]**2
    )
    return compute_skewness_kurtosis(u_mag.flatten())


def compute_velocity_pdf(velocity: np.ndarray, bins: int = 100,
                        normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute smooth PDFs for velocity components using KDE."""
    ux = velocity[:, :, :, 0].flatten()[np.isfinite(velocity[:, :, :, 0].flatten())]
    uy = velocity[:, :, :, 1].flatten()[np.isfinite(velocity[:, :, :, 1].flatten())]
    uz = velocity[:, :, :, 2].flatten()[np.isfinite(velocity[:, :, :, 2].flatten())]
    if len(ux) == 0 or len(uy) == 0 or len(uz) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    normalization_factor = 1.0
    if normalize:
        all_u = np.concatenate([ux, uy, uz])
        rms_u = np.sqrt(np.mean(all_u**2))
        if rms_u > 0:
            ux, uy, uz = ux / rms_u, uy / rms_u, uz / rms_u
            normalization_factor = rms_u
    u_min = min(ux.min(), uy.min(), uz.min())
    u_max = max(ux.max(), uy.max(), uz.max())
    u_range = u_max - u_min
    u_min -= 0.1 * u_range
    u_max += 0.1 * u_range
    u_grid = np.linspace(u_min, u_max, bins)
    def _kde_or_hist(d):
        try:
            return gaussian_kde(d)(u_grid)
        except Exception:
            c, e = np.histogram(d, bins=bins, range=(u_min, u_max), density=True)
            return c
    pdf_u = _kde_or_hist(ux)
    pdf_v = _kde_or_hist(uy)
    pdf_w = _kde_or_hist(uz)
    if normalize and normalization_factor > 0:
        pdf_u = pdf_u * normalization_factor
        pdf_v = pdf_v * normalization_factor
        pdf_w = pdf_w * normalization_factor
    return u_grid, pdf_u, pdf_v, pdf_w


def compute_velocity_component_statistics(velocity: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
    """Compute statistics for each velocity component."""
    return {
        'u': compute_skewness_kurtosis(velocity[:, :, :, 0].flatten()),
        'v': compute_skewness_kurtosis(velocity[:, :, :, 1].flatten()),
        'w': compute_skewness_kurtosis(velocity[:, :, :, 2].flatten()),
    }


def compute_dissipation_pdf(velocity: np.ndarray, nu: float = 1.0, bins: int = 100,
                            dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                            normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smooth PDF for dissipation rate epsilon = 2 nu S_ij S_ij."""
    _, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    S_squared_sum = np.einsum('ijklm,ijklm->ijk', S, S)
    dissipation = 2.0 * nu * S_squared_sum
    eps_flat = dissipation.flatten()
    eps_flat = eps_flat[np.isfinite(eps_flat)]
    eps_flat = eps_flat[eps_flat >= 0]
    if len(eps_flat) == 0:
        return np.array([]), np.array([])
    return _pdf_kde(eps_flat, bins, normalize, norm_by_mean=True)


def compute_dissipation_statistics(velocity: np.ndarray, nu: float = 1.0,
                                   dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Tuple[float, float, float, float]:
    """Compute statistical moments for dissipation rate."""
    _, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    S_squared_sum = np.einsum('ijklm,ijklm->ijk', S, S)
    dissipation = 2.0 * nu * S_squared_sum
    eps_flat = dissipation.flatten()[np.isfinite(dissipation.flatten())]
    eps_flat = eps_flat[eps_flat >= 0]
    return compute_skewness_kurtosis(eps_flat)


def compute_vorticity_pdf(velocity: np.ndarray, bins: int = 100,
                          dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                          normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smooth PDF for vorticity magnitude."""
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    omega_mag = np.sqrt(
        vorticity[:, :, :, 0]**2 + vorticity[:, :, :, 1]**2 + vorticity[:, :, :, 2]**2
    )
    omega_flat = omega_mag.flatten()[np.isfinite(omega_mag.flatten())]
    return _pdf_kde(omega_flat, bins, normalize, norm_by_mean=False)


def compute_vorticity_statistics(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Tuple[float, float, float, float]:
    """Compute statistics for vorticity magnitude."""
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    omega_mag = np.sqrt(
        vorticity[:, :, :, 0]**2 + vorticity[:, :, :, 1]**2 + vorticity[:, :, :, 2]**2
    )
    return compute_skewness_kurtosis(omega_mag.flatten())


def compute_enstrophy_pdf(velocity: np.ndarray, bins: int = 100,
                          dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                          normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smooth PDF for enstrophy Omega = |omega|^2."""
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    enstrophy = (
        vorticity[:, :, :, 0]**2 + vorticity[:, :, :, 1]**2 + vorticity[:, :, :, 2]**2
    )
    enstrophy_flat = enstrophy.flatten()[np.isfinite(enstrophy.flatten())]
    return _pdf_kde(enstrophy_flat, bins, normalize, norm_by_mean=True)


def compute_enstrophy_statistics(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Tuple[float, float, float, float]:
    """Compute statistics for enstrophy."""
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    enstrophy = (
        vorticity[:, :, :, 0]**2 + vorticity[:, :, :, 1]**2 + vorticity[:, :, :, 2]**2
    )
    return compute_skewness_kurtosis(enstrophy.flatten())


def compute_velocity_dissipation_joint_pdf(velocity: np.ndarray, nu: float = 1.0, bins: int = 100,
                                            dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                                            u_range: Optional[Tuple[float, float]] = None,
                                            eps_range: Optional[Tuple[float, float]] = None,
                                            normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute joint PDF P(|u|, epsilon)."""
    u_mag = np.sqrt(velocity[:, :, :, 0]**2 + velocity[:, :, :, 1]**2 + velocity[:, :, :, 2]**2)
    _, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    dissipation = 2.0 * nu * np.einsum('ijklm,ijklm->ijk', S, S)
    u_flat = u_mag.flatten()
    eps_flat = dissipation.flatten()
    valid = np.isfinite(u_flat) & np.isfinite(eps_flat) & (eps_flat >= 0)
    u_flat, eps_flat = u_flat[valid], eps_flat[valid]
    if len(u_flat) == 0:
        return np.array([]), np.array([]), np.array([])
    norm_u, norm_eps = 1.0, 1.0
    if normalize:
        rms_u = np.sqrt(np.mean(u_flat**2))
        if rms_u > 0:
            u_flat = u_flat / rms_u
            norm_u = rms_u
        mean_eps = np.mean(eps_flat)
        if mean_eps > 0:
            eps_flat = eps_flat / mean_eps
            norm_eps = mean_eps
    u_range = u_range or (u_flat.min(), u_flat.max())
    eps_range = eps_range or (eps_flat.min(), eps_flat.max())
    joint_hist, u_edges, eps_edges = np.histogram2d(u_flat, eps_flat, bins=[bins, bins], range=[u_range, eps_range], density=False)
    bin_area = (u_edges[1] - u_edges[0]) * (eps_edges[1] - eps_edges[0])
    joint_pdf = joint_hist / (len(u_flat) * bin_area)
    if normalize:
        joint_pdf = joint_pdf * norm_u * norm_eps
    u_centers = (u_edges[:-1] + u_edges[1:]) / 2
    eps_centers = (eps_edges[:-1] + eps_edges[1:]) / 2
    return u_centers, eps_centers, joint_pdf.T


def compute_velocity_enstrophy_joint_pdf(velocity: np.ndarray, bins: int = 100,
                                          dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                                          u_range: Optional[Tuple[float, float]] = None,
                                          omega_range: Optional[Tuple[float, float]] = None,
                                          normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute joint PDF P(|u|, |omega|)."""
    u_mag = np.sqrt(velocity[:, :, :, 0]**2 + velocity[:, :, :, 1]**2 + velocity[:, :, :, 2]**2)
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    omega_mag = np.sqrt(vorticity[:, :, :, 0]**2 + vorticity[:, :, :, 1]**2 + vorticity[:, :, :, 2]**2)
    u_flat = u_mag.flatten()[np.isfinite(u_mag.flatten())]
    omega_flat = omega_mag.flatten()[np.isfinite(omega_mag.flatten())]
    valid = np.isfinite(u_flat) & np.isfinite(omega_flat)
    u_flat, omega_flat = u_flat[valid], omega_flat[valid]
    if len(u_flat) == 0:
        return np.array([]), np.array([]), np.array([])
    u_range = u_range or (u_flat.min(), u_flat.max())
    omega_range = omega_range or (omega_flat.min(), omega_flat.max())
    joint_hist, u_edges, omega_edges = np.histogram2d(u_flat, omega_flat, bins=[bins, bins], range=[u_range, omega_range], density=False)
    bin_area = (u_edges[1] - u_edges[0]) * (omega_edges[1] - omega_edges[0])
    joint_pdf = joint_hist / (len(u_flat) * bin_area)
    u_centers = (u_edges[:-1] + u_edges[1:]) / 2
    omega_centers = (omega_edges[:-1] + omega_edges[1:]) / 2
    return u_centers, omega_centers, joint_pdf.T


def compute_dissipation_enstrophy_joint_pdf(velocity: np.ndarray, nu: float = 1.0, bins: int = 100,
                                             dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                                             eps_range: Optional[Tuple[float, float]] = None,
                                             omega_range: Optional[Tuple[float, float]] = None,
                                             normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute joint PDF P(epsilon, |omega|)."""
    _, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    dissipation = 2.0 * nu * np.einsum('ijklm,ijklm->ijk', S, S)
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    omega_mag = np.sqrt(vorticity[:, :, :, 0]**2 + vorticity[:, :, :, 1]**2 + vorticity[:, :, :, 2]**2)
    eps_flat = dissipation.flatten()
    omega_flat = omega_mag.flatten()
    valid = np.isfinite(eps_flat) & np.isfinite(omega_flat) & (eps_flat >= 0)
    eps_flat, omega_flat = eps_flat[valid], omega_flat[valid]
    if len(eps_flat) == 0:
        return np.array([]), np.array([]), np.array([])
    norm_eps, norm_omega = 1.0, 1.0
    if normalize:
        mean_eps = np.mean(eps_flat)
        if mean_eps > 0:
            eps_flat = eps_flat / mean_eps
            norm_eps = mean_eps
        rms_omega = np.sqrt(np.mean(omega_flat**2))
        if rms_omega > 0:
            omega_flat = omega_flat / rms_omega
            norm_omega = rms_omega
    eps_range = eps_range or (eps_flat.min(), eps_flat.max())
    omega_range = omega_range or (omega_flat.min(), omega_flat.max())
    joint_hist, eps_edges, omega_edges = np.histogram2d(eps_flat, omega_flat, bins=[bins, bins], range=[eps_range, omega_range], density=False)
    bin_area = (eps_edges[1] - eps_edges[0]) * (omega_edges[1] - omega_edges[0])
    joint_pdf = joint_hist / (len(eps_flat) * bin_area)
    if normalize:
        joint_pdf = joint_pdf * norm_eps * norm_omega
    eps_centers = (eps_edges[:-1] + eps_edges[1:]) / 2
    omega_centers = (omega_edges[:-1] + omega_edges[1:]) / 2
    return eps_centers, omega_centers, joint_pdf.T


def compute_rq_joint_pdf(velocity: np.ndarray, r_bins: int = 100, q_bins: int = 100,
                         r_range: Optional[Tuple[float, float]] = None,
                         q_range: Optional[Tuple[float, float]] = None,
                         dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute joint PDF of Q and R invariants, normalized by <S_ij S_ij>."""
    Q = compute_q_invariant(velocity, dx, dy, dz)
    R = compute_r_invariant(velocity, dx, dy, dz)
    _, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    S_squared_sum = np.einsum('ijklm,ijklm->ijk', S, S)
    valid_S = S_squared_sum[np.isfinite(S_squared_sum)]
    mean_S_squared = np.mean(valid_S) if len(valid_S) > 0 else 1.0
    if mean_S_squared > 0:
        Q_normalized = Q / mean_S_squared
        R_normalized = R / (mean_S_squared ** 1.5)
    else:
        Q_normalized, R_normalized = Q, R
    Q_flat = Q_normalized.flatten()[np.isfinite(Q_normalized.flatten())]
    R_flat = R_normalized.flatten()[np.isfinite(R_normalized.flatten())]
    valid = np.isfinite(Q_flat) & np.isfinite(R_flat)
    Q_flat, R_flat = Q_flat[valid], R_flat[valid]
    if len(Q_flat) == 0:
        return np.array([]), np.array([]), np.array([])
    r_range = r_range or (R_flat.min(), R_flat.max())
    q_range = q_range or (Q_flat.min(), Q_flat.max())
    joint_hist, r_edges, q_edges = np.histogram2d(R_flat, Q_flat, bins=[r_bins, q_bins], range=[r_range, q_range], density=False)
    bin_area = (r_edges[1] - r_edges[0]) * (q_edges[1] - q_edges[0])
    joint_pdf = joint_hist / (len(R_flat) * bin_area)
    R_centers = (r_edges[:-1] + r_edges[1:]) / 2
    Q_centers = (q_edges[:-1] + q_edges[1:]) / 2
    return R_centers, Q_centers, joint_pdf.T


def compute_discriminant_line(r_values: np.ndarray) -> np.ndarray:
    """Compute Q values for D=0 line: Q = -3*(R/2)^(2/3)."""
    return -3 * np.power(np.abs(r_values) / 2.0, 2.0/3.0)
