"""FFT helpers with Parseval normalization and physical / discrete wave numbers."""
from __future__ import annotations

from typing import Dict

import numpy as np


def fluctuations(v):
    a = np.asarray(v, float)
    m = np.mean(a, axis=(0, 1, 2), keepdims=True)
    u = a - m
    return u[..., 0], u[..., 1], u[..., 2]


def wavevector_grid(shape, spacing=1.0):
    if np.isscalar(spacing):
        spacing = (float(spacing),) * 3
    ks = [2 * np.pi * np.fft.fftfreq(n, d=d) for n, d in zip(shape, spacing)]
    return np.meshgrid(*ks, indexing="ij")


def wavenumber_magnitude(shape, spacing=1.0):
    K = wavevector_grid(shape, spacing)
    return np.sqrt(sum(x * x for x in K))


def component_spectra(velocity, spacing=1.0, *, truncate_trustworthy=False):
    ux, uy, uz = fluctuations(velocity)
    n = ux.size
    h = [np.fft.fftn(q) for q in (ux, uy, uz)]
    energies = [0.5 * np.abs(q) ** 2 / n**2 for q in h]
    km = wavenumber_magnitude(ux.shape, spacing)
    if np.isscalar(spacing):
        spacing = (float(spacing),) * 3
    dk = min(2 * np.pi / (nn * d) for nn, d in zip(ux.shape, spacing))
    bins = np.floor(km / dk + 0.5).astype(int)
    maxbin = int(bins.max())
    trust_k = min(np.pi / d for d in spacing)
    trust_bin = int(np.floor(trust_k / dk))
    if truncate_trustworthy:
        maxbin = trust_bin

    def shell(e):
        out = np.zeros(maxbin + 1)
        valid = bins <= maxbin
        np.add.at(out, bins[valid], e[valid])
        return out

    return {
        "k": np.arange(maxbin + 1) * dk,
        "E": shell(sum(energies)),
        "E11": shell(energies[0]),
        "E22": shell(energies[1]),
        "E33": shell(energies[2]),
        "shell_count": np.bincount(bins.ravel(), minlength=maxbin + 1)[: maxbin + 1],
        "trustworthy_k_max": trust_k,
        "normalization": "sum(E)=0.5*mean(|u-prime|^2)",
    }


def component_spectra_discrete(velocity) -> Dict[str, np.ndarray]:
    """
    Discrete-shell spectra for isotropy_coeff products.

    - Discrete mode indices kx,ky,kz (fftfreq * N)
    - Skip Nyquist-aliased modes (|kx| > N/2, …)
    - Skip k_mag < 0.5
    - bin = nint(k_mag), k_values = bin (integer centers)
    - Shell-integrated sums (not divided by mode count)
    """
    ux, uy, uz = fluctuations(velocity)
    nx, ny, nz = ux.shape
    n = ux.size
    energies = [0.5 * np.abs(np.fft.fftn(q)) ** 2 / n**2 for q in (ux, uy, uz)]

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(KX * KX + KY * KY + KZ * KZ)

    # Skip Nyquist-aliased modes and near-zero shells
    nyqx, nyqy, nyqz = nx / 2.0, ny / 2.0, nz / 2.0
    mask = (
        (np.abs(KX) <= nyqx)
        & (np.abs(KY) <= nyqy)
        & (np.abs(KZ) <= nyqz)
        & (k_mag >= 0.5)
    )

    nbin = int(min(nx, ny, nz) // 2)
    bins = np.rint(k_mag).astype(int)
    valid = mask & (bins >= 1) & (bins <= nbin)

    def shell(e: np.ndarray) -> np.ndarray:
        out = np.zeros(nbin + 1)
        np.add.at(out, bins[valid], e[valid])
        return out

    count = np.zeros(nbin + 1)
    np.add.at(count, bins[valid], 1.0)

    e11, e22, e33 = shell(energies[0]), shell(energies[1]), shell(energies[2])
    return {
        "k": np.arange(nbin + 1, dtype=float),
        "E11": e11,
        "E22": e22,
        "E33": e33,
        "E": e11 + e22 + e33,
        "shell_count": count,
        "nx": float(nx),
        "dk": 1.0,
    }


def dE11_dk_shell(e11: np.ndarray, count: np.ndarray, dk: float = 1.0) -> np.ndarray:
    """5-point / central / one-sided dE11/dk on discrete shells for IC_derivative."""
    nbin = len(e11) - 1
    dE = np.zeros_like(e11)
    for bin_i in range(3, nbin - 1):  # 3 .. nbin-2 inclusive in 1-based → 3..nbin-2
        if (
            count[bin_i - 2] > 0
            and count[bin_i - 1] > 0
            and count[bin_i + 1] > 0
            and count[bin_i + 2] > 0
        ):
            dE[bin_i] = (
                -e11[bin_i + 2]
                + 8.0 * e11[bin_i + 1]
                - 8.0 * e11[bin_i - 1]
                + e11[bin_i - 2]
            ) / (12.0 * dk)
        elif count[bin_i - 1] > 0 and count[bin_i + 1] > 0:
            dE[bin_i] = (e11[bin_i + 1] - e11[bin_i - 1]) / (2.0 * dk)

    # Boundaries (1-based bins 1,2 and nbin-1,nbin)
    if nbin >= 2 and count[1] > 0 and count[2] > 0:
        dE[1] = (e11[2] - e11[1]) / dk
    if nbin >= 3 and count[1] > 0 and count[3] > 0:
        dE[2] = (e11[3] - e11[1]) / (2.0 * dk)
    if nbin >= 2 and count[nbin - 1] > 0 and count[nbin] > 0:
        dE[nbin] = (e11[nbin] - e11[nbin - 1]) / dk
    if nbin >= 3 and count[nbin - 2] > 0 and count[nbin] > 0:
        dE[nbin - 1] = (e11[nbin] - e11[nbin - 2]) / (2.0 * dk)
    return dE
