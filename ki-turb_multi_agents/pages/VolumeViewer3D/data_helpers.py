"""
3D Volume Viewer — Cached readers, grid helpers, clipping, colormaps.
"""

import streamlit as st
import numpy as np
from pathlib import Path

from data_readers.vti_reader import read_vti_file
from data_readers.hdf5_reader import read_hdf5_file


@st.cache_data(show_spinner=True)
def cached_read_vti(filepath: str):
    """Cached VTI file reading for performance."""
    abs_path = str(Path(filepath).resolve())
    return read_vti_file(abs_path)


@st.cache_data(show_spinner=True)
def cached_read_hdf5(filepath: str, fortran_order: bool = True, _cache_version: str = "v2"):
    """Cached HDF5 file reading for performance.
    fortran_order: If True, apply transpose for Fortran-written HDF5.
    _cache_version: Internal parameter to invalidate cache when reader is updated.
    """
    abs_path = str(Path(filepath).resolve())
    return read_hdf5_file(abs_path, fortran_order=fortran_order)


def load_velocity_file(filepath: str):
    """Load velocity data from either VTI or HDF5 file."""
    abs_filepath = str(Path(filepath).resolve())
    filepath_lower = abs_filepath.lower()
    fortran_order = st.session_state.get('hdf5_fortran_order', True)
    if filepath_lower.endswith((".h5", ".hdf5")):
        return cached_read_hdf5(abs_filepath, fortran_order=fortran_order)
    elif filepath_lower.endswith(".vti"):
        return cached_read_vti(abs_filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {filepath}. Expected .vti, .h5, or .hdf5"
        )


def safe_minmax(a):
    """Return (vmin, vmax) for array, handling empty/invalid cases."""
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(a.min()), float(a.max())
    if vmin == vmax:
        if vmin == 0.0:
            vmax = 1.0
        else:
            vmax = vmin * (1.0 + 1e-6) if vmin > 0 else vmin * (1.0 - 1e-6)
    return vmin, vmax


def downsample3d(field, step):
    """Downsample 3D field by step."""
    if step <= 1:
        return field
    return field[::step, ::step, ::step]


def downsample_vectors(velocity, step):
    """Downsample vector field."""
    if step <= 1:
        return velocity
    return velocity[::step, ::step, ::step, :]


def make_grid(nx, ny, nz):
    """Create coordinate grids for 3D visualization."""
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    return x, y, z


def apply_clip(field, xmin, xmax, ymin, ymax, zmin, zmax):
    """Apply clipping box to field (set outside region to NaN)."""
    clipped = field.copy()
    mask = np.ones_like(clipped, dtype=bool)
    mask &= np.arange(clipped.shape[0])[:, None, None] >= xmin
    mask &= np.arange(clipped.shape[0])[:, None, None] <= xmax
    mask &= np.arange(clipped.shape[1])[None, :, None] >= ymin
    mask &= np.arange(clipped.shape[1])[None, :, None] <= ymax
    mask &= np.arange(clipped.shape[2])[None, None, :] >= zmin
    mask &= np.arange(clipped.shape[2])[None, None, :] <= zmax
    clipped[~mask] = np.nan
    return clipped


def colormap_options():
    """Return list of available colormap names."""
    return [
        "viridis", "cividis", "plasma", "magma", "inferno",
        "turbo", "rainbow", "jet", "portland", "rdbu",
        "spectral", "ice", "electric", "hot", "icefire",
        "greys", "ylorrd", "blues", "reds", "greens",
    ]

