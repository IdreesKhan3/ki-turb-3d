"""
3D Volume Viewer — Cached readers, grid helpers, clipping, colormaps.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from data_readers.vti_reader import read_vti_file
from data_readers.hdf5_reader import read_hdf5_file


@st.cache_data(show_spinner=True)
def cached_read_vti(filepath: str):
    """Cached VTI file reading for performance."""
    abs_path = str(Path(filepath).resolve())
    return read_vti_file(abs_path)


@st.cache_data(show_spinner=True)
def cached_read_hdf5(filepath: str, _cache_version: str = "v2"):
    """Cached HDF5 file reading for performance.
    _cache_version: Internal parameter to invalidate cache when reader is updated.
    """
    abs_path = str(Path(filepath).resolve())
    return read_hdf5_file(abs_path)


def load_velocity_file(filepath: str):
    """Load velocity data from either VTI or HDF5 file."""
    abs_filepath = str(Path(filepath).resolve())
    filepath_lower = abs_filepath.lower()
    if filepath_lower.endswith((".h5", ".hdf5")):
        return cached_read_hdf5(abs_filepath)
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


def create_slice_surface(x_coords, y_coords, z_coords, field_slice, vmin, vmax, cmap, opacity):
    """Create a surface trace for a slice plane.
    Matches ParaView's coordinate system: X horizontal, Y vertical, Z out-of-plane.
    """
    return go.Surface(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        surfacecolor=np.nan_to_num(field_slice, nan=np.nan),
        cmin=vmin,
        cmax=vmax,
        colorscale=cmap,
        opacity=opacity,
        showscale=False,
        hovertemplate="Value: %{surfacecolor:.4f}<extra></extra>",
        connectgaps=False,
    )
