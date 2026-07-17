"""
VTI file reader for 3D velocity fields.
Reads VTK ImageData (.vti) written by bin_for_vec_field.
Falls back to PyVista for standard VTK .vti (e.g., from OpenLB, Palabos, foamToVTK).
"""

import logging
import struct
import xml.etree.ElementTree as ET
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def _read_vti_pyvista(filepath: str) -> Dict:
    """
    Fallback: read standard VTK ImageData via PyVista.
    Supports .vti from OpenLB, Palabos, OpenFOAM foamToVTK, etc.
    Returns same dict format as read_vti_file.
    """
    import pyvista as pv
    mesh = pv.read(filepath)
    if not isinstance(mesh, pv.ImageData):
        raise ValueError(f"PyVista fallback: expected ImageData, got {type(mesh).__name__}")

    dims = mesh.dimensions  # (nx, ny, nz) = grid point dimensions
    n_points = int(dims[0]) * int(dims[1]) * int(dims[2])
    n_cells = max(1, int(dims[0]) - 1) * max(1, int(dims[1]) - 1) * max(1, int(dims[2]) - 1)

    # Prefer point data (common for LBM); fallback to cell data
    velocity_arr = None
    varname = None
    use_cell_data = False

    for attr_name in ('point_data', 'cell_data'):
        attrs = getattr(mesh, attr_name, None)
        if attrs is None:
            continue
        for key in list(attrs.keys()):
            arr = np.asarray(attrs[key])
            if arr.ndim != 2 or arr.shape[1] != 3:
                continue
            n = arr.shape[0]
            if n == n_points:
                velocity_arr = arr
                varname = key
                use_cell_data = False
                break
            if n == n_cells:
                velocity_arr = arr
                varname = key
                use_cell_data = True
                break
        if velocity_arr is not None:
            break

    if velocity_arr is None:
        raise ValueError(
            "PyVista fallback: no 3-component velocity array found in point_data or cell_data"
        )

    if use_cell_data:
        nx, ny, nz = int(dims[0]) - 1, int(dims[1]) - 1, int(dims[2]) - 1
    else:
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

    # Reshape to (nx, ny, nz, 3); VTK uses Fortran-like order (x fastest)
    velocity = velocity_arr.reshape((nx, ny, nz, 3), order='F')
    return {
        'dimensions': (nx, ny, nz),
        'velocity': np.ascontiguousarray(velocity),
        'varname': varname or 'Velocity',
        'nx': nx,
        'ny': ny,
        'nz': nz
    }


def read_vti_file(filepath: str) -> Dict:
    """
    Read VTI (VTK ImageData): XML header + appended binary (nbyte + Float64 velocity).
    Fortran order: x fastest, then y, then z.
    Falls back to PyVista for standard VTK .vti (OpenLB, Palabos, foamToVTK).
    Returns: dimensions, velocity (nx,ny,nz,3), varname.
    """
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            # Parse XML header (before AppendedData)
            xml_end = content.find(b'<AppendedData')
            if xml_end == -1:
                raise ValueError("Could not find AppendedData section")
            xml_content = content[:xml_end].decode('utf-8', errors='ignore').rstrip() + '\n</VTKFile>'
            root = ET.fromstring(xml_content)
            # Extract dimensions and varname from WholeExtent
            imagedata = root.find('.//ImageData')
            if imagedata is None:
                raise ValueError("Could not find ImageData element")
            whole_extent = imagedata.get('WholeExtent', '')
            extents = [int(x) for x in whole_extent.split()]
            nx = extents[1] - extents[0] + 1
            ny = extents[3] - extents[2] + 1
            nz = extents[5] - extents[4] + 1
            data_array = root.find('.//DataArray')
            varname = data_array.get('Name', 'Velocity') if data_array is not None else 'Velocity'
            # Locate binary data (after '_' marker)
            appended_start = content.find(b'<AppendedData')
            if appended_start == -1:
                raise ValueError("Could not find AppendedData section")
            marker_pos = content.find(b'_', appended_start)
            if marker_pos == -1:
                raise ValueError("Could not find data marker '_'")
            f.seek(marker_pos + 1)
            # Read nbyte (4B int) + velocity block
            nbyte_bytes = f.read(4)
            nbyte = struct.unpack('<i', nbyte_bytes)[0]
            # Auto-detect float64 vs float32 from size
            expected_nbyte_float64 = 3 * nx * ny * nz * 8
            expected_nbyte_float32 = 3 * nx * ny * nz * 4
            
            if nbyte == expected_nbyte_float64:
                dtype = np.float64
            elif nbyte == expected_nbyte_float32:
                dtype = np.float32
            else:
                dtype = np.float64
                logger.warning(
                    "nbyte mismatch. Expected %s or %s, got %s. Using float64.",
                    expected_nbyte_float64, expected_nbyte_float32, nbyte,
                )
            data = np.frombuffer(f.read(nbyte), dtype=dtype)
            velocity_flat = data.reshape((nx * ny * nz, 3), order='C')
            velocity = np.zeros((nx, ny, nz, 3), dtype=dtype)
            for zi in range(nz):
                for yi in range(ny):
                    for xi in range(nx):
                        flat_idx = zi * nx * ny + yi * nx + xi
                        velocity[xi, yi, zi, :] = velocity_flat[flat_idx, :]
            
            return {
                'dimensions': (nx, ny, nz),
                'velocity': velocity,
                'varname': varname,
                'nx': nx,
                'ny': ny,
                'nz': nz
            }
            
    except Exception as e:
        logger.debug("Custom VTI reader failed for %s: %s. Trying PyVista fallback.", filepath, e)
        try:
            return _read_vti_pyvista(filepath)
        except Exception as fallback_err:
            raise ValueError(
                f"Error reading VTI file {filepath}. Custom reader: {e}. PyVista fallback: {fallback_err}"
            ) from fallback_err


def compute_velocity_magnitude(velocity: np.ndarray) -> np.ndarray:
    """|u| = √(ux² + uy² + uz²). Returns (nx,ny,nz)."""
    return np.sqrt(velocity[:, :, :, 0]**2 + 
                   velocity[:, :, :, 1]**2 + 
                   velocity[:, :, :, 2]**2)


def compute_vorticity(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """ω = ∇ × u. Returns (nx,ny,nz,3)."""
    # ωx=∂uz/∂y-∂uy/∂z, ωy=∂ux/∂z-∂uz/∂x, ωz=∂uy/∂x-∂ux/∂y
    ux = velocity[:, :, :, 0]
    uy = velocity[:, :, :, 1]
    uz = velocity[:, :, :, 2]
    dux_dy = np.gradient(ux, dy, axis=1)
    dux_dz = np.gradient(ux, dz, axis=2)
    duy_dx = np.gradient(uy, dx, axis=0)
    duy_dz = np.gradient(uy, dz, axis=2)
    duz_dx = np.gradient(uz, dx, axis=0)
    duz_dy = np.gradient(uz, dy, axis=1)
    
    omega_x = duz_dy - duy_dz
    omega_y = dux_dz - duz_dx
    omega_z = duy_dx - dux_dy
    
    vorticity = np.zeros_like(velocity)
    vorticity[:, :, :, 0] = omega_x
    vorticity[:, :, :, 1] = omega_y
    vorticity[:, :, :, 2] = omega_z
    
    return vorticity


def compute_divergence(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z. Returns (nx,ny,nz)."""
    ux = velocity[:, :, :, 0]
    uy = velocity[:, :, :, 1]
    uz = velocity[:, :, :, 2]
    dux_dx = np.gradient(ux, dx, axis=0)
    duy_dy = np.gradient(uy, dy, axis=1)
    duz_dz = np.gradient(uz, dz, axis=2)
    divergence = dux_dx + duy_dy + duz_dz
    
    return divergence


def compute_compressibility_metrics(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Dict:
    """Divergence metrics: max, mean, rms, max_relative."""
    divergence = compute_divergence(velocity, dx, dy, dz)
    velocity_mag = compute_velocity_magnitude(velocity)
    
    max_div = np.max(np.abs(divergence))
    mean_div = np.mean(divergence)
    rms_div = np.sqrt(np.mean(divergence**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_div = np.abs(divergence) / (velocity_mag + 1e-10)  # avoid div by zero
    max_rel_div = np.max(relative_div)
    
    return {
        'max_divergence': max_div,
        'mean_divergence': mean_div,
        'rms_divergence': rms_div,
        'max_relative_divergence': max_rel_div,
        'divergence_field': divergence
    }

