"""
HDF5 file reader for 3D velocity fields
Reads HDF5 files containing velocity data from Fortran solver

Expected HDF5 structure:
- /velocity: (nx, ny, nz, 3) or (3, nx, ny, nz) array of velocity components (ux, uy, uz)
- /dimensions: (3,) array [nx, ny, nz] (optional, can be inferred from velocity shape)
- /metadata: Group with optional attributes (timestep, iteration, etc.)
"""

import numpy as np
import h5py
from typing import Dict


def read_hdf5_file(filepath: str, fortran_order: bool = True) -> Dict:
    """
    Read HDF5 file containing 3D velocity field
    
    Args:
        filepath: Path to .h5 or .hdf5 file
        fortran_order: If True, apply transpose for Fortran-written HDF5 (h5py reads
            reversed dims). If False, use data as-is (Python/default layout).
        
    Returns:
        Dictionary with:
        - 'dimensions': (nx, ny, nz)
        - 'velocity': (nx, ny, nz, 3) array of (ux, uy, uz)
        - 'varname': Variable name (default: 'Velocity')
        - 'nx', 'ny', 'nz': Individual dimensions
        - 'metadata': Dictionary of metadata if available
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Try to read velocity data
            if 'velocity' in f:
                velocity = np.array(f['velocity'])
            elif 'Velocity' in f:
                velocity = np.array(f['Velocity'])
            elif 'u' in f:
                velocity = np.array(f['u'])
            else:
                # Fallback: find any 4D dataset (nx,ny,nz,3) or (3,nx,ny,nz)
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data = np.array(f[key])
                        if len(data.shape) == 4 and (data.shape[3] == 3 or data.shape[0] == 3):
                            velocity = data
                            break
                else:
                    raise ValueError("Could not find velocity data in HDF5 file")
            
            # Handle (nx,ny,nz,3) and (3,nx,ny,nz) layouts
            if len(velocity.shape) == 4:
                if velocity.shape[0] == 3:
                    ncomp, nx, ny, nz = velocity.shape
                    velocity = np.transpose(velocity, (1, 2, 3, 0))
                elif velocity.shape[3] == 3:
                    nx, ny, nz, ncomp = velocity.shape
                else:
                    raise ValueError(f"Expected 3 velocity components, got shape {velocity.shape}")
                if ncomp != 3:
                    raise ValueError(f"Expected 3 velocity components, got {ncomp}")
            else:
                raise ValueError(f"Expected 4D velocity array, got shape {velocity.shape}")
            
            # Optional: override dimensions from file
            if 'dimensions' in f:
                dims = np.array(f['dimensions'])
                if len(dims) == 3:
                    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
            
            # Read metadata (group + root attributes)
            metadata = {}
            if 'metadata' in f:
                metadata_group = f['metadata']
                for key in metadata_group.attrs:
                    metadata[key] = metadata_group.attrs[key]
            for key in f.attrs:
                if key not in metadata:
                    metadata[key] = f.attrs[key]
            
            varname = metadata.get('varname', metadata.get('name', 'Velocity'))
            
            # Fortran writes velocity(l,m,n,3) with dims=[l,m,n,3]; h5py reads reversed → (3,n,m,l).
            # After transpose(1,2,3,0) we have (n,m,l,3). Transpose (2,1,0,3) → (l,m,n,3) = (nx,ny,nz,3).
            # When fortran_order=False, skip this (use Python/default layout as-is).
            if fortran_order:
                velocity = np.transpose(velocity, (2, 1, 0, 3))
            
            return {
                'dimensions': (nx, ny, nz),
                'velocity': velocity,
                'varname': varname,
                'nx': nx,
                'ny': ny,
                'nz': nz,
                'metadata': metadata
            }
            
    except Exception as e:
        raise ValueError(f"Error reading HDF5 file {filepath}: {e}")


def read_hdf5_file_fortran_order(filepath: str) -> Dict:
    """Read HDF5 file (Fortran-written). Same as read_hdf5_file with fortran_order=True."""
    return read_hdf5_file(filepath, fortran_order=True)


def compute_velocity_magnitude(velocity: np.ndarray) -> np.ndarray:
    """
    Compute velocity magnitude: |u| = √(ux² + uy² + uz²)
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        
    Returns:
        (nx, ny, nz) array of velocity magnitudes
    """
    return np.sqrt(velocity[:, :, :, 0]**2 + 
                 velocity[:, :, :, 1]**2 + 
                 velocity[:, :, :, 2]**2)


def compute_vorticity(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Compute vorticity: ω = ∇ × u
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing (default 1.0)
        
    Returns:
        (nx, ny, nz, 3) array of vorticity components (ωx, ωy, ωz)
    """
    ux = velocity[:, :, :, 0]
    uy = velocity[:, :, :, 1]
    uz = velocity[:, :, :, 2]
    
    # ωx=∂uz/∂y-∂uy/∂z, ωy=∂ux/∂z-∂uz/∂x, ωz=∂uy/∂x-∂ux/∂y
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
    """
    Compute velocity divergence: ∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
    
    For incompressible flow, divergence should be close to zero.
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing (default 1.0)
        
    Returns:
        (nx, ny, nz) array of divergence values
    """
    ux = velocity[:, :, :, 0]
    uy = velocity[:, :, :, 1]
    uz = velocity[:, :, :, 2]
    
    # Compute gradients using numpy gradient
    dux_dx = np.gradient(ux, dx, axis=0)
    duy_dy = np.gradient(uy, dy, axis=1)
    duz_dz = np.gradient(uz, dz, axis=2)
    
    # Divergence = sum of diagonal terms
    divergence = dux_dx + duy_dy + duz_dz
    
    return divergence


def compute_compressibility_metrics(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Dict:
    """
    Compute compressibility metrics from velocity field
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing (default 1.0)
        
    Returns:
        Dictionary with:
        - 'max_divergence': Maximum absolute divergence
        - 'mean_divergence': Mean divergence
        - 'rms_divergence': RMS divergence
        - 'max_relative_divergence': Maximum relative divergence (normalized by velocity magnitude)
    """
    divergence = compute_divergence(velocity, dx, dy, dz)
    velocity_mag = compute_velocity_magnitude(velocity)
    
    max_div = np.max(np.abs(divergence))
    mean_div = np.mean(divergence)
    rms_div = np.sqrt(np.mean(divergence**2))
    
    # Relative divergence (normalized by velocity magnitude)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_div = np.abs(divergence) / (velocity_mag + 1e-10)
    max_rel_div = np.max(relative_div)
    
    return {
        'max_divergence': max_div,
        'mean_divergence': mean_div,
        'rms_divergence': rms_div,
        'max_relative_divergence': max_rel_div,
        'divergence_field': divergence
    }

