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
from pathlib import Path
from typing import Dict, Optional


def read_hdf5_file(filepath: str) -> Dict:
    """
    Read HDF5 file containing 3D velocity field
    
    Args:
        filepath: Path to .h5 or .hdf5 file
        
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
                # Try to find any 4D dataset (nx, ny, nz, 3) or (3, nx, ny, nz)
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data = np.array(f[key])
                        if len(data.shape) == 4 and (data.shape[3] == 3 or data.shape[0] == 3):
                            velocity = data
                            break
                else:
                    raise ValueError("Could not find velocity data in HDF5 file")
            
            # Get dimensions - handle both (nx, ny, nz, 3) and (3, nx, ny, nz) formats
            if len(velocity.shape) == 4:
                if velocity.shape[0] == 3:
                    # Format: (3, nx, ny, nz) - transpose to (nx, ny, nz, 3)
                    ncomp, nx, ny, nz = velocity.shape
                    velocity = np.transpose(velocity, (1, 2, 3, 0))
                elif velocity.shape[3] == 3:
                    # Format: (nx, ny, nz, 3)
                    nx, ny, nz, ncomp = velocity.shape
                else:
                    raise ValueError(f"Expected 3 velocity components, got shape {velocity.shape}")
                if ncomp != 3:
                    raise ValueError(f"Expected 3 velocity components, got {ncomp}")
            else:
                raise ValueError(f"Expected 4D velocity array, got shape {velocity.shape}")
            
            # Try to read explicit dimensions (optional)
            if 'dimensions' in f:
                dims = np.array(f['dimensions'])
                if len(dims) == 3:
                    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
            
            # Read metadata if available
            metadata = {}
            if 'metadata' in f:
                metadata_group = f['metadata']
                for key in metadata_group.attrs:
                    metadata[key] = metadata_group.attrs[key]
            
            # Also check root-level attributes
            for key in f.attrs:
                if key not in metadata:
                    metadata[key] = f.attrs[key]
            
            # Get variable name from metadata or default
            varname = metadata.get('varname', metadata.get('name', 'Velocity'))
            
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
    """
    Read HDF5 file where velocity is stored in Fortran order (column-major)
    This is useful if Fortran writes data in its native order
    
    Args:
        filepath: Path to .h5 or .hdf5 file
        
    Returns:
        Same as read_hdf5_file, but with velocity transposed to match Python/C order
    """
    data = read_hdf5_file(filepath)
    # If data was written in Fortran order, we may need to transpose
    # This depends on how Fortran writes the data
    # For now, assume standard read works, but this function can be customized
    return data


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
    
    # Compute gradients using central differences
    dudy = np.gradient(uy, dy, axis=1)
    dudz = np.gradient(uy, dz, axis=2)
    dvdx = np.gradient(ux, dx, axis=0)
    dvdz = np.gradient(uz, dz, axis=2)
    dwdx = np.gradient(uz, dx, axis=0)
    dwdy = np.gradient(uz, dy, axis=1)
    
    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy
    
    vorticity = np.zeros_like(velocity)
    vorticity[:, :, :, 0] = omega_x
    vorticity[:, :, :, 1] = omega_y
    vorticity[:, :, :, 2] = omega_z
    
    return vorticity

