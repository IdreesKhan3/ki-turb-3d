"""
VTI file reader for 3D velocity fields
Reads VTK ImageData (.vti) files written by bin_for_vec_field subroutine
"""

import numpy as np
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional


def read_vti_file(filepath: str) -> Dict:
    """
    Read VTI file (VTK ImageData format) written by bin_for_vec_field
    
    Format matches Fortran code (lines 9857-9968):
    - XML header with ImageData structure
    - Appended binary data: nbyte (int) + velocity data (Float64, 3 components)
    - Fortran ordering: x fastest, then y, then z
    
    Args:
        filepath: Path to .vti file
        
    Returns:
        Dictionary with:
        - 'dimensions': (nx, ny, nz)
        - 'velocity': (nx, ny, nz, 3) array of (ux, uy, uz)
        - 'varname': Variable name from file
    """
    try:
        with open(filepath, 'rb') as f:
            # Read until we find the XML header
            content = f.read()
            
            # Find XML section (before AppendedData)
            xml_end = content.find(b'<AppendedData')
            if xml_end == -1:
                raise ValueError("Could not find AppendedData section")
            
            xml_content = content[:xml_end].decode('utf-8', errors='ignore')
            
            # Parse XML to get dimensions and variable name
            root = ET.fromstring(xml_content)
            imagedata = root.find('.//ImageData')
            if imagedata is None:
                raise ValueError("Could not find ImageData element")
            
            # Extract WholeExtent
            whole_extent = imagedata.get('WholeExtent', '')
            extents = [int(x) for x in whole_extent.split()]
            nx = extents[1] - extents[0] + 1
            ny = extents[3] - extents[2] + 1
            nz = extents[5] - extents[4] + 1
            
            # Extract variable name
            data_array = root.find('.//DataArray')
            varname = data_array.get('Name', 'Velocity') if data_array is not None else 'Velocity'
            
            # Find AppendedData section
            appended_start = content.find(b'<AppendedData')
            if appended_start == -1:
                raise ValueError("Could not find AppendedData section")
            
            # Find the '_' marker after AppendedData tag
            marker_pos = content.find(b'_', appended_start)
            if marker_pos == -1:
                raise ValueError("Could not find data marker '_'")
            
            # Read data starting after '_'
            f.seek(marker_pos + 1)
            
            # Read nbyte (4 bytes, integer)
            nbyte_bytes = f.read(4)
            nbyte = struct.unpack('<i', nbyte_bytes)[0]  # Little-endian integer
            
            # Verify nbyte matches expected size
            expected_nbyte = 3 * nx * ny * nz * 8  # 3 components * dimensions * 8 bytes (Float64)
            if nbyte != expected_nbyte:
                print(f"Warning: nbyte mismatch. Expected {expected_nbyte}, got {nbyte}")
            
            # Read velocity data (Float64, 3 components per point)
            # Data is in Fortran order: (xi=1,nx, yi=1,ny, zi=1,nz)
            # So we read all data and reshape
            data = np.frombuffer(f.read(nbyte), dtype=np.float64)
            
            # Reshape to (nx, ny, nz, 3) - Fortran order
            # Data is stored as: (x1,y1,z1), (x2,y1,z1), ..., (xn,y1,z1), (x1,y2,z1), ...
            velocity = data.reshape((nx, ny, nz, 3), order='F')
            
            return {
                'dimensions': (nx, ny, nz),
                'velocity': velocity,
                'varname': varname,
                'nx': nx,
                'ny': ny,
                'nz': nz
            }
            
    except Exception as e:
        raise ValueError(f"Error reading VTI file {filepath}: {e}")


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
    # ωx = ∂uz/∂y - ∂uy/∂z
    # ωy = ∂ux/∂z - ∂uz/∂x
    # ωz = ∂uy/∂x - ∂ux/∂y
    
    # For interior points (using central differences)
    # Note: This is a simplified version - may need boundary handling
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

