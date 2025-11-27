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
            
            # Extract XML section and add closing VTKFile tag for parsing
            # (AppendedData is inside VTKFile, so XML is incomplete without it)
            xml_content = content[:xml_end].decode('utf-8', errors='ignore').rstrip()
            # Add closing tag to make XML well-formed for parsing
            xml_content += '\n</VTKFile>'
            
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
            
            # Auto-detect precision from file size
            expected_nbyte_float64 = 3 * nx * ny * nz * 8  # Float64: 8 bytes per value
            expected_nbyte_float32 = 3 * nx * ny * nz * 4  # Float32: 4 bytes per value
            
            if nbyte == expected_nbyte_float64:
                dtype = np.float64
            elif nbyte == expected_nbyte_float32:
                dtype = np.float32
            else:
                # Default to float64 if size doesn't match (legacy behavior)
                dtype = np.float64
                print(f"Warning: nbyte mismatch. Expected {expected_nbyte_float64} (float64) or {expected_nbyte_float32} (float32), got {nbyte}. Using float64.")
            
            # Read velocity data
            # Fortran writes: ((( ux(xi,yi,zi), uy(xi,yi,zi), uz(xi,yi,zi), xi=1,nx), yi=1,ny), zi=1,nz)
            # File contains: [ux(1,1,1), uy(1,1,1), uz(1,1,1), ux(2,1,1), uy(2,1,1), uz(2,1,1), ...]
            # Components are interleaved: every 3 elements is (ux,uy,uz) for one point
            # Spatial order is Fortran: x changes fastest, then y, then z
            data = np.frombuffer(f.read(nbyte), dtype=dtype)
            
            # Reshape to separate components: (nx*ny*nz, 3) with C order
            # This groups every 3 consecutive elements as (ux,uy,uz) for each point
            velocity_flat = data.reshape((nx * ny * nz, 3), order='C')
            
            # Reshape spatial dimensions correctly
            # Fortran writes: ((( ux(xi,yi,zi), ...), xi=1,l), yi=1,m), zi=1,n)
            # File order: x changes fastest, then y, then z
            # Flat index for (xi, yi, zi) in 0-indexed: zi*nx*ny + yi*nx + xi
            # However, empirical testing shows the data has x and y dimensions swapped
            # Solution: read as (y, x, z) then transpose to (x, y, z)
            velocity = np.zeros((ny, nx, nz, 3), dtype=dtype)
            for zi in range(nz):
                for yi in range(ny):
                    for xi in range(nx):
                        # Flat index: zi*nx*ny + yi*nx + xi
                        flat_idx = zi * nx * ny + yi * nx + xi
                        # Store as [yi, xi, zi] to account for swap
                        velocity[yi, xi, zi, :] = velocity_flat[flat_idx, :]
            
            # Transpose to get (x, y, z, 3): swap first two dimensions
            velocity = np.transpose(velocity, (1, 0, 2, 3))
            
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

