"""
Binary file reader for structure functions and other binary data
Reads structure_funcs*_t*.bin files and tau_analysis_*.bin files
"""

import numpy as np
import struct
from typing import Dict


def read_structure_function_file(filepath: str) -> Dict:
    """
    Read binary structure function file (structure_funcs*_t*.bin)
    
    Format: Header (nx, ny, nz, max_dr, norders, u_rms, dx) + data
    
    Args:
        filepath: Path to binary file
        
    Returns:
        Dictionary with structure function data
    """
    try:
        with open(filepath, 'rb') as f:
            # Read header
            nx = struct.unpack('i', f.read(4))[0]
            ny = struct.unpack('i', f.read(4))[0]
            nz = struct.unpack('i', f.read(4))[0]
            max_dr = struct.unpack('i', f.read(4))[0]
            norders = struct.unpack('i', f.read(4))[0]
            u_rms = struct.unpack('f', f.read(4))[0]
            dx = struct.unpack('f', f.read(4))[0]
            
            # Read r values
            r = np.frombuffer(f.read(max_dr * 4), dtype=np.float32)
            
            # Read S_p for each order
            S_p = {}
            for p in range(1, norders + 1):
                S_p[p] = np.frombuffer(f.read(max_dr * 4), dtype=np.float32)
            
            # Find minimum length
            min_len = min(len(r), *(len(v) for v in S_p.values()))
            
            return {
                'nx': nx, 'ny': ny, 'nz': nz,
                'max_dr': max_dr, 'norders': norders,
                'u_rms': u_rms, 'dx': dx,
                'r': r[:min_len],
                'S_p': {p: S_p[p][:min_len] for p in S_p}
            }
    except Exception as e:
        raise ValueError(f"Error reading binary file {filepath}: {e}")


def read_tau_analysis_file(filepath: str, nx: int, ny: int, nz: int) -> float:
    """
    Read tau_analysis_*.bin file and return average effective relaxation time τ_e.
    
    File format: L * M * N * 2 * 4 bytes (float32)
    - Data written in Fortran order: (k, j, i, component)
    - Each grid point: [tau_offset, normalized_offset]
    - tau_offset = τ_e - 0.5, where τ_e = 1.0 / s9_field
    
    Args:
        filepath: Path to tau_analysis_*.bin file
        nx, ny, nz: Grid dimensions (L, M, N)
        
    Returns:
        Average effective relaxation time: τ_e = mean(tau_offset) + 0.5
    """
    try:
        import os
        expected_bytes = nx * ny * nz * 2 * 4
        fsize = os.path.getsize(filepath)
        if fsize != expected_bytes:
            raise ValueError(f"{filepath}: expected {expected_bytes} bytes, got {fsize}")
        
        data = np.fromfile(filepath, dtype=np.float32)
        arr = data.reshape(nz, ny, nx, 2)  # Fortran order: (k, j, i, component)
        tau_offset_3d = arr[..., 0]  # Extract tau_offset channel
        tau_e_3d = tau_offset_3d + 0.5
        
        return float(np.mean(tau_e_3d))
    except Exception as e:
        raise ValueError(f"Error reading tau_analysis file {filepath}: {e}")

