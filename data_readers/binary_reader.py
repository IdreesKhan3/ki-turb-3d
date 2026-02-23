"""
Binary file reader for structure functions and other binary data
Reads structure_funcs*_t*.bin files and tau_analysis_*.bin files
Auto-detects float32 vs float64 from file size.
"""

import os
import warnings
import numpy as np
import struct
from typing import Dict


def read_structure_function_file(filepath: str) -> Dict:
    """
    Read binary structure function file (structure_funcs*_t*.bin)
    Format: Header (nx, ny, nz, max_dr, norders, u_rms, dx) + data
    Auto-detects float32 vs float64 from file size.
    """
    try:
        with open(filepath, 'rb') as f:
            # Read header (6 ints + 2 floats = 32 bytes)
            nx = struct.unpack('i', f.read(4))[0]
            ny = struct.unpack('i', f.read(4))[0]
            nz = struct.unpack('i', f.read(4))[0]
            max_dr = struct.unpack('i', f.read(4))[0]
            norders = struct.unpack('i', f.read(4))[0]
            u_rms = struct.unpack('f', f.read(4))[0]
            dx = struct.unpack('f', f.read(4))[0]
            
            # Auto-detect float32 vs float64 from file size
            header_size = 32
            data_size = max_dr * 4 + norders * max_dr * 4  # float32
            expected_f32 = header_size + data_size
            data_size_f64 = max_dr * 8 + norders * max_dr * 8
            expected_f64 = header_size + data_size_f64
            fsize = os.path.getsize(filepath)
            if fsize == expected_f64:
                dtype = np.float64
                bytes_per_val = 8
            elif fsize == expected_f32:
                dtype = np.float32
                bytes_per_val = 4
            elif abs(fsize - expected_f32) <= 8:
                # Tolerate minor size mismatch (format variation); use float32
                dtype = np.float32
                bytes_per_val = 4
            else:
                dtype = np.float32
                bytes_per_val = 4
                warnings.warn(f"Structure function file size mismatch. Expected {expected_f32} (float32) or {expected_f64} (float64), got {fsize}. Using float32.")
            
            r = np.frombuffer(f.read(max_dr * bytes_per_val), dtype=dtype)
            S_p = {}
            for p in range(1, norders + 1):
                S_p[p] = np.frombuffer(f.read(max_dr * bytes_per_val), dtype=dtype)
            
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
    Auto-detects float32 vs float64 from file size.
    """
    try:
        expected_f32 = nx * ny * nz * 2 * 4
        expected_f64 = nx * ny * nz * 2 * 8
        fsize = os.path.getsize(filepath)
        if fsize == expected_f64:
            dtype = np.float64
        elif fsize == expected_f32:
            dtype = np.float32
        else:
            raise ValueError(f"{filepath}: expected {expected_f32} (float32) or {expected_f64} (float64) bytes, got {fsize}")
        
        data = np.fromfile(filepath, dtype=dtype)
        arr = data.reshape(nz, ny, nx, 2)  # Fortran order: (k, j, i, component)
        tau_offset_3d = arr[..., 0]  # Extract tau_offset channel
        tau_e_3d = tau_offset_3d + 0.5
        
        return float(np.mean(tau_e_3d))
    except Exception as e:
        raise ValueError(f"Error reading tau_analysis file {filepath}: {e}")

