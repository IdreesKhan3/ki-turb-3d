"""
Binary file reader for structure functions and other binary data
Reads structure_funcs*_t*.bin files
"""

import numpy as np
import struct
from pathlib import Path
from typing import Dict, Optional


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

