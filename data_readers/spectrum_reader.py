"""
Spectrum file reader
Reads spectrum_*.dat files (energy spectra)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def read_spectrum_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read spectrum file (spectrum_*.dat)
    
    Format: k, E(k) (two columns)
    
    Args:
        filepath: Path to spectrum file
        
    Returns:
        Tuple of (k_values, E_values)
    """
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns, got {data.shape[1]}")
        
        k = data[:, 0]
        E = data[:, 1]
        
        return k, E
    except Exception as e:
        raise ValueError(f"Error reading spectrum file {filepath}: {e}")

