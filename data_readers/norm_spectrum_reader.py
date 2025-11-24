"""
Normalized spectrum file reader
Reads norm_*.dat files (normalized spectrum with Pope model)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def read_norm_spectrum_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read normalized spectrum file (norm_*.dat)
    
    Format: k_eta, E_norm, E_pope_norm (three columns)
    
    Args:
        filepath: Path to normalized spectrum file
        
    Returns:
        Tuple of (k_eta, E_norm, E_pope_norm)
    """
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.shape[1] < 3:
            raise ValueError(f"Expected at least 3 columns, got {data.shape[1]}")
        
        k_eta = data[:, 0]
        E_norm = data[:, 1]
        E_pope_norm = data[:, 2]
        
        return k_eta, E_norm, E_pope_norm
    except Exception as e:
        raise ValueError(f"Error reading normalized spectrum file {filepath}: {e}")

