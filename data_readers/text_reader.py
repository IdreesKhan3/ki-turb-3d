"""
Text file reader for structure functions and flatness data
Reads structure_functions_*.txt and flatness_data*_*.txt files
"""

import numpy as np
from typing import Tuple


def read_structure_function_txt(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read structure function text file (structure_functions_*.txt)
    
    Args:
        filepath: Path to text file
        
    Returns:
        Tuple of (r_values, S_p_values) or multiple orders
    """
    try:
        data = np.loadtxt(filepath, skiprows=1)
        r = data[:, 0]
        # Assume remaining columns are different orders
        S_p_data = data[:, 1:]
        return r, S_p_data
    except Exception as e:
        raise ValueError(f"Error reading structure function file {filepath}: {e}")


def read_flatness_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read flatness file (flatness_data*_*.txt)
    
    Format: r, F(r) (two columns, skip first row)
    
    Args:
        filepath: Path to flatness file
        
    Returns:
        Tuple of (r_values, flatness_values)
    """
    try:
        data = np.loadtxt(filepath, skiprows=1)
        r = data[:, 0]
        flatness = data[:, 1]
        return r, flatness
    except Exception as e:
        raise ValueError(f"Error reading flatness file {filepath}: {e}")

