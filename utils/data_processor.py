"""
Data processing utilities
Time-averaging, statistics, etc.
"""

import numpy as np
from typing import List, Tuple, Optional


def process_time_average(data_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time average and standard deviation from list of arrays
    
    Args:
        data_list: List of arrays (each array is one time snapshot)
        
    Returns:
        Tuple of (mean_array, std_array)
    """
    if not data_list:
        raise ValueError("Empty data list")
    
    data_array = np.array(data_list)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    
    return mean, std


def compute_energy_variance(energy_accum: np.ndarray, energy_sq_accum: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute energy variance and standard deviation from accumulators
    
    Same as notebook: energy_var = (energy_sq_accum / count) - energy_avg²
    
    Args:
        energy_accum: Accumulated energy values
        energy_sq_accum: Accumulated energy² values
        count: Number of samples
        
    Returns:
        Tuple of (mean, std)
    """
    energy_avg = energy_accum / count
    energy_var = (energy_sq_accum / count) - energy_avg**2
    energy_std = np.sqrt(np.maximum(energy_var, 0.0))
    
    return energy_avg, energy_std

