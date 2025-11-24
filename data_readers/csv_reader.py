"""
CSV file reader for turbulence statistics
Reads turbulence_stats1.csv and eps_real_validation_*.csv files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def read_csv_data(filepath: str) -> pd.DataFrame:
    """
    Read CSV file containing turbulence statistics
    
    Args:
        filepath: Path to CSV file (e.g., turbulence_stats1.csv)
        
    Returns:
        DataFrame with turbulence statistics
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file {filepath}: {e}")


def read_eps_validation_csv(filepath: str) -> pd.DataFrame:
    """
    Read eps_real_validation_*.csv file
    
    Columns: iter, iter_norm, eps_real, eps_spectral, TKE_real, u_rms_real,
             Sij_Sij_mean, Re_Tp, Re_Lp, Re_Bp, eta_cell_real, eta_domain_real,
             kmax_eta_cell_real, kmax_eta_domain_real, eta_over_dx, imbalance,
             dTKE_dt, forcing_power, eps_balance, energy_balance_ratio,
             frac_x, frac_y, frac_z
    
    Args:
        filepath: Path to eps_real_validation_*.csv file
        
    Returns:
        DataFrame with real-space validation data
    """
    try:
        df = pd.read_csv(filepath)
        # Clean numeric columns
        numeric_cols = ['iter', 'iter_norm', 'eps_real', 'eps_spectral', 'TKE_real', 
                       'u_rms_real', 'energy_balance_ratio', 'frac_x', 'frac_y', 'frac_z']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['iter', 'energy_balance_ratio'])
        return df
    except Exception as e:
        raise ValueError(f"Error reading validation CSV file {filepath}: {e}")

