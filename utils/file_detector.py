"""
File detection utility
Scans directories for simulation data files
"""

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def detect_simulation_files(directory: str) -> Dict[str, List[str]]:
    """
    Scan directory and detect all available simulation files
    
    Args:
        directory: Path to simulation output directory
        
    Returns:
        Dictionary with file types as keys and file lists as values
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return {}
    
    files = {
        'real_turb_stats': [],
        'spectral_turb_stats': [],
        'parameters': [],
        'spectrum': [],
        'norm_spectrum': [],
        'structure_functions_txt': [],
        'structure_functions_bin': [],
        'flatness': [],
        'isotropy': [],
        'tau_analysis': [],
    }
    
    # Find real-space turbulence statistics files (turbulence_stats*.csv)
    files['real_turb_stats'] = sorted(dir_path.glob('turbulence_stats*.csv'), key=lambda f: natural_sort_key(str(f)))
    # Find spectral turbulence statistics files (eps_real_validation*.csv)
    eps_files = list(dir_path.glob('eps_real_validation*.csv'))
    files['spectral_turb_stats'] = sorted(eps_files, key=lambda f: natural_sort_key(str(f)))
    
    # Find parameter file
    files['parameters'] = list(dir_path.glob('simulation.input'))
    
    # Find spectrum files
    files['spectrum'] = sorted(dir_path.glob('spectrum*.dat'), key=lambda f: natural_sort_key(str(f)))
    files['norm_spectrum'] = sorted(dir_path.glob('norm*.dat'), key=lambda f: natural_sort_key(str(f)))
    
    # Find structure function files
    files['structure_functions_txt'] = sorted(dir_path.glob('structure_functions_*.txt'))
    files['structure_functions_bin'] = sorted(dir_path.glob('structure_funcs*_t*.bin'), key=lambda f: natural_sort_key(str(f)))
    
    # Find flatness files
    files['flatness'] = sorted(dir_path.glob('flatness_data*_*.txt'), key=lambda f: natural_sort_key(str(f)))
    
    # Find isotropy files
    files['isotropy'] = sorted(dir_path.glob('isotropy_coeff_*.dat'))
    
    # Find tau_analysis files (for LES effective relaxation time)
    files['tau_analysis'] = sorted(dir_path.glob('tau_analysis_*.bin'), key=lambda f: natural_sort_key(str(f)))
    
    return files


def natural_sort_key(s: str) -> List:
    """
    Natural sort key for file names with numbers
    
    Args:
        s: String to sort
        
    Returns:
        List for sorting
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def group_files_by_simulation(files: List[str], pattern: str) -> Dict[str, List[str]]:
    """
    Group files by simulation type using regex pattern
    
    Args:
        files: List of file paths
        pattern: Regex pattern to extract simulation prefix (e.g., r'(spectrum\d+)_\d+\.dat')
        
    Returns:
        Dictionary mapping simulation prefix to list of files
    """
    groups = defaultdict(list)
    for f in files:
        match = re.match(pattern, Path(f).name)
        if match:
            prefix = match.group(1)
            groups[prefix].append(f)
    
    # Sort files in each group
    for prefix in groups:
        groups[prefix] = sorted(groups[prefix], key=lambda f: natural_sort_key(str(f)))
    
    return dict(groups)

