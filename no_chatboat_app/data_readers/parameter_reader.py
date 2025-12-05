"""
Parameter file reader for simulation.input (Fortran namelist format)
Extracts all parameters from parameters module
"""

import re
from pathlib import Path
from typing import Dict, Optional


def read_parameters(filepath: str) -> Dict:
    """
    Read simulation.input file (Fortran namelist format)
    
    Format:
    &input_params
    nx = 64
    ny = 64
    nz = 64
    nu = 0.003
    ...
    /
    
    Args:
        filepath: Path to simulation.input file
        
    Returns:
        Dictionary of parameters with user-friendly labels
    """
    params = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract content between &input_params and /
        match = re.search(r'&input_params\s*(.*?)\s*/', content, re.DOTALL)
        if not match:
            return params
        
        param_block = match.group(1)
        
        # Parse key-value pairs
        for line in param_block.split('\n'):
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('!'):
                continue
            
            # Remove comments from line
            if '!' in line:
                line = line.split('!')[0].strip()
            
            # Parse key = value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                # Convert to appropriate type
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
        
        # Compute derived parameters
        if 'nu' in params and 'Cs2' not in params:
            Cs2 = 1.0 / 3.0
            if 'tau' not in params:
                params['tau'] = params.get('nu', 0.0) / Cs2 + 0.5
        
        return params
        
    except Exception as e:
        raise ValueError(f"Error reading parameter file {filepath}: {e}")


def format_parameters_for_display(params: Dict) -> Dict:
    """
    Format parameters with user-friendly labels and units
    
    Args:
        params: Raw parameters dictionary
        
    Returns:
        Dictionary with formatted labels and units
    """
    formatted = {}
    
    # Grid parameters
    grid_params = {
        'nx': ('Grid Size X', 'cells'),
        'ny': ('Grid Size Y', 'cells'),
        'nz': ('Grid Size Z', 'cells'),
        'num_time_steps': ('Time Steps', 'iterations'),
        'vtk_interval': ('VTK Interval', 'iterations'),
        'data_interval': ('Data Interval', 'iterations'),
        'tag': ('Simulation Tag', ''),
    }
    
    # Physical parameters
    physical_params = {
        'nu': ('Viscosity', 'lattice units'),
        'u0': ('Reference Velocity', 'lattice units'),
        'tau': ('Relaxation Time', 'lattice units'),
        'F_amp': ('Forcing Amplitude', 'lattice units'),
        'perturb_temp': ('Perturbation Scale', 'lattice units'),
    }
    
    # LBM parameters
    lbm_params = {
        'q': ('Lattice Model', 'D3Q19'),
        'Cs': ('Speed of Sound', '1/√3'),
        'Cs2': ('Speed of Sound²', '1/3'),
        'Lc': ('Characteristic Length', 'lattice units'),
        'SmogC': ('Smagorinsky Constant', ''),
    }
    
    # Filtering parameters
    filter_params = {
        'downsample_factor': ('Downsample Factor', ''),
        'FILTER_CHOICE': ('Filter Type', '1=Gaussian, 2=Box'),
    }
    
    all_params = {**grid_params, **physical_params, **lbm_params, **filter_params}
    
    for key, (label, unit) in all_params.items():
        if key in params:
            formatted[label] = {
                'value': params[key],
                'unit': unit,
                'key': key
            }
    
    return formatted

