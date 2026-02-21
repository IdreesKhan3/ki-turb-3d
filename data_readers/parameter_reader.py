"""
Parameter file reader for simulation.input (LBM Fortran namelist) and simulation.json (NS JSON).
Supports both LBM and Navier-Stokes FHIT configurations.
"""

import json
import re
from pathlib import Path
from typing import Dict


def _read_parameters_input(filepath: str) -> Dict:
    """Read simulation.input (Fortran namelist format)."""
    params = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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


def _read_parameters_json(filepath: str) -> Dict:
    """Read simulation.json (NS JSON format). Supports nx, ny, nz, nu, c_sound, L, Re, etc."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error reading JSON parameter file {filepath}: {e}")

    # Flatten nested dicts and normalize keys
    params = {}

    def _flatten(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = str(k).lower()
            key = key.replace('n_x', 'nx').replace('n_y', 'ny').replace('n_z', 'nz')
            if key == 'viscosity':
                key = 'nu'
            elif key in ('c_sound', 'c_speed', 'speed_of_sound'):
                key = 'c_sound'
            elif key in ('integral_scale', 'length_scale', 'l'):
                key = 'L'
            if isinstance(v, dict):
                _flatten(v, prefix + key + "_")
            else:
                params[key] = v

    if isinstance(data, dict):
        _flatten(data)

    # Ensure L for Knudsen: use domain size if L not given
    if 'L' not in params and all(k in params for k in ['nx', 'ny', 'nz']):
        params['L'] = float(min(params['nx'], params['ny'], params['nz']))

    return params


def read_parameters(filepath: str) -> Dict:
    """
    Read simulation parameters from simulation.input (LBM) or simulation.json (NS).
    Auto-detects format from file extension.
    """
    path = Path(filepath)
    if path.suffix.lower() == '.json':
        return _read_parameters_json(filepath)
    return _read_parameters_input(filepath)


def is_lbm_params(filepath: str) -> bool:
    """Return True if file is LBM format (simulation.input), False for NS (simulation.json)."""
    return Path(filepath).suffix.lower() != '.json'


def format_parameters_for_display(params: Dict, is_lbm: bool = True) -> Dict:
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
    
    # Physical parameters (LBM and NS)
    physical_params = {
        'nu': ('Viscosity', 'lattice units' if is_lbm else 'm²/s'),
        'u0': ('Reference Velocity', 'lattice units' if is_lbm else 'm/s'),
        'tau': ('Relaxation Time', 'lattice units'),
        'F_amp': ('Forcing Amplitude', 'lattice units'),
        'perturb_temp': ('Perturbation Scale', 'lattice units'),
        'c_sound': ('Speed of Sound', 'm/s'),
        'L': ('Characteristic Length', 'm'),
        'Re': ('Reynolds Number', ''),
    }
    
    # LBM parameters (only shown for LBM)
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
    # For NS, skip LBM-only params
    if not is_lbm:
        all_params = {k: v for k, v in all_params.items() if k not in ('tau', 'q', 'Cs', 'Cs2', 'SmogC')}

    for key, (label, unit) in all_params.items():
        if key in params:
            formatted[label] = {
                'value': params[key],
                'unit': unit,
                'key': key
            }
    
    return formatted

