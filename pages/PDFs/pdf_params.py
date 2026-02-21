"""
Shared parameter logic for PDF pages.
Grid spacing (dx, dy, dz) for LBM vs NS: LBM uses 1, NS uses L/nx from parameters.
Reads both simulation.input and simulation.json when present; user chooses which to use.
"""

from pathlib import Path
from typing import Dict, Tuple

from data_readers.parameter_reader import read_parameters


def get_grid_spacing_options(data_dir: Path) -> Dict[str, Tuple[float, float, float]]:
    """
    Read both simulation.input and simulation.json, return available grid spacing options.
    Keys are user-facing labels; values are (dx, dy, dz).
    Always includes "LBM (dx=1)" for lattice units.
    Adds "NS (simulation.json)" when simulation.json has L, nx, ny, nz.
    """
    options = {"LBM (dx=1, lattice units)": (1.0, 1.0, 1.0)}

    json_path = data_dir / "simulation.json"
    if json_path.exists():
        try:
            params = read_parameters(str(json_path))
            if params and all(k in params for k in ["L", "nx", "ny", "nz"]):
                L = float(params["L"])
                nx = int(params["nx"])
                ny = int(params["ny"])
                nz = int(params["nz"])
                if nx > 0 and ny > 0 and nz > 0:
                    dx_ns = L / nx
                    dy_ns = L / ny
                    dz_ns = L / nz
                    options["NS (simulation.json: dx=L/nx)"] = (dx_ns, dy_ns, dz_ns)
        except Exception:
            pass

    return options


def get_grid_spacing(data_dir: Path, param_file: Path = None) -> Tuple[float, float, float]:
    """
    Legacy: get dx, dy, dz. Prefers NS (simulation.json) when it has L,nx,ny,nz, else 1.
    """
    options = get_grid_spacing_options(data_dir)
    # Prefer NS if available, else LBM
    for label, (dx, dy, dz) in options.items():
        if "NS" in label:
            return dx, dy, dz
    return 1.0, 1.0, 1.0
