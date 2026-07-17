"""
File detection utility
Scans directories for simulation data files
"""

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def expand_analysis_search_dirs(data_dir: str | Path) -> List[Path]:
    """
    Expand a session data directory into places that may hold analysis products.

    Supports classic flat example folders and OpenLB jobs where ``data_directory``
    points at ``.../raw`` while CSVs/spectra live under ``.../processed/<kind>/``.
    """
    d = Path(data_dir).resolve()
    dirs: List[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        if path.is_dir():
            key = str(path)
            if key not in seen:
                seen.add(key)
                dirs.append(path)

    _add(d)
    if d.name == "raw":
        processed = d.parent / "processed"
    else:
        processed = d / "processed"
    if processed.is_dir():
        _add(processed)
        for sub in (
            "stats",
            "spectra",
            "isotropy",
            "flatness",
            "structure_functions",
            "pdfs",
        ):
            _add(processed / sub)
    return dirs


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
        'velocity_vti': [],
        'velocity_h5': [],
        'manifest': [],
    }

    def _analysis_roots() -> List[Path]:
        """Dirs that may hold analysis products (flat examples/ or OpenLB processed/)."""
        roots = [dir_path]
        # OpenLB job: session often points at .../raw while products live in ../processed/*
        if dir_path.name == "raw":
            sibling = dir_path.parent / "processed"
            if sibling.is_dir():
                roots.append(sibling)
        nested = dir_path / "processed"
        if nested.is_dir():
            roots.append(nested)
        return roots

    def _find(*patterns: str) -> List:
        """Match each pattern in the directory and OpenLB processed/* product folders."""
        found = []
        for root in _analysis_roots():
            for pattern in patterns:
                found += list(root.glob(pattern))
                # processed/stats/file, processed/spectra/file, ...
                found += list(root.glob(f"*/{pattern}"))
                found += list(root.glob(f"processed/*/{pattern}"))
        return sorted(set(found), key=lambda f: natural_sort_key(str(f)))

    # KI-TURB dataset manifest (top level, processed/, or one folder down).
    files['manifest'] = sorted(
        set(dir_path.glob('dataset_manifest.json'))
        | set(dir_path.glob('manifest.json'))
        | set(dir_path.glob('*/dataset_manifest.json'))
        | set(dir_path.glob('*/manifest.json')),
        key=lambda f: natural_sort_key(str(f)),
    )

    files['analysis_products'] = _find('hit_analysis_products.json')
    files['velocity_pdf'] = _find('velocity_pdf*.dat', 'pdf_velocity*.dat')
    files['gradient_pdf'] = _find('gradient_pdf*.dat', 'pdf_gradient*.dat')
    files['dissipation_pdf'] = _find('dissipation_pdf*.dat', 'pdf_dissipation*.dat')
    files['enstrophy_pdf'] = _find('enstrophy_pdf*.dat', 'pdf_enstrophy*.dat')
    files['joint_pdf'] = _find('joint_pdf*.dat', 'pdf_joint*.dat')
    files['rq_pdf'] = _find('rq_pdf*.dat', 'pdf_rq*.dat')
    files['reynolds_stress'] = _find('reynolds_stress*.csv', 'reynolds_stress_validation*.csv')
    files['diagnostics'] = _find('diagnostics.jsonl', 'diagnostics*.jsonl')

    # Real-space turbulence statistics (turbulence_stats*.csv)
    files['real_turb_stats'] = _find('turbulence_stats*.csv')
    # Spectral/validation statistics (LBM: eps_real_validation, NS: turbulence_validation)
    files['spectral_turb_stats'] = _find('eps_real_validation*.csv', 'turbulence_validation*.csv')

    # Parameter file (LBM: simulation.input, NS: simulation.json)
    files['parameters'] = list(dir_path.glob('simulation.input')) + list(dir_path.glob('simulation.json'))

    # Energy spectra
    files['spectrum'] = _find('spectrum*.dat')
    files['norm_spectrum'] = _find('norm*.dat')

    # Structure functions (accept both underscore and grouping-style names)
    files['structure_functions_txt'] = _find('structure_functions*.txt')
    files['structure_functions_bin'] = _find('structure_funcs*_t*.bin')

    # Flatness
    files['flatness'] = _find('flatness_data*_*.txt')

    # Spectral isotropy
    files['isotropy'] = _find('isotropy_coeff_*.dat')

    # Effective relaxation-time analysis (LES)
    files['tau_analysis'] = _find('tau_analysis_*.bin')

    # Velocity fields (.vti and .h5/.hdf5) — exclude density/vorticity companion dumps
    velocity_paths = [Path(p) for p in list_velocity_field_files(dir_path)]
    files['velocity_vti'] = [p for p in velocity_paths if p.suffix.lower() == ".vti"]
    files['velocity_h5'] = [p for p in velocity_paths if p.suffix.lower() in {".h5", ".hdf5"}]

    return files


def is_velocity_field_filename(name: str) -> bool:
    """True when a volume file name looks like a 3-component velocity field."""
    stem = Path(name).stem.lower()
    # Explicit non-velocity OpenLB/CFD field dumps.
    excluded_prefixes = (
        "density",
        "pressure",
        "vorticity",
        "forcing",
        "force",
        "tau",
        "omega",
        "qcrit",
        "q_criterion",
        "lambda2",
    )
    if any(stem.startswith(p) or stem.startswith(p + "_") for p in excluded_prefixes):
        return False
    # Prefer explicit velocity naming (examples + OpenLB).
    if stem.startswith("velocity") or "velocity" in stem:
        return True
    # Legacy unlabeled volumes: allow generic names, still exclude known scalars/vectors above.
    return not any(p in stem for p in ("density", "pressure", "vorticity", "forcing"))


def list_velocity_field_files(data_dirs) -> List[str]:
    """
    Collect velocity volume files from one or more directories.

    Prefers ``velocity_*.vti`` / ``Velocity_*.vti`` / ``velocity_*.h5`` and excludes
    OpenLB companion dumps such as ``density_*.vti`` and ``vorticity_*.vti``.
    Works for classic ``examples/DNS|LES`` folders and simulation ``raw/`` trees.
    """
    if isinstance(data_dirs, (str, Path)):
        dirs = [data_dirs]
    else:
        dirs = list(data_dirs or [])

    found: List[str] = []
    seen: set[str] = set()
    for data_dir in dirs:
        root = Path(data_dir).resolve()
        if not root.is_dir():
            continue
        candidates = (
            list(root.glob("*.vti"))
            + list(root.glob("*.VTI"))
            + list(root.glob("*.h5"))
            + list(root.glob("*.H5"))
            + list(root.glob("*.hdf5"))
            + list(root.glob("*.HDF5"))
        )
        preferred = [p for p in candidates if is_velocity_field_filename(p.name)]
        # If explicit velocity names exist, use only those; otherwise keep filtered set.
        explicit = [
            p for p in preferred
            if p.stem.lower().startswith("velocity") or "velocity" in p.stem.lower()
        ]
        chosen = explicit or preferred
        for path in sorted(chosen, key=lambda p: natural_sort_key(str(p))):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                found.append(key)
    return found


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
    r"""
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

