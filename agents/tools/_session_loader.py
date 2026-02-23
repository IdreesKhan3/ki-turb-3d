"""
Session loader for app_control tools.
Loads simulation directories into Streamlit session state.
Agent-only; not used by manual app sidebar.
"""

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional

from utils.file_detector import detect_simulation_files, natural_sort_key

logger = logging.getLogger(__name__)


def _get_velocity_files(data_dir: Path) -> List[str]:
    """Scan directory for velocity files (.vti, .h5, .hdf5)."""
    vti_files = sorted(
        glob.glob(str(data_dir / "*.vti")) + glob.glob(str(data_dir / "*.VTI")),
        key=natural_sort_key
    )
    hdf5_files = sorted(
        glob.glob(str(data_dir / "*.h5")) + glob.glob(str(data_dir / "*.H5")) +
        glob.glob(str(data_dir / "*.hdf5")) + glob.glob(str(data_dir / "*.HDF5")),
        key=natural_sort_key
    )
    return vti_files + hdf5_files


def resolve_data_directory(project_root: Path, user_dir: str, log_warnings: bool = True) -> Optional[Path]:
    """
    Resolve a directory path (absolute or relative to project root).
    """
    if not user_dir:
        if log_warnings:
            logger.debug("Empty directory path provided")
        return None

    user_dir = user_dir.strip().replace("\\", "/")

    abs_path = Path(user_dir).resolve()
    if abs_path.exists() and abs_path.is_dir():
        return abs_path

    relative_path = (project_root / user_dir).resolve()
    if relative_path.exists() and relative_path.is_dir():
        return relative_path

    if log_warnings:
        logger.warning(f"Could not resolve directory path: {user_dir}")
    return None


def _scan_directory_for_files(data_dir_path: str) -> Dict[str, List[Dict]]:
    """Scan directory and collect simulation files by type."""
    data_dir = Path(data_dir_path)
    all_files_by_type = {}

    files_dict = detect_simulation_files(str(data_dir))
    for file_type, file_list in files_dict.items():
        if file_type not in all_files_by_type:
            all_files_by_type[file_type] = []
        for file_path in file_list:
            all_files_by_type[file_type].append({
                "full_path": str(Path(file_path)),
                "directory": str(data_dir),
                "filename": Path(file_path).name,
            })

    velocity_files = _get_velocity_files(data_dir)
    if "velocity_files" not in all_files_by_type:
        all_files_by_type["velocity_files"] = []
    for f in velocity_files:
        all_files_by_type["velocity_files"].append({
            "full_path": str(f),
            "directory": str(data_dir),
            "filename": Path(f).name,
        })

    return all_files_by_type


def load_data_into_session(
    project_root: Path,
    paths: List[str],
    multi_directory_mode: bool,
    session_state,
) -> tuple[bool, str]:
    """
    Load one or more directories into Streamlit session state.
    Updates data_directory, data_directories, all_loaded_files, data_loaded.
    """
    valid_dirs = []
    for p in paths:
        if not p or not p.strip():
            continue
        resolved = resolve_data_directory(project_root, p.strip(), log_warnings=False)
        if resolved:
            valid_dirs.append(str(resolved))
        else:
            return False, f"Directory not found: {p}"

    if not valid_dirs:
        return False, "No valid directories provided."

    logger.info(f"Loading {len(valid_dirs)} directory(ies): {[Path(d).name for d in valid_dirs]}")

    all_files_by_type = {}
    for data_dir_path in valid_dirs:
        dir_files = _scan_directory_for_files(data_dir_path)
        for file_type, file_list in dir_files.items():
            if file_type not in all_files_by_type:
                all_files_by_type[file_type] = []
            all_files_by_type[file_type].extend(file_list)

    for ft, lst in all_files_by_type.items():
        seen = set()
        dedup = []
        for item in lst:
            if item["full_path"] not in seen:
                dedup.append(item)
                seen.add(item["full_path"])
        all_files_by_type[ft] = dedup

    session_state.data_directories = valid_dirs
    session_state.data_directory = valid_dirs[0]
    session_state.all_loaded_files = all_files_by_type
    session_state.data_loaded = True

    return True, f"Loaded {len(valid_dirs)} director{'ies' if len(valid_dirs) > 1 else 'y'}: {', '.join(Path(d).name for d in valid_dirs)}"
