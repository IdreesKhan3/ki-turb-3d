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
    """Scan directory for velocity volume files (.vti/.h5), excluding density dumps."""
    from utils.file_detector import list_velocity_field_files
    return list_velocity_field_files(data_dir)


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


def _enrich_loaded_files_from_manifest(
    manifest,
    all_loaded_files: Dict[str, List[Dict]],
    project_root: Path,
) -> Dict[str, str]:
    """Add manifest-recorded analysis files to the session index (solver-neutral)."""
    from analysis.product_loader import AnalysisProductLoader

    loader = AnalysisProductLoader(project_root, {"dataset_manifest": manifest.model_dump(mode="json")})
    loader._manifest = manifest
    return loader.enrich_session_files(all_loaded_files)


def load_manifest_into_session(
    project_root: Path, manifest_path: str, session_state
) -> tuple[bool, str]:
    """Load a KI-TURB dataset manifest and scan its base directory into the session."""
    from schemas import DatasetManifest
    from analysis.product_loader import AnalysisProductLoader

    path = Path(manifest_path)
    if not path.is_absolute():
        path = (project_root / manifest_path).resolve()
    if not path.is_file():
        return False, f"Manifest not found: {manifest_path}"

    manifest = DatasetManifest.from_json(path.read_text(encoding="utf-8"))
    base_dir = manifest.base_dir

    session_state.dataset_manifest = manifest.model_dump(mode="json")
    session_state.data_directory = base_dir
    from utils.file_detector import expand_analysis_search_dirs

    expanded = [str(p) for p in expand_analysis_search_dirs(base_dir)]
    session_state.data_directories = expanded or [base_dir]
    all_files = _scan_directory_for_files(base_dir)
    # Also index processed product folders so manual pages see OpenLB analysis CSVs/dats.
    for extra in expanded[1:]:
        for file_type, file_list in _scan_directory_for_files(extra).items():
            all_files.setdefault(file_type, [])
            existing = {item.get("full_path") for item in all_files[file_type]}
            for item in file_list:
                if item["full_path"] not in existing:
                    all_files[file_type].append(item)
                    existing.add(item["full_path"])
    hints = _enrich_loaded_files_from_manifest(manifest, all_files, project_root)
    session_state.all_loaded_files = all_files
    if hints.get("spectra_data_directory"):
        session_state.spectra_data_directory = hints["spectra_data_directory"]
    if hints.get("stats_data_directory"):
        session_state.stats_data_directory = hints["stats_data_directory"]
    if hints.get("isotropy_data_directory"):
        session_state.isotropy_data_directory = hints["isotropy_data_directory"]
    if hints.get("structure_functions_data_directory"):
        session_state.structure_functions_data_directory = hints["structure_functions_data_directory"]
    if hints.get("analysis_products_path"):
        session_state.analysis_products_path = hints["analysis_products_path"]

    products = AnalysisProductLoader(project_root, {"dataset_manifest": manifest.model_dump(mode="json")}).products()
    if products is not None:
        session_state.analysis_products = products.model_dump(mode="json")
    session_state.data_loaded = True
    session_state.multi_directory_mode = False

    return True, f"Loaded KI-TURB manifest ({len(manifest.files)} files) from {path.name}"


def load_manifest_into_context(
    project_root: Path,
    manifest_path: str,
    session_context: dict,
) -> tuple[bool, str]:
    """Load a manifest into the agent session_context dict (no Streamlit required)."""
    from schemas import DatasetManifest
    from analysis.product_loader import AnalysisProductLoader

    path = Path(manifest_path)
    if not path.is_absolute():
        path = (project_root / manifest_path).resolve()
    if not path.is_file():
        return False, f"Manifest not found: {manifest_path}"

    manifest = DatasetManifest.from_json(path.read_text(encoding="utf-8"))
    base_dir = manifest.base_dir

    session_context["dataset_manifest"] = manifest.model_dump(mode="json")
    session_context["manifest_path"] = str(path)
    session_context["data_directory"] = base_dir
    from utils.file_detector import expand_analysis_search_dirs

    expanded = [str(p) for p in expand_analysis_search_dirs(base_dir)]
    session_context["data_directories"] = expanded or [base_dir]
    all_files = _scan_directory_for_files(base_dir)
    for extra in expanded[1:]:
        for file_type, file_list in _scan_directory_for_files(extra).items():
            all_files.setdefault(file_type, [])
            existing = {item.get("full_path") for item in all_files[file_type]}
            for item in file_list:
                if item["full_path"] not in existing:
                    all_files[file_type].append(item)
                    existing.add(item["full_path"])
    hints = _enrich_loaded_files_from_manifest(manifest, all_files, project_root)
    session_context["all_loaded_files"] = all_files
    if hints.get("spectra_data_directory"):
        session_context["spectra_data_directory"] = hints["spectra_data_directory"]
    if hints.get("stats_data_directory"):
        session_context["stats_data_directory"] = hints["stats_data_directory"]
    if hints.get("isotropy_data_directory"):
        session_context["isotropy_data_directory"] = hints["isotropy_data_directory"]
    if hints.get("structure_functions_data_directory"):
        session_context["structure_functions_data_directory"] = hints["structure_functions_data_directory"]
    if hints.get("analysis_products_path"):
        session_context["analysis_products_path"] = hints["analysis_products_path"]
    products = AnalysisProductLoader(project_root, session_context).products()
    if products is not None:
        session_context["analysis_products"] = products.model_dump(mode="json")
    session_context["data_loaded"] = True
    session_context["multi_directory_mode"] = False
    return True, f"Loaded KI-TURB manifest ({len(manifest.files)} files) from {path.name}"
