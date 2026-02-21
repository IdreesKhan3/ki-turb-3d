"""
Shared utilities for agent tools (cache, path resolution).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def _group_files_by_simulation(files: List[str], pattern: str) -> Dict[str, List[str]]:
    """Group files by simulation prefix using regex. Matches file_detector.group_files_by_simulation."""
    groups: Dict[str, List[str]] = {}
    for f in files:
        match = re.match(pattern, Path(f).name)
        if match:
            prefix = match.group(1)
            groups.setdefault(prefix, []).append(f)
    for prefix in groups:
        groups[prefix] = sorted(groups[prefix], key=lambda f: _natural_sort_key(str(f)))
    return groups


def resolve_data_dir_and_find_files(
    data_dir: str,
    pattern: str,
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
    max_files: int = 100,
) -> List[Path]:
    """
    Resolve data directory and find files matching pattern.
    Used by spectra, spectral_isotropy, real_isotropy tools.

    - Uses session data_directory / data_directories when data_dir empty
    - Tries examples/ prefix when path doesn't exist (e.g. /DNS/512 -> examples/DNS/512)
    - rglob fallback: searches project when no files in directory
    """
    sess = session_context or {}

    def _files_in_dir(d: Path) -> List[Path]:
        if not d.exists() or not d.is_dir():
            return []
        return sorted(d.glob(pattern), key=lambda f: _natural_sort_key(str(f)))

    # Resolve data_dir: use session context when empty
    search_dir = data_dir or sess.get("data_directory") or ""
    if not search_dir and sess.get("data_directories"):
        dirs = sess["data_directories"]
        search_dir = dirs[0] if isinstance(dirs, list) and dirs else ""

    if search_dir:
        p = Path(search_dir)
        if not p.is_absolute():
            p = (project_root / search_dir).resolve()
        if p.exists() and p.is_dir():
            files = _files_in_dir(p)
            if files:
                return files[:max_files]
        # Try examples/ prefix (e.g. /DNS/512 -> examples/DNS/512, LES/64 -> examples/LES/64)
        stripped = search_dir.lstrip("/")
        if search_dir.startswith("/") or not (project_root / search_dir).exists():
            alt = project_root / "examples" / stripped
            if alt.exists() and alt.is_dir():
                files = _files_in_dir(alt)
                if files:
                    return files[:max_files]

    # Fallback: rglob project for pattern
    found = list(project_root.rglob(pattern))
    if found:
        by_dir: Dict[Path, List[Path]] = {}
        for f in found:
            by_dir.setdefault(f.parent, []).append(f)
        best_dir = max(by_dir.keys(), key=lambda d: len(by_dir[d]))
        files = sorted(by_dir[best_dir], key=lambda f: _natural_sort_key(str(f)))
        return files[:max_files]
    return []


def resolve_data_dirs_and_group_isotropy(
    data_dirs: Optional[List[str]] = None,
    data_dir: str = "",
    pattern: str = "isotropy_coeff_*.dat",
    project_root: Optional[Path] = None,
    session_context: Optional[Dict[str, Any]] = None,
    max_files_per_group: int = 1000,
) -> Dict[str, List[Path]]:
    """
    Resolve data directories and group isotropy files by simulation.
    Returns {sim_prefix: [Path, ...]} for multi-sim, or {"default": [Path, ...]} for single.
    """
    sess = session_context or {}
    project_root = project_root or Path(".")

    # Resolve search dirs: data_dirs > data_dir > session data_directories > session data_directory
    dirs_to_search: List[str] = []
    if data_dirs and isinstance(data_dirs, list) and len(data_dirs) > 0:
        dirs_to_search = list(data_dirs)
    elif data_dir:
        dirs_to_search = [data_dir]
    elif sess.get("data_directories"):
        d = sess["data_directories"]
        dirs_to_search = list(d) if isinstance(d, list) else [d]
    elif sess.get("data_directory"):
        dirs_to_search = [sess["data_directory"]]

    ic_groups: Dict[str, List[Path]] = {}
    group_pattern = r"(isotropy_coeff[_\w]*\d+)_\d+\.dat"
    fallback_pattern = r"isotropy_coeff_(\d+)_\d+\.dat"

    for search_dir in dirs_to_search:
        if not search_dir:
            continue
        p = Path(search_dir)
        if not p.is_absolute():
            p = (project_root / search_dir.lstrip("/")).resolve()
        if not p.exists() or not p.is_dir():
            alt = project_root / "examples" / search_dir.lstrip("/")
            if alt.exists() and alt.is_dir():
                p = alt
            else:
                continue
        files = sorted(p.glob(pattern), key=lambda f: _natural_sort_key(str(f)))
        file_strs = [str(f) for f in files]
        if not file_strs:
            continue
        grouped = _group_files_by_simulation(file_strs, group_pattern)
        if not grouped:
            grouped = _group_files_by_simulation(file_strs, fallback_pattern)
        if grouped:
            dir_name = p.name
            for key, flist in grouped.items():
                new_key = f"{dir_name}_{key}" if len(dirs_to_search) > 1 else key
                if new_key not in ic_groups:
                    ic_groups[new_key] = []
                paths = [Path(f) for f in flist[:max_files_per_group]]
                ic_groups[new_key].extend(paths)
        else:
            group_key = p.name if len(dirs_to_search) > 1 else "default"
            if group_key not in ic_groups:
                ic_groups[group_key] = []
            ic_groups[group_key].extend([Path(f) for f in file_strs[:max_files_per_group]])

    # Sort within each group
    for key in ic_groups:
        ic_groups[key] = sorted(ic_groups[key], key=lambda f: _natural_sort_key(str(f)))

    if not ic_groups and not dirs_to_search:
        # Fallback: single dir from resolve_data_dir_and_find_files
        flat = resolve_data_dir_and_find_files(
            data_dir, pattern, project_root, session_context, max_files_per_group
        )
        if flat:
            ic_groups["default"] = flat
    return ic_groups


def save_to_cache(session_context: Dict[str, Any], key: str, data: Any) -> None:
    """Store heavy data in session cache to avoid flooding LLM context."""
    cache = session_context.setdefault("agent_data_cache", {})
    cache[key] = data


def get_from_cache(session_context: Dict[str, Any], key: str) -> Any:
    """Retrieve data from session cache."""
    return session_context.get("agent_data_cache", {}).get(key)


def resolve_path(filepath: str, project_root: Path) -> Path:
    """Resolve path to project. Raises ValueError if outside project."""
    p = (project_root / filepath).resolve()
    if ".." in filepath or not str(p).startswith(str(project_root.resolve())):
        raise ValueError("Path must be inside project")
    return p
