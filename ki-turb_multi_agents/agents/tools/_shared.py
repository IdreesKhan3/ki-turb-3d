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


def _rglob_fallback_and_group(
    project_root: Path,
    pattern: str,
    group_pattern: str,
    fallback_pattern: str,
    max_files_per_group: int,
) -> Dict[str, List[Path]]:
    """
    Shared rglob fallback: when explicit dirs had no files, search project and group by parent dir.
    Used by flatness, spectral isotropy, and any page with same (pattern, group_pattern) structure.
    """
    groups: Dict[str, List[Path]] = {}
    found = list(project_root.rglob(pattern))
    if not found:
        return groups
    by_parent: Dict[Path, List[Path]] = {}
    for f in found:
        p = f.parent if isinstance(f, Path) else Path(f).parent
        by_parent.setdefault(p, []).append(f if isinstance(f, Path) else Path(f))
    for parent, flist in by_parent.items():
        dir_name = parent.name
        sorted_files = sorted(flist, key=lambda x: _natural_sort_key(str(x)))
        file_strs = [str(x) for x in sorted_files[:max_files_per_group]]
        grouped = _group_files_by_simulation(file_strs, group_pattern)
        if not grouped:
            grouped = _group_files_by_simulation(file_strs, fallback_pattern)
        if grouped:
            for key, flist_inner in grouped.items():
                new_key = f"{dir_name}_{key}" if len(by_parent) > 1 else key
                if new_key not in groups:
                    groups[new_key] = []
                groups[new_key].extend([Path(f) for f in flist_inner[:max_files_per_group]])
        else:
            group_key = dir_name if len(by_parent) > 1 else "default"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].extend([Path(f) for f in file_strs[:max_files_per_group]])
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda f: _natural_sort_key(str(f)))
    return groups


def _rglob_fallback_multi_pattern(
    project_root: Path,
    configs: List[Tuple[str, str, str, str, int]],
) -> Dict[str, Tuple[List[Path], str]]:
    """
    Shared rglob fallback for pages with multiple patterns (e.g. structure functions: bin + txt).
    configs: [(pattern, group_re, fallback_re, kind, max_files_per_group), ...]
    Tries each config in order; returns first non-empty result as {sim_prefix: (paths, kind)}.
    """
    for pattern, group_re, fallback_re, kind, max_files in configs:
        g = _rglob_fallback_and_group(project_root, pattern, group_re, fallback_re, max_files)
        if g:
            return {k: (v, kind) for k, v in g.items()}
    return {}


def _manifest_kind_for_pattern(pattern: str) -> Optional[str]:
    """Map a glob pattern to a dataset-manifest file kind when possible."""
    lowered = pattern.lower()
    if "spectrum" in lowered:
        return "energy_spectrum"
    if "norm" in lowered:
        return "normalized_spectrum"
    return None


def _paths_from_manifest_index(sess: Dict[str, Any], pattern: str) -> List[Path]:
    """Return on-disk paths indexed in all_loaded_files for this pattern."""
    kind = _manifest_kind_for_pattern(pattern)
    if not kind:
        return []
    from analysis.manifest_index import MANIFEST_KIND_TO_SESSION_KEY

    session_key = MANIFEST_KIND_TO_SESSION_KEY.get(kind, kind)
    loaded = sess.get("all_loaded_files") or {}
    paths = [
        Path(item["full_path"])
        for item in loaded.get(session_key, [])
        if isinstance(item, dict) and item.get("full_path") and Path(item["full_path"]).is_file()
    ]
    return sorted(paths, key=lambda f: _natural_sort_key(str(f)))


def _paths_from_manifest_loader(
    sess: Dict[str, Any],
    pattern: str,
    project_root: Path,
) -> List[Path]:
    """Resolve files via manifest_path / simulation job when raw data_directory has no matches."""
    kind = _manifest_kind_for_pattern(pattern)
    if not kind:
        return []
    from analysis.product_loader import AnalysisProductLoader

    manifest_path = str(sess.get("manifest_path") or sess.get("dataset_manifest_path") or "").strip()
    if not manifest_path:
        job_id = str(sess.get("simulation_job_id") or sess.get("sim_workflow_job") or "").strip()
        if job_id:
            manifest_path = str((project_root / "simulations" / job_id / "manifest.json").resolve())

    if not manifest_path or not Path(manifest_path).is_file():
        return []
    loader = AnalysisProductLoader.from_manifest_path(project_root, manifest_path, sess)
    return loader.files_of_kind(kind)


def _is_examples_archive_shorthand(path_like: str) -> bool:
    """True for explicit archive shortcuts like DNS/512 or LES/64 (not OpenLB jobs)."""
    stripped = str(path_like or "").lstrip("/").replace("\\", "/")
    if stripped.lower().startswith("examples/"):
        stripped = stripped[9:]
    top = stripped.split("/", 1)[0].upper()
    return top in {"DNS", "LES"}


def _examples_archive_alt(project_root: Path, path_like: str) -> Optional[Path]:
    """Map DNS/LES shorthand to examples/<path> when that directory exists."""
    if not _is_examples_archive_shorthand(path_like):
        return None
    stripped = str(path_like).lstrip("/").replace("\\", "/")
    if stripped.lower().startswith("examples/"):
        candidate = (project_root / stripped).resolve()
    else:
        candidate = (project_root / "examples" / stripped).resolve()
    return candidate if candidate.is_dir() else None


def _under_examples_dns_les(path: Path, project_root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(Path(project_root).resolve())
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) >= 2 and parts[0] == "examples" and parts[1].upper() in {"DNS", "LES"}


def active_job_data_dirs(
    project_root: Path,
    session_context: Optional[Dict[str, Any]],
) -> List[str]:
    """Prefer raw/ then other active-job roots for analysis tools."""
    roots = _context_search_roots(project_root, session_context)
    if not roots:
        return []
    ordered: List[str] = []
    seen: set[str] = set()
    for preferred in roots:
        key = str(preferred.resolve())
        if key in seen or not preferred.is_dir():
            continue
        seen.add(key)
        ordered.append(key)
    # Prefer .../raw first when present.
    ordered.sort(key=lambda p: (0 if Path(p).name == "raw" else 1, p))
    return ordered


def prefer_active_job_over_examples_mix(
    dirs: List[str],
    project_root: Path,
    session_context: Optional[Dict[str, Any]],
) -> List[str]:
    """Prefer the active job when session dirs silently include examples/DNS|LES."""
    job = _session_job_id(session_context)
    if not job:
        return dirs
    job_root = str((Path(project_root) / "simulations" / job).resolve())
    job_dirs = [
        d for d in (dirs or [])
        if str(Path(d).resolve()).startswith(job_root)
    ]
    example_dirs = [
        d for d in (dirs or [])
        if _under_examples_dns_les(Path(d), project_root)
    ]
    if job_dirs and example_dirs:
        return job_dirs
    if example_dirs and not job_dirs:
        fallback = active_job_data_dirs(project_root, session_context)
        if fallback:
            return fallback
    return dirs


def _resolve_search_dirs(
    data_dir: str,
    pattern: str,
    project_root: Path,
    sess: Dict[str, Any],
) -> tuple[str, List[Path]]:
    """Collect directories to search, including processed spectra paths for OpenLB jobs."""
    search_dir = data_dir or sess.get("data_directory") or ""
    if not search_dir and sess.get("data_directories"):
        dirs = sess["data_directories"]
        search_dir = dirs[0] if isinstance(dirs, list) and dirs else ""
    if not search_dir:
        job_dirs = active_job_data_dirs(project_root, sess)
        if job_dirs:
            search_dir = job_dirs[0]

    search_paths: List[Path] = []
    seen: set[str] = set()

    def _add(path_like: str | Path) -> None:
        p = Path(path_like)
        if not p.is_absolute():
            p = (project_root / str(path_like).lstrip("/")).resolve()
        key = str(p)
        if key not in seen and p.is_dir():
            seen.add(key)
            search_paths.append(p)

    if search_dir:
        _add(search_dir)
        alt = _examples_archive_alt(project_root, search_dir)
        if alt is not None:
            _add(alt)

    if "spectrum" in pattern.lower():
        spec_dir = sess.get("spectra_data_directory")
        if spec_dir:
            _add(spec_dir)

    return search_dir, search_paths


def resolve_data_dir_and_find_files(
    data_dir: str,
    pattern: str,
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
    max_files: int = 100,
) -> List[Path]:
    """
    Resolve data directory and find files matching pattern.

    Uses session / active-job directories and manifest products. Does not scan the
    whole project (that silently mixed examples/DNS|LES into OpenLB runs).
    Archive shorthand DNS/* or LES/* still maps to examples/ when explicit.
    """
    sess = {} if session_context is None else session_context

    indexed = _paths_from_manifest_index(sess, pattern)
    if indexed:
        return indexed[:max_files]

    def _files_in_dir(d: Path) -> List[Path]:
        if not d.exists() or not d.is_dir():
            return []
        return sorted(d.glob(pattern), key=lambda f: _natural_sort_key(str(f)))

    search_dir, search_paths = _resolve_search_dirs(data_dir, pattern, project_root, sess)
    for p in search_paths:
        files = _files_in_dir(p)
        if files:
            return files[:max_files]

    manifest_paths = _paths_from_manifest_loader(sess, pattern, project_root)
    if manifest_paths:
        return sorted(manifest_paths, key=lambda f: _natural_sort_key(str(f)))[:max_files]

    # Selected or active-job directory is authoritative — do not invent another dataset.
    if search_dir:
        return []

    for job_dir in active_job_data_dirs(project_root, sess):
        files = _files_in_dir(Path(job_dir))
        if files:
            return files[:max_files]
    return []


def resolve_data_dirs_and_group_files(
    data_dirs: Optional[List[str]] = None,
    data_dir: str = "",
    pattern: str = "flatness_data*_*.txt",
    project_root: Optional[Path] = None,
    session_context: Optional[Dict[str, Any]] = None,
    max_files_per_group: int = 1000,
    group_pattern: Optional[str] = None,
    fallback_pattern: Optional[str] = None,
) -> Dict[str, List[Path]]:
    """
    Resolve data directories and group files by simulation.
    Generic version for flatness, etc. Returns {sim_prefix: [Path, ...]}.
    """
    sess = {} if session_context is None else session_context
    project_root = project_root or Path(".")

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
    else:
        dirs_to_search = active_job_data_dirs(project_root, sess)
    dirs_to_search = prefer_active_job_over_examples_mix(
        dirs_to_search, project_root, sess
    )

    # Flatness-specific group patterns when not provided
    if group_pattern is None and "flatness" in pattern:
        group_pattern = r"(flatness_data\d+)_t\d+\.txt"
    if fallback_pattern is None and "flatness" in pattern:
        fallback_pattern = r"(flatness_data\d+)_\d+\.txt"
    if group_pattern is None:
        group_pattern = r"(\w+)_\d+\.\w+"
    if fallback_pattern is None:
        fallback_pattern = group_pattern

    groups: Dict[str, List[Path]] = {}
    for search_dir in dirs_to_search:
        if not search_dir:
            continue
        p = Path(search_dir)
        if not p.is_absolute():
            p = (project_root / search_dir.lstrip("/")).resolve()
        if not p.exists() or not p.is_dir():
            alt = _examples_archive_alt(project_root, search_dir)
            if alt is not None:
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
                if new_key not in groups:
                    groups[new_key] = []
                groups[new_key].extend([Path(f) for f in flist[:max_files_per_group]])
        else:
            group_key = p.name if len(dirs_to_search) > 1 else "default"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].extend([Path(f) for f in file_strs[:max_files_per_group]])

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda f: _natural_sort_key(str(f)))

    if not groups:
        flat = resolve_data_dir_and_find_files(
            data_dir, pattern, project_root, session_context, max_files_per_group
        )
        if flat:
            file_strs = [str(f) for f in flat]
            grouped = _group_files_by_simulation(file_strs, group_pattern)
            if not grouped:
                grouped = _group_files_by_simulation(file_strs, fallback_pattern)
            if grouped:
                for key, flist in grouped.items():
                    groups[key] = [Path(f) for f in flist[:max_files_per_group]]
            else:
                groups["default"] = flat[:max_files_per_group]

    # Do not cross-load another simulation when explicit directories were selected.

    return groups


def resolve_data_dirs_and_group_structure_functions(
    data_dirs: Optional[List[str]] = None,
    data_dir: str = "",
    project_root: Optional[Path] = None,
    session_context: Optional[Dict[str, Any]] = None,
    max_files_per_group: int = 1000,
) -> Dict[str, Tuple[List[Path], str]]:
    """
    Resolve structure function files (bin + txt) and group by simulation.
    Returns {sim_prefix: (files, kind)} where kind is "bin" or "txt".

    OpenLB products use names like ``structure_functions1_t500.txt`` (no underscore
    before the series index), so the txt glob is ``structure_functions*.txt``.
    """
    from utils.file_detector import expand_analysis_search_dirs

    sess = {} if session_context is None else session_context
    project_root = project_root or Path(".")

    dirs_to_search: List[str] = []
    if data_dirs and isinstance(data_dirs, list) and len(data_dirs) > 0:
        dirs_to_search = list(data_dirs)
    elif data_dir:
        dirs_to_search = [data_dir]
    elif sess.get("structure_functions_data_directory"):
        dirs_to_search = [str(sess["structure_functions_data_directory"])]
    elif sess.get("data_directories"):
        d = sess["data_directories"]
        dirs_to_search = list(d) if isinstance(d, list) else [d]
    elif sess.get("data_directory"):
        dirs_to_search = [sess["data_directory"]]
    else:
        dirs_to_search = active_job_data_dirs(project_root, sess)
    dirs_to_search = prefer_active_job_over_examples_mix(
        dirs_to_search, project_root, sess
    )

    # Prefer files already indexed when the manifest/session was loaded.
    indexed_bin = list((sess.get("all_loaded_files") or {}).get("structure_functions_bin") or [])
    indexed_txt = list((sess.get("all_loaded_files") or {}).get("structure_functions_txt") or [])
    if indexed_bin or indexed_txt:
        kind = "bin" if indexed_bin else "txt"
        entries = indexed_bin if indexed_bin else indexed_txt
        file_strs = []
        for item in entries:
            if isinstance(item, dict):
                path = item.get("full_path") or item.get("path")
            else:
                path = str(item)
            if path:
                file_strs.append(str(path))
        if file_strs:
            bin_group_re = r"(structure_funcs\d+)_t\d+\.bin"
            txt_group_re = r"(structure_functions\d+)_t\d+\.txt"
            group_re = bin_group_re if kind == "bin" else txt_group_re
            grouped = _group_files_by_simulation(file_strs, group_re)
            if not grouped:
                grouped = {"structure_functions": file_strs}
            return {
                key: ([Path(f) for f in flist[:max_files_per_group]], kind)
                for key, flist in grouped.items()
            }

    groups: Dict[str, Tuple[List[Path], str]] = {}
    bin_pattern = "structure_funcs*_t*.bin"
    # Match both structure_functions_1_t*.txt and OpenLB structure_functions1_t*.txt
    txt_pattern = "structure_functions*.txt"
    bin_group_re = r"(structure_funcs\d+)_t\d+\.bin"
    txt_group_re = r"(structure_functions\d+)_t\d+\.txt"

    expanded_dirs: List[Path] = []
    seen_dirs: set[str] = set()
    for search_dir in dirs_to_search:
        if not search_dir:
            continue
        p = Path(search_dir)
        if not p.is_absolute():
            p = (project_root / search_dir.lstrip("/")).resolve()
        if not p.exists() or not p.is_dir():
            alt = _examples_archive_alt(project_root, search_dir)
            if alt is not None:
                p = alt
            else:
                continue
        for candidate in expand_analysis_search_dirs(p):
            key = str(candidate.resolve())
            if key not in seen_dirs:
                seen_dirs.add(key)
                expanded_dirs.append(candidate)

    for p in expanded_dirs:
        # Prefer bin, fallback to txt
        bin_files = sorted(p.glob(bin_pattern), key=lambda f: _natural_sort_key(str(f)))
        txt_files = sorted(p.glob(txt_pattern), key=lambda f: _natural_sort_key(str(f)))
        file_strs = [str(f) for f in bin_files] if bin_files else [str(f) for f in txt_files]
        kind = "bin" if bin_files else "txt"
        group_re = bin_group_re if bin_files else txt_group_re

        if not file_strs:
            continue

        grouped = _group_files_by_simulation(file_strs, group_re)
        if not grouped:
            grouped = _group_files_by_simulation(file_strs, r"(structure_funcs_data\d+)_t\d+\.bin")
        if not grouped:
            grouped = {"structure_funcs": file_strs} if bin_files else {"structure_functions": file_strs}

        dir_name = p.name
        for key, flist in grouped.items():
            new_key = f"{dir_name}_{key}" if len(expanded_dirs) > 1 else key
            paths = [Path(f) for f in flist[:max_files_per_group]]
            groups[new_key] = (paths, kind)

    return groups


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
    sess = {} if session_context is None else session_context
    project_root = project_root or Path(".")

    # Resolve search dirs: data_dirs > data_dir > session > active job
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
    else:
        dirs_to_search = active_job_data_dirs(project_root, sess)
    dirs_to_search = prefer_active_job_over_examples_mix(
        dirs_to_search, project_root, sess
    )

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
            alt = _examples_archive_alt(project_root, search_dir)
            if alt is not None:
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

    # Explicit directories are authoritative; do not silently use another grid.

    return ic_groups


def save_to_cache(session_context: Dict[str, Any], key: str, data: Any) -> None:
    """Store heavy data in session cache to avoid flooding LLM context."""
    cache = session_context.setdefault("agent_data_cache", {})
    cache[key] = data


def get_from_cache(session_context: Dict[str, Any], key: str) -> Any:
    """Retrieve data from session cache."""
    return session_context.get("agent_data_cache", {}).get(key)


def update_data_directory_in_context(
    session_context: Optional[Dict[str, Any]],
    data_dir_path,
    data_dirs_list: Optional[List[str]] = None,
) -> None:
    """Update session_context with the data directory used by a tool.

    Enables manual pages (PDFs, Spectra, etc.) to render after an agent run.
    When data_dirs_list is provided (multi-sim), sets data_directories so the
    manual page shows all curves.
    """
    if not session_context:
        return
    if data_dirs_list and len(data_dirs_list) > 0:
        resolved = []
        for d in data_dirs_list:
            p = Path(d).resolve()
            if p.exists() and p.is_dir() and str(p) not in resolved:
                resolved.append(str(p))
        if resolved:
            session_context["data_directories"] = resolved
            session_context["data_directory"] = resolved[0]
        return
    if data_dir_path is not None:
        p = Path(data_dir_path).resolve()
        if p.exists():
            session_context["data_directory"] = str(p)


def resolve_path(filepath: str, project_root: Path) -> Path:
    """Resolve path to project. Raises ValueError if outside project."""
    p = (project_root / filepath).resolve()
    if ".." in filepath or not str(p).startswith(str(project_root.resolve())):
        raise ValueError("Path must be inside project")
    return p


def _session_job_id(session_context: Optional[Dict[str, Any]]) -> str:
    ctx = session_context or {}
    job = str(ctx.get("simulation_job_id") or ctx.get("sim_workflow_job") or "").strip()
    if job:
        return job
    mem = ctx.get("turn_memory") if isinstance(ctx.get("turn_memory"), dict) else {}
    return str(mem.get("job_id") or "").strip()


def _context_search_roots(
    project_root: Path,
    session_context: Optional[Dict[str, Any]],
) -> List[Path]:
    """Directories implied by active job + turn_memory (for follow-up bare filenames)."""
    root = Path(project_root).resolve()
    ctx = session_context or {}
    mem = ctx.get("turn_memory") if isinstance(ctx.get("turn_memory"), dict) else {}
    roots: List[Path] = []

    job = _session_job_id(ctx)
    if job:
        job_dir = root / "simulations" / job
        if job_dir.is_dir():
            roots.append(job_dir)
            for sub in ("executable", "raw", "processed", "case"):
                candidate = job_dir / sub
                if candidate.is_dir():
                    roots.append(candidate)

    for path in list(mem.get("last_paths") or []) + list(ctx.get("last_paths") or []):
        rel = str(path or "").strip().replace("\\", "/")
        if not rel:
            continue
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if p.is_file():
            roots.append(p.parent)
        elif p.is_dir():
            roots.append(p)

    seen: set[str] = set()
    out: List[Path] = []
    for item in roots:
        try:
            key = str(item.resolve())
        except Exception:
            continue
        if key in seen or not Path(key).is_dir():
            continue
        if not key.startswith(str(root)):
            continue
        seen.add(key)
        out.append(Path(key))
    return out


def resolve_existing_project_file(
    filepath: str,
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Resolve a readable file path, including bare names against active job context.

    If the path is missing as given, search under ``simulations/<job_id>/…`` and
    last_paths parents for the same basename — any filename, not a fixed list.
    """
    raw = (filepath or "").strip().replace("\\", "/")
    if not raw:
        return None
    root = Path(project_root).resolve()
    direct = Path(raw) if Path(raw).is_absolute() else root / raw
    try:
        if direct.is_file() and str(direct.resolve()).startswith(str(root)):
            return direct.resolve()
    except Exception:
        pass

    name = Path(raw).name
    if not name or name in {".", ".."}:
        return None

    matches: List[Path] = []
    for search_root in _context_search_roots(root, session_context):
        hit = search_root / name
        if hit.is_file():
            matches.append(hit.resolve())
            continue
        try:
            for found in search_root.rglob(name):
                if found.is_file():
                    matches.append(found.resolve())
                    if len(matches) >= 12:
                        break
        except Exception:
            continue
        if len(matches) >= 12:
            break

    uniq: List[Path] = []
    seen: set[str] = set()
    for match in matches:
        key = str(match)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(match)
    if not uniq:
        return None
    if len(uniq) == 1:
        return uniq[0]

    # Prefer the shallowest path under the active job when several matches exist.
    job = _session_job_id(session_context)
    if job:
        job_prefix = str((root / "simulations" / job).resolve())
        under_job = [p for p in uniq if str(p).startswith(job_prefix)]
        if under_job:
            under_job.sort(key=lambda p: (len(p.parts), str(p)))
            return under_job[0]
    uniq.sort(key=lambda p: (len(p.parts), str(p)))
    return uniq[0]
