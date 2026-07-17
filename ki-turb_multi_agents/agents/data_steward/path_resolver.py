"""Path resolution helpers for ActionExecutor."""
from pathlib import Path
from typing import Optional, Dict, Any
import difflib


def should_skip_path(path: Path, skip_dirs: set) -> bool:
    """True if any path part is in skip_dirs."""
    return any(skip in path.parts for skip in skip_dirs)


def normalize_path(filepath: str, project_root: Path) -> str:
    """Return a project-relative, platform-agnostic path string.

    Treats leading-slash paths like ``/examples`` as project-relative
    (``project_root/examples``) so the same agent paths work on Windows and Linux.
    """
    if not filepath:
        return ""

    try:
        path_obj = Path(filepath.strip())
    except Exception:
        return filepath.strip()

    # Prefer project-relative interpretation (handles "/examples" on Linux).
    potential_rel_path = project_root.joinpath(
        path_obj.anchor and path_obj.as_posix().lstrip('/') or str(path_obj)
    )
    if potential_rel_path.exists():
        try:
            return str(potential_rel_path.relative_to(project_root))
        except ValueError:
            return str(potential_rel_path)

    # Absolute path that exists: prefer relative-to-project when possible.
    if path_obj.is_absolute() and path_obj.exists():
        try:
            return str(path_obj.relative_to(project_root))
        except ValueError:
            return str(path_obj)

    # Non-existent paths (e.g. create_file): strip leading slash → project-relative.
    if path_obj.is_absolute():
        return path_obj.as_posix().lstrip('/')

    return str(path_obj)


def resolve_path(
    filepath: str,
    project_root: Path,
    resolver,
    skip_dirs: set,
    must_be_dir: bool = False,
    allow_nonexistent: bool = False
) -> Optional[Path]:
    """Resolve filepath via normalize → exact → Resolver → name/fuzzy match."""
    if not filepath:
        return None

    search_path = normalize_path(filepath, project_root)

    # "./APP/file.py" when APP is the project root
    parts = Path(search_path).parts
    if parts and (parts[0] == "APP" or parts[0] == "."):
        potential_path = Path(*parts[1:])
        test_path = project_root / potential_path
        if test_path.exists() and (not must_be_dir or test_path.is_dir()):
            return test_path

    path = Path(search_path)
    if path.is_absolute():
        if allow_nonexistent or path.exists():
            if not must_be_dir or (path.exists() and path.is_dir()):
                return path

    path = project_root / search_path
    if (allow_nonexistent or path.exists()) and (not must_be_dir or (path.exists() and path.is_dir())):
        return path

    try:
        resolved = resolver.resolve_with_fallback(search_path, must_be_dir=must_be_dir, max_attempts=3)
        if resolved:
            return resolved
    except Exception:
        pass

    if not must_be_dir:
        target_name = Path(search_path).name
        candidates = [
            p for p in project_root.rglob(target_name)
            if p.is_file() and not should_skip_path(p, skip_dirs)
        ]
        if candidates:
            return candidates[0]

    all_paths = []
    try:
        for item in project_root.rglob("*"):
            if should_skip_path(item, skip_dirs):
                continue
            if item.name.startswith('.'):
                continue
            if must_be_dir and not item.is_dir():
                continue
            if not must_be_dir and item.is_dir():
                continue

            try:
                rel_path = str(item.relative_to(project_root))
                all_paths.append((rel_path, item))
            except ValueError:
                continue
    except Exception:
        pass

    if all_paths:
        path_strings = [p[0] for p in all_paths]
        close_matches = difflib.get_close_matches(search_path, path_strings, n=1, cutoff=0.6)
        if close_matches:
            matched_path_str = close_matches[0]
            for path_str, path_obj in all_paths:
                if path_str == matched_path_str:
                    return path_obj

    if allow_nonexistent:
        return project_root / search_path

    return None


def format_error(context: str, error: Exception) -> Dict[str, Any]:
    return {"success": False, "message": f"{context}: {str(error)}"}


def validate_path_exists(path: Path, must_be_dir: bool = False) -> Optional[Dict[str, Any]]:
    if not path or not path.exists():
        return {"success": False, "message": f"Path not found: {path}"}
    if must_be_dir and not path.is_dir():
        return {"success": False, "message": f"Path is not a directory: {path}"}
    if not must_be_dir and path.is_dir():
        return {"success": False, "message": f"Path is a directory, expected file: {path}"}
    return None
