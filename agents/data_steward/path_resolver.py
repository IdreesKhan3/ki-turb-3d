"""
Path resolution utilities for ActionExecutor
"""
from pathlib import Path
from typing import Optional, Dict, Any
import difflib


def should_skip_path(path: Path, skip_dirs: set) -> bool:
    """Check if a path should be skipped during searches"""
    return any(skip in path.parts for skip in skip_dirs)


def normalize_path(filepath: str, project_root: Path) -> str:
    """
    Normalize file paths to be relative to the project root and platform-agnostic.
    This function is critical for ensuring the agent works correctly on both
    Windows and Linux, and for interpreting ambiguous user paths like "/examples".
    """
    if not filepath:
        return ""

    # Use pathlib for robust parsing
    try:
        path_obj = Path(filepath.strip())
    except Exception:
        # Handle invalid path strings gracefully
        return filepath.strip()

    # 1. Check if the path, when interpreted as relative to project_root, exists.
    # This correctly handles "/examples" on Linux by checking "project_root/examples".
    # It also handles "examples/test.F90" directly.
    potential_rel_path = project_root.joinpath(path_obj.anchor and path_obj.as_posix().lstrip('/') or str(path_obj))
    if potential_rel_path.exists():
        # If it exists, we are confident this is what the user meant.
        # Return it as a relative, platform-agnostic string.
        try:
            return str(potential_rel_path.relative_to(project_root))
        except ValueError:
            # This can happen if the path is outside the project, but we already
            # established it exists within, so this case is unlikely.
            return str(potential_rel_path)

    # 2. If it's an absolute path that exists on the system, respect it,
    # but return it relative to the project root if possible for consistency.
    if path_obj.is_absolute() and path_obj.exists():
        try:
            return str(path_obj.relative_to(project_root))
        except ValueError:
            # The absolute path is outside the project root.
            # The caller (e.g., ActionExecutor) should handle this security risk.
            return str(path_obj)

    # 3. Fallback for non-existent paths (e.g., for create_file).
    # Strip leading slashes to treat it as project-relative.
    if path_obj.is_absolute():
        return path_obj.as_posix().lstrip('/')
    
    # 4. If it's already a relative path, just return it.
    return str(path_obj)


def resolve_path(
    filepath: str,
    project_root: Path,
    resolver,
    skip_dirs: set,
    must_be_dir: bool = False,
    allow_nonexistent: bool = False
) -> Optional[Path]:
    """
    Centralized path resolution helper with intelligent search
    
    Args:
        filepath: Path string (absolute, relative, or fuzzy)
        project_root: Project root directory
        resolver: Resolver instance for enhanced path finding
        skip_dirs: Set of directory names to skip
        must_be_dir: If True, path must be a directory
        allow_nonexistent: If True, return path even if it doesn't exist
    
    Returns:
        Resolved Path object or None
    """
    if not filepath:
        return None
    
    # Normalize path first (handles LLM-style absolute paths, './', 'APP/' prefix)
    search_path = normalize_path(filepath, project_root)
    
    # 0. Handle "./APP/file.py" pattern where "APP" is actually the root
    parts = Path(search_path).parts
    if parts and (parts[0] == "APP" or parts[0] == "."):
        potential_path = Path(*parts[1:])
        test_path = project_root / potential_path
        if test_path.exists() and (not must_be_dir or test_path.is_dir()):
            return test_path
    
    # 1. Try exact match (absolute or relative) - fast path
    path = Path(search_path)
    if path.is_absolute():
        if allow_nonexistent or path.exists():
            # If must_be_dir, verify it's actually a directory (only if path exists)
            if not must_be_dir or (path.exists() and path.is_dir()):
                return path
    
    # Try relative to project root
    path = project_root / search_path
    if (allow_nonexistent or path.exists()) and (not must_be_dir or (path.exists() and path.is_dir())):
        return path
    
    # 2. Use Resolver for enhanced path finding (symbol search, content indexing, multi-signal ranking)
    # Resolver should handle errors internally, but catch any unexpected exceptions
    try:
        resolved = resolver.resolve_with_fallback(search_path, must_be_dir=must_be_dir, max_attempts=3)
        if resolved:
            return resolved
    except Exception:
        # If Resolver fails unexpectedly, fall back to legacy method
        pass
    
    # 3. Legacy fallback: Simple filename match
    if not must_be_dir:
        target_name = Path(search_path).name
        candidates = [
            p for p in project_root.rglob(target_name)
            if p.is_file() and not should_skip_path(p, skip_dirs)
        ]
        if candidates:
            return candidates[0]
    
    # 4. Legacy fallback: Fuzzy match using difflib
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
    
    # If allow_nonexistent, return the relative path anyway (use normalized search_path)
    if allow_nonexistent:
        return project_root / search_path
    
    return None


def format_error(context: str, error: Exception) -> Dict[str, Any]:
    """Format error messages consistently"""
    return {"success": False, "message": f"{context}: {str(error)}"}


def validate_path_exists(path: Path, must_be_dir: bool = False) -> Optional[Dict[str, Any]]:
    """Validate that a path exists and matches requirements"""
    if not path or not path.exists():
        return {"success": False, "message": f"Path not found: {path}"}
    if must_be_dir and not path.is_dir():
        return {"success": False, "message": f"Path is not a directory: {path}"}
    if not must_be_dir and path.is_dir():
        return {"success": False, "message": f"Path is a directory, expected file: {path}"}
    return None
