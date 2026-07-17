"""
Diff preview utilities for Autonomous Lab.
Computes unified diff and content for modify_file, write_file, create_file confirmations.
"""

import difflib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _resolve_filepath(filepath: str, project_root: Path) -> Path:
    """Resolve filepath relative to project root."""
    p = Path(filepath)
    if not p.is_absolute():
        p = project_root / p
    return p


def compute_modify_file_diff(
    args: Dict[str, Any],
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    Compute current content, new content, and unified diff for a modify_file action.
    Returns dict with current_content, new_content, diff_text, filename, or None on error.
    """
    filepath = args.get("filepath", "")
    if not filepath:
        return None
    path = _resolve_filepath(filepath, project_root)
    if not path.exists() or not path.is_file():
        return None
    try:
        current_content = path.read_text(encoding="utf-8")
    except Exception:
        return None
    new_content = args.get("new_content")
    search_text = args.get("search_text")
    replace_text = args.get("replace_text")
    if new_content is not None:
        pass  # use as-is
    elif search_text is not None and replace_text is not None:
        new_content = current_content.replace(search_text, replace_text)
    else:
        return None
    current_lines = current_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_text = "".join(
        difflib.unified_diff(
            current_lines,
            new_lines,
            fromfile=f"Original: {path.name}",
            tofile=f"Modified: {path.name}",
            lineterm="",
        )
    )
    return {
        "filename": path.name,
        "filepath": str(path),
        "current_content": current_content,
        "new_content": new_content,
        "diff_text": diff_text.strip() or "(no changes)",
        "mode": "search_replace" if (search_text and replace_text) else "full_rewrite",
    }


def compute_write_file_diff(
    args: Dict[str, Any],
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    Compute diff for write_file (create or overwrite).
    Returns dict with current_content (or empty), new_content, diff_text, filename.
    """
    filepath = args.get("filepath", "")
    content = args.get("content", "")
    if not filepath:
        return None
    path = _resolve_filepath(filepath, project_root)
    current_content = path.read_text(encoding="utf-8") if path.exists() and path.is_file() else ""
    new_content = content
    current_lines = current_content.splitlines(keepends=True) if current_content else []
    new_lines = new_content.splitlines(keepends=True) if new_content else []
    diff_text = "".join(
        difflib.unified_diff(
            current_lines,
            new_lines,
            fromfile=f"Original: {path.name}" if current_content else "(new file)",
            tofile=f"New: {path.name}",
            lineterm="",
        )
    )
    return {
        "filename": path.name,
        "filepath": str(path),
        "current_content": current_content,
        "new_content": new_content,
        "diff_text": diff_text.strip() or "(no changes)",
        "mode": "overwrite" if current_content else "create",
    }


def compute_create_file_preview(
    args: Dict[str, Any],
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    Preview for create_file (new file only).
    Returns dict with content, filename, line_count.
    """
    filepath = args.get("filepath", "")
    content = args.get("content", "")
    if not filepath:
        return None
    path = _resolve_filepath(filepath, project_root)
    if path.exists():
        return {"filename": path.name, "content": content, "line_count": len(content.splitlines()), "exists": True}
    return {
        "filename": path.name,
        "filepath": str(path),
        "content": content,
        "line_count": len(content.splitlines()),
        "exists": False,
    }


def get_diff_for_pending_tool(
    tool: str,
    args: Dict[str, Any],
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    Get diff/preview data for a pending confirmable tool.
    Returns dict suitable for rendering in the confirmation UI.
    """
    if tool == "modify_file":
        return compute_modify_file_diff(args, project_root)
    if tool == "write_file":
        return compute_write_file_diff(args, project_root)
    if tool == "create_file":
        return compute_create_file_preview(args, project_root)
    return None
