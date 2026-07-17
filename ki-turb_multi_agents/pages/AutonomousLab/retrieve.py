"""
Retrieve (undo) snapshots for agent file modifications in Autonomous Lab.

Captures original file state *before* an accepted write/modify/delete/rename,
then lets the user restore those originals ("Retrieve previous version").
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REVERTABLE_TOOLS = frozenset({
    "write_file",
    "modify_file",
    "create_file",
    "delete_file",
    "rename_file",
})

MAX_RETRIEVE_STACK = 20


def resolve_filepath(filepath: str, project_root: Path) -> Path:
    p = Path(filepath)
    if not p.is_absolute():
        p = project_root / p
    return p


def _read_text_safe(path: Path) -> Optional[str]:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def capture_retrieve_entry(
    tool: str,
    args: Dict[str, Any],
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    Snapshot one file-modifying tool call before it runs.

    Returns a retrieve entry, or None if the tool is not revertable / unsupported.
    """
    tool = (tool or "").strip()
    if tool not in _REVERTABLE_TOOLS:
        return None
    args = args or {}

    if tool == "rename_file":
        filepath = str(args.get("filepath") or "").strip()
        new_filepath = str(args.get("new_filepath") or "").strip()
        if not filepath or not new_filepath:
            return None
        src = resolve_filepath(filepath, project_root)
        dst = resolve_filepath(new_filepath, project_root)
        if not src.exists() or not src.is_file():
            return None
        return {
            "kind": "rename",
            "tool": tool,
            "filepath": str(src),
            "new_filepath": str(dst),
            "filename": src.name,
            "file_existed": True,
            "original_content": None,
            "summary": f"rename `{src.name}` → `{dst.name}`",
        }

    filepath = str(args.get("filepath") or "").strip()
    if not filepath:
        return None
    path = resolve_filepath(filepath, project_root)

    if tool == "delete_file":
        if not path.exists():
            return None
        if path.is_dir():
            # Full directory trees are not snapshotted (too large / unsafe).
            return None
        if not path.is_file() and not path.is_symlink():
            return None
        content = _read_text_safe(path)
        if content is None and path.is_file():
            # Binary or unreadable — cannot safely retrieve as text.
            return None
        return {
            "kind": "delete",
            "tool": tool,
            "filepath": str(path),
            "new_filepath": None,
            "filename": path.name,
            "file_existed": True,
            "original_content": content if content is not None else "",
            "summary": f"delete `{path.name}`",
        }

    if tool == "create_file":
        return {
            "kind": "create",
            "tool": tool,
            "filepath": str(path),
            "new_filepath": None,
            "filename": path.name,
            "file_existed": False,
            "original_content": "",
            "summary": f"create `{path.name}`",
        }

    # write_file / modify_file
    existed = path.exists() and path.is_file()
    if tool == "modify_file" and not existed:
        return None
    content = _read_text_safe(path) if existed else ""
    if existed and content is None:
        return None
    kind = "modify" if existed else "create"
    return {
        "kind": kind,
        "tool": tool,
        "filepath": str(path),
        "new_filepath": None,
        "filename": path.name,
        "file_existed": existed,
        "original_content": content if content is not None else "",
        "summary": f"{'overwrite' if existed else 'create'} `{path.name}`",
    }


def capture_retrieve_batch(
    action_requests: List[Dict[str, Any]],
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """Capture a batch of retrieve entries from HITL action_requests."""
    entries: List[Dict[str, Any]] = []
    for action in action_requests or []:
        name = action.get("name") or action.get("tool") or ""
        args = action.get("args") or action.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        entry = capture_retrieve_entry(str(name), args, project_root)
        if entry:
            entries.append(entry)
    if not entries:
        return None
    names = [e.get("filename") or Path(e["filepath"]).name for e in entries]
    label = names[0] if len(names) == 1 else f"{names[0]} +{len(names) - 1} more"
    return {
        "id": f"retrieve-{int(time.time() * 1000)}",
        "captured_at": time.time(),
        "entries": entries,
        "label": label,
        "count": len(entries),
    }


def apply_retrieve_entry(entry: Dict[str, Any]) -> str:
    """
    Restore one entry. Returns a short human-readable result label.
    Raises OSError / ValueError on failure.
    """
    kind = entry.get("kind") or ""
    filepath = Path(entry["filepath"])
    filename = entry.get("filename") or filepath.name

    if kind == "rename":
        new_filepath = Path(entry["new_filepath"])
        if new_filepath.exists() and new_filepath.is_file():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            new_filepath.rename(filepath)
            return f"renamed `{new_filepath.name}` → `{filename}`"
        if filepath.exists():
            return f"already at `{filename}`"
        raise FileNotFoundError(f"Cannot undo rename; `{new_filepath.name}` not found")

    if kind == "create" or not entry.get("file_existed", True):
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            return f"removed `{filename}`"
        return f"`{filename}` already absent"

    # modify / overwrite / delete → restore original content
    original = entry.get("original_content")
    if original is None:
        raise ValueError(f"No snapshot content for `{filename}`")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(original, encoding="utf-8")
    if kind == "delete":
        return f"restored deleted `{filename}`"
    return f"restored `{filename}`"


def apply_retrieve_batch(batch: Dict[str, Any]) -> List[str]:
    """Apply all entries in reverse order (safer for multi-file batches)."""
    entries = list(batch.get("entries") or [])
    results: List[str] = []
    for entry in reversed(entries):
        results.append(apply_retrieve_entry(entry))
    return results


def push_retrieve_batch(stack: Optional[List[Dict[str, Any]]], batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Push a batch onto the retrieve stack (newest last), capped at MAX_RETRIEVE_STACK."""
    out = list(stack or [])
    out.append(batch)
    if len(out) > MAX_RETRIEVE_STACK:
        out = out[-MAX_RETRIEVE_STACK:]
    return out


def batch_summary_lines(batch: Dict[str, Any]) -> List[str]:
    lines = []
    for entry in batch.get("entries") or []:
        lines.append(str(entry.get("summary") or entry.get("filename") or "change"))
    return lines


__all__ = [
    "MAX_RETRIEVE_STACK",
    "apply_retrieve_batch",
    "apply_retrieve_entry",
    "batch_summary_lines",
    "capture_retrieve_batch",
    "capture_retrieve_entry",
    "push_retrieve_batch",
    "resolve_filepath",
]
