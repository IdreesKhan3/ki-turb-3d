"""
Regex Search Tool — Pattern matching with context, and regex replace in files.
Self-contained; does not depend on extra/.
"""

import re
from pathlib import Path
from typing import Dict, Any, List

SKIP_DIRS = {'venv', 'myenv', '__pycache__', '.git', 'node_modules', '.venv', 'local_tools'}


def regex_search(
    project_root: Path,
    pattern: str,
    file_pattern: str = "*.py",
    context_lines: int = 2,
    max_results: int = 50,
    case_sensitive: bool = True,
) -> Dict[str, Any]:
    """Search for regex pattern in files. Returns matches with context."""
    matches: List[Dict[str, Any]] = []
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"ok": False, "message": f"Invalid regex pattern: {e}", "matches": []}

    matching_files = list(project_root.rglob(file_pattern))
    for filepath in matching_files:
        if any(part in filepath.parts for part in SKIP_DIRS):
            continue
        try:
            lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line_num, line in enumerate(lines, start=1):
                for match in regex.finditer(line):
                    if len(matches) >= max_results:
                        break
                    before = lines[max(0, line_num - context_lines - 1) : line_num - 1] if context_lines else []
                    after = lines[line_num : line_num + context_lines] if context_lines else []
                    matches.append({
                        "file": str(filepath.relative_to(project_root)),
                        "line": line_num,
                        "content": line,
                        "matched_text": match.group(0),
                        "groups": list(match.groups()) if match.groups() else [],
                        "before_context": before,
                        "after_context": after,
                    })
                if len(matches) >= max_results:
                    break
        except (UnicodeDecodeError, PermissionError):
            continue
        if len(matches) >= max_results:
            break

    return {
        "ok": True,
        "message": f"Found {len(matches)} match(es) for pattern '{pattern}'",
        "matches": matches,
    }


def replace_regex(
    project_root: Path,
    filepath: str,
    pattern: str,
    replacement: str,
    case_sensitive: bool = True,
) -> Dict[str, Any]:
    """Replace regex matches in a file. Uses \\1, \\2 for capture groups."""
    if ".." in filepath:
        return {"ok": False, "message": "Path must be inside project", "replacements": 0}
    full_path = (project_root / filepath).resolve()
    if not full_path.exists():
        return {"ok": False, "message": f"File not found: {filepath}", "replacements": 0}
    if not str(full_path).startswith(str(project_root.resolve())):
        return {"ok": False, "message": "Path must be inside project", "replacements": 0}

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"ok": False, "message": f"Invalid regex pattern: {e}", "replacements": 0}

    try:
        content = full_path.read_text(encoding="utf-8")
        new_content, count = regex.subn(replacement, content)
        if count == 0:
            return {"ok": True, "message": f"No matches for pattern '{pattern}'", "replacements": 0}
        full_path.write_text(new_content, encoding="utf-8")
        return {
            "ok": True,
            "message": f"Replaced {count} occurrence(s) in {filepath}",
            "replacements": count,
        }
    except Exception as e:
        return {"ok": False, "message": str(e), "replacements": 0}
