"""
Core agent tools: file ops, list, read, write, modify, delete, find, search_codebase, extract_section.
"""

from pathlib import Path
from typing import Any, Dict, List

from ._shared import resolve_path


CORE_TOOL_NAMES = frozenset({
    "list_directory", "read_file", "write_file",
    "modify_file", "delete_file", "rename_file", "find_file",
    "search_codebase", "extract_section",
})


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for core tools."""
    return [
        {
            "name": "list_directory",
            "description": "List contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path"}},
            },
        },
        {
            "name": "read_file",
            "description": "Read contents of a file. Use start_line/end_line for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"},
                    "start_line": {"type": "integer", "description": "First line (1-based)"},
                    "end_line": {"type": "integer", "description": "Last line (1-based)"},
                },
            },
        },
        {
            "name": "write_file",
            "description": "Create or overwrite a file. Path relative to project (e.g. examples/test.dat).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
            },
        },
        {
            "name": "modify_file",
            "description": "Edit a file: new_content (full rewrite) or search_text/replace_text (exact substring). For regex, use regex_search and replace_regex tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"},
                    "new_content": {"type": "string", "description": "Full new content (for full rewrite)"},
                    "search_text": {"type": "string", "description": "Exact text to find"},
                    "replace_text": {"type": "string", "description": "Replacement text"},
                },
            },
        },
        {
            "name": "delete_file",
            "description": "Delete a file. Path relative to project.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string", "description": "Path to file"}},
            },
        },
        {
            "name": "rename_file",
            "description": "Rename or move a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Current path"},
                    "new_filepath": {"type": "string", "description": "New path"},
                },
            },
        },
        {
            "name": "find_file",
            "description": "Locate files by filename or glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Filename or glob (e.g. *.py, config.json)"},
                    "directory": {"type": "string", "description": "Directory to search (default: project root)"},
                },
            },
        },
        {
            "name": "search_codebase",
            "description": "Grep-style content search across files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for"},
                    "file_pattern": {"type": "string", "description": "File glob (default *)"},
                },
            },
        },
        {
            "name": "extract_section",
            "description": "Extract a section from a file by query (finds first match, returns surrounding lines).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"},
                    "query": {"type": "string", "description": "Text to find"},
                    "context_lines": {"type": "integer", "description": "Lines after match (default 30)"},
                },
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    """Execute a core tool. Returns result string."""
    if name == "list_directory":
        path = args.get("path", ".")
        p = Path(path)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists() or not p.is_dir():
            # Try examples/ prefix (e.g. LES/64 -> examples/LES/64)
            alt = project_root / "examples" / path.lstrip("/")
            if alt.exists() and alt.is_dir():
                p = alt
            else:
                return f"Error: Directory not found: {path}"
        items = [f"[dir] {x.name}" if x.is_dir() else x.name for x in sorted(p.iterdir())]
        return "\n".join(items[:50])

    if name == "read_file":
        filepath = args.get("filepath", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if not filepath:
            return "Error: filepath required"
        p = Path(filepath)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists():
            return f"Error: File not found: {filepath}"
        text = p.read_text(encoding="utf-8", errors="replace")
        if start_line is not None and end_line is not None:
            lines = text.splitlines()
            start = max(0, int(start_line) - 1)
            end = min(len(lines), int(end_line))
            return "\n".join(lines[start:end])
        return text[:20000]

    if name == "write_file":
        filepath = args.get("filepath", "")
        content = args.get("content", "")
        if not filepath:
            return "Error: filepath required"
        try:
            p = resolve_path(filepath, project_root)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"File written: {p.relative_to(project_root)}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"

    if name == "modify_file":
        filepath = args.get("filepath", "")
        new_content = args.get("new_content")
        search_text = args.get("search_text")
        replace_text = args.get("replace_text")
        if not filepath:
            return "Error: filepath required"
        try:
            p = resolve_path(filepath, project_root)
            if not p.exists():
                return f"Error: File not found: {filepath}"
            text = p.read_text(encoding="utf-8")
            if new_content is not None:
                text = new_content
            elif search_text is not None and replace_text is not None:
                text = text.replace(search_text, replace_text)
            else:
                return "Error: Provide new_content (full rewrite) or both search_text and replace_text"
            p.write_text(text, encoding="utf-8")
            return f"File modified: {p.relative_to(project_root)}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error modifying file: {e}"

    if name == "delete_file":
        filepath = args.get("filepath", "")
        if not filepath:
            return "Error: filepath required"
        try:
            p = resolve_path(filepath, project_root)
            if not p.exists():
                return f"Error: File not found: {filepath}"
            if not p.is_file():
                return "Error: Not a file (cannot delete directory)"
            p.unlink()
            return f"File deleted: {p.relative_to(project_root)}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error deleting file: {e}"

    if name == "rename_file":
        filepath = args.get("filepath", "")
        new_filepath = args.get("new_filepath", "")
        if not filepath or not new_filepath:
            return "Error: filepath and new_filepath required"
        try:
            p = resolve_path(filepath, project_root)
            q = resolve_path(new_filepath, project_root)
            if not p.exists():
                return f"Error: File not found: {filepath}"
            if not p.is_file():
                return "Error: Not a file"
            p.rename(q)
            return f"Renamed to {q.relative_to(project_root)}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error renaming: {e}"

    if name == "find_file":
        pattern = args.get("pattern", "")
        directory = args.get("directory", ".")
        if not pattern:
            return "Error: pattern required"
        p = Path(directory)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists() or not p.is_dir():
            # Try examples/ prefix (e.g. LES/64 -> examples/LES/64)
            alt = project_root / "examples" / directory.lstrip("/")
            if alt.exists() and alt.is_dir():
                p = alt
            else:
                p = project_root
        found = [str(f.relative_to(project_root)) for f in p.rglob(pattern) if f.is_file()]
        return "\n".join(sorted(found)[:100])

    if name == "search_codebase":
        query = args.get("query", "")
        file_pattern = args.get("file_pattern", "*")
        if not query:
            return "Error: query required"
        skip = {"venv", "myenv", "__pycache__", ".git", "node_modules", ".venv", "local_tools"}
        out = []
        for f in project_root.rglob(file_pattern):
            if not f.is_file() or any(p in f.parts for p in skip):
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if query in line:
                        out.append(f"{f.relative_to(project_root)}:{i}: {line.strip()}")
                        if len(out) >= 50:
                            break
            except Exception:
                continue
            if len(out) >= 50:
                break
        return "\n".join(out) if out else "No matches"

    if name == "extract_section":
        filepath = args.get("filepath", "")
        query = args.get("query", "")
        context_lines = int(args.get("context_lines", 30))
        if not filepath or not query:
            return "Error: filepath and query required"
        p = Path(filepath)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists() or not p.is_file():
            return f"Error: File not found: {filepath}"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if query in line:
                end = min(len(lines), i + context_lines)
                section = "\n".join(f"{j+1}: {lines[j]}" for j in range(i, end))
                return section
        return "No match"

    return f"Error: Unknown core tool '{name}'"
