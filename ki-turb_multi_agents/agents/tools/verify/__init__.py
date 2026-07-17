"""Scoped verification tools for the engineer role."""
from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

VERIFY_TOOL_NAMES = frozenset({"run_pytest", "run_import_check", "run_verify_command"})

_SAFE_VERIFY = re.compile(
    r"^(?:pytest|python\s+-m\s+pytest|python\s+-c\s+|python\s+-m\s+compileall)\b",
    re.I,
)
_FORBIDDEN = re.compile(r"[;&|`$]|&&|\|\||>|>>|<|\nrm\b|\bsudo\b", re.I)


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "run_pytest",
            "description": (
                "Run a scoped pytest invocation under the project root. "
                "Pass test paths relative to the project (e.g. tests/evals/test_engineering_smoke.py)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Test file or directory paths relative to project root.",
                    },
                    "extra_args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional extra pytest args (e.g. -q, -k name).",
                    },
                },
                "required": ["paths"],
            },
        },
        {
            "name": "run_import_check",
            "description": "Import a Python module (dotted name or file path) to smoke-check it loads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "Dotted module name (preferred) or project-relative .py path.",
                    },
                },
                "required": ["module"],
            },
        },
        {
            "name": "run_verify_command",
            "description": (
                "Run an allowlisted verify command (pytest / python -m pytest / "
                "python -c / python -m compileall). Mutating shell is not allowed here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full allowlisted verify command string.",
                    },
                },
                "required": ["command"],
            },
        },
    ]


def _run(cmd: List[str], cwd: Path, timeout: int = 180) -> str:
    env = os.environ.copy()
    # Prevent matplotlib GUI backends from blocking headless agent/Streamlit runs.
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "Error: verify command timed out."
    except Exception as exc:
        return f"Error: verify failed: {type(exc).__name__}: {exc}"
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    status = "ok" if proc.returncode == 0 else "failed"
    return f"status: {status}\nexit_code: {proc.returncode}\n{out.strip()}"


def _resolve_under_root(project_root: Path, rel: str) -> Path | None:
    raw = str(rel or "").strip().lstrip("./")
    if not raw or ".." in Path(raw).parts:
        return None
    path = (project_root / raw).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError:
        return None
    return path


def _run_pytest(args: Dict[str, Any], project_root: Path) -> str:
    paths = args.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        return "Error: run_pytest requires at least one path."
    resolved: List[str] = []
    for item in paths:
        path = _resolve_under_root(project_root, str(item))
        if path is None:
            return f"Error: path not allowed: {item}"
        resolved.append(str(path.relative_to(project_root.resolve())))
    extra = args.get("extra_args") or []
    if isinstance(extra, str):
        extra = [extra]
    cmd = [sys.executable, "-m", "pytest", *resolved, *[str(x) for x in extra]]
    return _run(cmd, project_root)


def _run_import_check(args: Dict[str, Any], project_root: Path) -> str:
    module = str(args.get("module") or "").strip()
    if not module:
        return "Error: module is required."
    if module.endswith(".py") or "/" in module or "\\" in module:
        path = _resolve_under_root(project_root, module)
        if path is None or not path.is_file():
            return f"Error: file not found or not allowed: {module}"
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            return f"status: failed\nError: syntax check failed: {exc}"
        return f"status: ok\nParsed {path.relative_to(project_root)}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", module):
        return "Error: invalid module name."
    code = (
        "import importlib, sys; "
        f"sys.path.insert(0, {str(project_root)!r}); "
        f"mod = importlib.import_module({module!r}); "
        "print('imported', mod.__name__)"
    )
    return _run([sys.executable, "-c", code], project_root)


_INSPECT_CMD = re.compile(
    r"^(?:cat|head|tail|wc|less|more|nl)\b\s+(.+)$",
    re.I,
)


def _run_inspect_as_read(command: str, project_root: Path) -> str | None:
    """Handle cat/head/… verify leftovers by reading the file (no shell)."""
    match = _INSPECT_CMD.match((command or "").strip())
    if not match:
        return None
    target = match.group(1).strip().strip("'\"")
    # Drop simple flags like -n 20 for head/tail.
    parts = [p for p in target.split() if not p.startswith("-")]
    if not parts:
        return f"Error: no path in inspect command: {command}"
    path = _resolve_under_root(project_root, parts[0])
    if path is None:
        return f"Error: path not allowed: {parts[0]}"
    if not path.exists():
        return f"status: failed\nexists=False size=0\nError: file not found: {parts[0]}"
    if not path.is_file():
        return f"status: ok\nexists=True size=0\n(not a file)"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"status: failed\nError: could not read {parts[0]}: {exc}"
    preview = text[0:4000]
    return (
        f"status: ok\nexists=True size={path.stat().st_size}\n"
        f"{preview}"
    )


def _run_python_c_argv(command: str, project_root: Path) -> Optional[str]:
    """
    Run `python -c '…'` via argv (no bash), so `;` / quotes inside the snippet
    are not mistaken for shell metacharacters.
    """
    if not re.match(r"^python\s+-c\s+", command, re.I):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"Error: could not parse python -c command: {exc}"
    if len(tokens) < 3 or tokens[0].lower() != "python" or tokens[1] != "-c":
        return "Error: invalid python -c command form."
    code = tokens[2]
    # Remaining tokens are unusual for -c; reject to stay strict.
    if len(tokens) > 3:
        return "Error: python -c verify commands must be a single -c snippet."
    return _run([sys.executable, "-c", code], project_root)


def _run_verify_command(args: Dict[str, Any], project_root: Path) -> str:
    command = str(args.get("command") or "").strip().strip("\x00")
    if not command:
        return "Error: command is required."
    inspect_out = _run_inspect_as_read(command, project_root)
    if inspect_out is not None:
        return inspect_out
    # python -c: execute without bash so script semicolons are allowed.
    py_c = _run_python_c_argv(command, project_root)
    if py_c is not None:
        return py_c
    if _FORBIDDEN.search(command):
        return "Error: verify command contains forbidden shell metacharacters."
    if not _SAFE_VERIFY.search(command):
        return (
            "Error: command not allowlisted. Use pytest, python -m pytest, "
            "python -c, or python -m compileall. "
            "(cat/head/… are handled as file reads; prefer read_file or compileall.)"
        )
    # Prefer argv form for pytest paths when possible.
    if re.match(r"^(?:pytest|python\s+-m\s+pytest)\b", command, re.I):
        tokens = command.split()
        # Drop leading pytest / python -m pytest
        if tokens[0].lower() == "pytest":
            rest = tokens[1:]
        else:
            rest = tokens[3:]  # python -m pytest ...
        paths = [t for t in rest if not t.startswith("-")]
        extras = [t for t in rest if t.startswith("-")]
        if paths:
            return _run_pytest({"paths": paths, "extra_args": extras}, project_root)
    if re.match(r"^python\s+-m\s+compileall\b", command, re.I):
        return _run(["bash", "-lc", command], project_root)
    return _run(["bash", "-lc", command], project_root)


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    root = Path(project_root).resolve()
    if name == "run_pytest":
        return _run_pytest(args or {}, root)
    if name == "run_import_check":
        return _run_import_check(args or {}, root)
    if name == "run_verify_command":
        return _run_verify_command(args or {}, root)
    return f"Error: Unknown verify tool '{name}'"


__all__ = ["VERIFY_TOOL_NAMES", "get_tool_definitions", "execute_tool"]
