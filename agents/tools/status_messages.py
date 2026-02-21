"""
Human-friendly status messages for agent tool calls and delegation.
Makes agents talk naturally: "Listing directory: test...", "Creating test/test.py...", etc.
"""

import re
from typing import Any, Dict


def friendly_delegation(agent_name: str, task: str) -> str:
    """Convert raw delegation task to a natural-language message."""
    t = (task or "").strip()[:120]
    name = agent_name.replace("_", " ").title()
    if any(w in t.lower() for w in ("create", "write", "add", "make", "put")):
        return f"Asking {name} to {t}."
    if any(w in t.lower() for w in ("remove", "delete", "rm ")):
        return f"Asking {name} to {t}."
    if any(w in t.lower() for w in ("run_shell_command", "run ", "execute ", "shell")):
        return f"Asking {name} to {t}."
    if "list_directory" in t:
        m = re.search(r"path\s*=\s*['\"]?([^\s'\"]+)['\"]?", t, re.I)
        path = m.group(1) if m else "directory"
        return f"Asking {name} to list {path}."
    if "find_file" in t:
        m = re.search(r"(?:query|pattern)\s*=\s*['\"]?([^\s'\"]+)['\"]?", t, re.I)
        q = m.group(1) if m else "files"
        return f"Asking {name} to search for {q}."
    if "read_file" in t:
        m = re.search(r"(?:filepath|path)\s*=\s*['\"]?([^\s'\"]+)['\"]?", t, re.I)
        p = m.group(1) if m else "file"
        return f"Asking {name} to read {p}."
    return f"Asking {name}: {t}."


def get_tool_status_before(tool_name: str, args: Dict[str, Any]) -> str:
    """Return a friendly message when the agent is about to call a tool. Includes tool name."""
    tool = (tool_name or "").strip().lower()
    path = args.get("filepath") or args.get("path") or args.get("dirpath") or ""
    query = (args.get("query") or args.get("pattern") or "")[:50]
    cmd = (args.get("command") or args.get("cmd") or "")[:60]

    def _msg(friendly: str) -> str:
        return f"{friendly} [{tool_name}]"

    if tool == "list_directory":
        return _msg(f"Listing directory: {path or '.'}...")
    if tool == "read_file":
        return _msg(f"Reading {path}...")
    if tool == "find_file":
        return _msg(f"Searching for files matching '{query or 'pattern'}'...")
    if tool == "write_file":
        return _msg(f"Creating {path}...")
    if tool == "search_codebase":
        return _msg(f"Searching codebase for '{query}'..." if query else "Searching codebase...")
    if tool == "regex_search":
        return _msg(f"Regex search for pattern '{query}'..." if query else "Regex search...")
    if tool == "semantic_search":
        return _msg(f"Semantic search for '{query}'..." if query else "Semantic search...")
    if tool == "modify_file":
        return _msg(f"Modifying {path}...")
    if tool == "delete_file":
        return _msg(f"Deleting {path}...")
    if tool == "rename_file":
        return _msg(f"Renaming to {args.get('new_filepath', '?')}...")
    if tool == "run_shell_command":
        return _msg(f"Running: {cmd}..." if cmd else "Running command...")
    if tool == "search_research_papers":
        return _msg(f"Searching papers for '{query}'..." if query else "Searching research papers...")
    if tool == "web_search":
        return _msg(f"Searching web for '{query}'..." if query else "Searching web...")
    if tool == "download_file":
        return _msg("Downloading file...")
    if tool == "browse_web":
        return _msg("Loading web page...")
    if tool == "compute_spectra":
        return _msg("Computing energy spectra...")
    if tool == "plot_spectrum":
        return _msg("Creating spectrum plot...")
    if tool == "plot_lumley_triangle":
        return _msg("Creating Lumley triangle...")
    if tool == "plot_diagonal_bii":
        return _msg("Creating diagonal b_ii plot...")
    if tool == "plot_cross_correlations":
        return _msg("Creating cross-correlations plot...")
    if tool == "plot_deviations":
        return _msg("Creating deviations plot...")
    if tool == "plot_convergence":
        return _msg("Creating convergence plot...")
    if tool == "get_real_isotropy_summary":
        return _msg("Creating real isotropy summary table...")
    if tool == "get_real_isotropy_theory":
        return _msg("Fetching Theory & Equations...")
    if tool == "get_overview_theory":
        return _msg("Fetching Overview Physics Validation Equations...")
    if tool == "get_spectral_isotropy_theory":
        return _msg("Fetching Spectral Isotropy Theory & Equations...")
    if tool == "export_figure":
        return _msg("Exporting figure...")
    if tool == "execute_code":
        return _msg("Executing code...")
    if tool == "run_in_terminal":
        return _msg(f"Running in terminal: {cmd}..." if cmd else "Running in terminal...")
    if tool == "git_operation":
        return _msg(f"Git: {args.get('operation', '')}...")

    return f"Running {tool_name}..."


def get_tool_status_after(tool_name: str, args: Dict[str, Any], result: Any) -> str:
    """Return a friendly message summarizing the tool result."""
    res_str = str(result)[:120] if result else ""
    path = args.get("filepath") or args.get("path") or ""

    if isinstance(result, dict):
        if result.get("status") == "pending_confirmation":
            return "Waiting for your approval."
        if result.get("success") is False or result.get("ok") is False:
            return f"⚠ {result.get('message', res_str)[:80]}"
        if "artifact_type" in result:
            return "✓ Done."
        msg = result.get("message", "")
        if msg and len(msg) < 100:
            return msg

    if "Error" in res_str or "error" in res_str.lower():
        return f"⚠ {res_str[:80]}"
    return "✓ Done."
