"""Compact per-chat memory so follow-ups see prior sim/plot/tool evidence."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _clip(text: Any, n: int = 400) -> str:
    value = str(text or "").strip()
    if len(value) <= n:
        return value
    return value[: n - 1] + "…"


_PATH_RE = re.compile(
    r"(?i)\b(?:File (?:written|deleted|modified|renamed(?: from)?|not found)|"
    r"imported)\s*:?\s*([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)"
)
_PATH_RE_ALT = re.compile(
    r"(?i)(?:examples|agents|pages|tests|knowledge|simulations)/[A-Za-z0-9_./\\-]+"
)


def empty_turn_memory() -> Dict[str, Any]:
    return {
        "job_id": "",
        "manifest_path": "",
        "last_action": "",
        "last_tools": [],
        "last_roles": [],
        "last_paths": [],
        "compile_mentioned": False,
        "run_mentioned": False,
        "plot_tools": [],
        "artifact_captions": [],
        "summary": "",
    }


def _extract_paths(*chunks: Any) -> List[str]:
    found: List[str] = []
    for chunk in chunks:
        text = str(chunk or "")
        for match in _PATH_RE.findall(text):
            found.append(match.replace("\\", "/"))
        for match in _PATH_RE_ALT.findall(text):
            found.append(match.replace("\\", "/"))
    # Dedupe preserving order
    seen = set()
    out: List[str] = []
    for path in found:
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def update_turn_memory(
    previous: Optional[Dict[str, Any]],
    *,
    user_request: str,
    plan: Optional[Dict[str, Any]] = None,
    task_results: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    final_text: str = "",
    status: str = "",
) -> Dict[str, Any]:
    """Merge the latest workflow outcome into durable chat memory."""
    mem = dict(previous or empty_turn_memory())
    ctx = session_context or {}
    job = str(
        ctx.get("simulation_job_id")
        or ctx.get("sim_workflow_job")
        or mem.get("job_id")
        or ""
    ).strip()
    manifest = str(
        ctx.get("manifest_path")
        or ctx.get("dataset_manifest_path")
        or mem.get("manifest_path")
        or ""
    ).strip()

    tools: List[str] = []
    roles: List[str] = []
    for step in (plan or {}).get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("tool"):
            tools.append(str(step["tool"]))
        if step.get("role"):
            roles.append(str(step["role"]))
    for item in task_results or []:
        if isinstance(item, dict) and item.get("role"):
            roles.append(str(item["role"]))
        for out in (item.get("tool_outputs") if isinstance(item, dict) else None) or []:
            if isinstance(out, dict) and out.get("tool"):
                tools.append(str(out["tool"]))

    # Deduplicate preserving order
    def _uniq(values: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return out

    tools = _uniq(tools)
    roles = _uniq(roles)
    plot_tools = [t for t in tools if t.startswith("plot_")]
    captions = list(mem.get("artifact_captions") or [])
    for art in artifacts or []:
        if not isinstance(art, dict):
            continue
        caption = art.get("artifact_title") or art.get("message") or art.get("caption")
        if caption:
            captions.append(_clip(caption, 120))
    captions = captions[-12:]

    path_chunks: List[Any] = [user_request, final_text]
    for item in task_results or []:
        if isinstance(item, dict):
            path_chunks.append(item.get("text"))
            for out in item.get("tool_outputs") or []:
                path_chunks.append(out)
    for step in (plan or {}).get("steps") or []:
        if isinstance(step, dict):
            path_chunks.append(step.get("instruction"))
            args = step.get("tool_args") if isinstance(step.get("tool_args"), dict) else {}
            for key in ("filepath", "path", "module"):
                if args.get(key):
                    path_chunks.append(args[key])
    last_paths = _uniq(list(mem.get("last_paths") or []) + _extract_paths(*path_chunks))
    # Keep the active job tree as a path root for follow-ups (any basename under it).
    if job:
        last_paths = _uniq(last_paths + [f"simulations/{job}"])
    last_paths = last_paths[-20:]

    joined = " ".join(tools).lower() + " " + (final_text or "").lower()
    compile_mentioned = bool(
        mem.get("compile_mentioned")
        or "compile_simulation" in tools
        or re.search(r"\bcompil(?:e|ed|ation)\b", joined)
    )
    run_mentioned = bool(
        mem.get("run_mentioned")
        or "start_simulation" in tools
        or "supervise_simulation" in tools
        or re.search(r"\b(?:started|running|completed)\b.*\b(?:simulation|job)\b", joined)
    )

    mem.update(
        {
            "job_id": job,
            "manifest_path": manifest,
            "last_action": _clip(user_request, 240),
            "last_tools": tools[-20:],
            "last_roles": roles[-12:],
            "last_paths": last_paths,
            "compile_mentioned": compile_mentioned,
            "run_mentioned": run_mentioned,
            "plot_tools": _uniq(list(mem.get("plot_tools") or []) + plot_tools)[-12:],
            "artifact_captions": captions,
            "summary": _clip(
                f"status={status or 'unknown'}; job={job or 'none'}; "
                f"manifest={manifest or 'none'}; tools={', '.join(tools[-8:]) or 'none'}; "
                f"paths={', '.join(last_paths[-6:]) or 'none'}; "
                f"answer={_clip(final_text, 220)}",
                800,
            ),
        }
    )
    return mem


def format_turn_memory(memory: Optional[Dict[str, Any]]) -> str:
    if not memory:
        return ""
    lines = [
        "Prior chat evidence (use this before inventing facts):",
        f"- last_user_request: {memory.get('last_action') or '(none)'}",
        f"- job_id: {memory.get('job_id') or '(none)'}",
        f"- manifest_path: {memory.get('manifest_path') or '(none)'}",
        f"- last_paths: {', '.join(memory.get('last_paths') or []) or '(none)'}",
        f"- compile_seen_in_session: {bool(memory.get('compile_mentioned'))}",
        f"- run_seen_in_session: {bool(memory.get('run_mentioned'))}",
        f"- last_tools: {', '.join(memory.get('last_tools') or []) or '(none)'}",
        f"- plot_tools: {', '.join(memory.get('plot_tools') or []) or '(none)'}",
        f"- artifacts: {', '.join(memory.get('artifact_captions') or []) or '(none)'}",
        f"- summary: {memory.get('summary') or '(none)'}",
        "Resolve relative/bare paths via job_id and last_paths before inventing locations.",
        "Answer THIS user request with THIS turn's tool evidence. "
        "Do not repeat a previous answer unless it still answers the new request.",
    ]
    return "\n".join(lines)


__all__ = [
    "empty_turn_memory",
    "update_turn_memory",
    "format_turn_memory",
]
