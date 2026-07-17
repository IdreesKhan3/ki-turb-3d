"""Load capability knowledge packs for engineering discovery/planning."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    yaml = None


def _default_knowledge_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / "knowledge" / "capabilities"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _load_index(knowledge_root: Path) -> Dict[str, Any]:
    index_path = knowledge_root / "_index.yaml"
    if not index_path.is_file():
        return {"capabilities": {}}
    text = _read_text(index_path)
    if yaml is not None:
        try:
            data = yaml.safe_load(text) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    # Minimal fallback parser: capability id lines under capabilities:
    caps: Dict[str, Any] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^  ([a-z0-9_]+):\s*$", line)
        if m:
            current = m.group(1)
            caps[current] = {"triggers": []}
            continue
        if current and "triggers:" in line:
            continue
        tm = re.match(r'^\s+-\s+"?([^"]+)"?\s*$', line)
        if current and tm and "path:" not in line:
            caps[current].setdefault("triggers", []).append(tm.group(1).strip().strip('"'))
        pm = re.match(r'^\s+path:\s+"?([^"]+)"?\s*$', line)
        if current and pm:
            caps[current]["path"] = pm.group(1).strip().strip('"')
    return {"capabilities": caps}


def match_capabilities(request: str, project_root: Path | str) -> List[str]:
    """Return capability ids whose trigger hints appear in the request."""
    root = _default_knowledge_root(Path(project_root))
    index = _load_index(root)
    caps = index.get("capabilities") or {}
    lower = (request or "").lower()
    matched: List[str] = []
    for cap_id, meta in caps.items():
        if not isinstance(meta, dict):
            continue
        triggers = meta.get("triggers") or []
        if any(str(t).lower() in lower for t in triggers):
            matched.append(str(cap_id))
    # User-script path edits are not platform capability work (avoid "plot"/"page" noise).
    if re.search(r"\bexamples/[\w./-]+\.py\b", lower):
        return []

    if not matched:
        # Lightweight heuristics when index is sparse — prefer product phrasing.
        heuristics = [
            ("app_pages", ("streamlit page", "page_schema", "autonomous lab", "new analysis page")),
            ("plotting", ("plot tool", "plotting tool", "register plot", "visualizer tool")),
            ("solvers", ("solver adapter", "cfd backend", "palabos", "openfoam", "ansys backend")),
            ("viz_external", ("vtk", "paraview", "external viz")),
            ("hpc", ("hpc", "slurm", "remote runner", "gpu cluster")),
        ]
        for cap_id, words in heuristics:
            if any(w in lower for w in words):
                matched.append(cap_id)
    # Preserve order, unique
    seen = set()
    ordered = []
    for item in matched:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def load_pack_text(capability: str, project_root: Path | str) -> str:
    root = _default_knowledge_root(Path(project_root))
    index = _load_index(root)
    meta = (index.get("capabilities") or {}).get(capability) or {}
    rel = meta.get("path") or capability
    pack_dir = root / str(rel)
    chunks: List[str] = []
    for name in ("layout.md", "change_patterns.md", "failure_playbook.md"):
        text = _read_text(pack_dir / name)
        if text.strip():
            chunks.append(f"# {capability}/{name}\n{text.strip()}")
    common = _read_text(root / "common" / "verify_recipes.md")
    if common.strip():
        chunks.append("# common/verify_recipes.md\n" + common.strip())
    return "\n\n".join(chunks)


def load_capability_context(
    request: str,
    project_root: Path | str,
    *,
    lessons_text: str = "",
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """Match capabilities and return concatenated pack context for prompts."""
    caps = match_capabilities(request, project_root)
    parts = [load_pack_text(cap, project_root) for cap in caps]
    body = "\n\n".join(p for p in parts if p.strip())
    if lessons_text.strip():
        body = (body + "\n\n# retrieved_lessons\n" + lessons_text.strip()).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    return {
        "capabilities": caps,
        "primary_capability": caps[0] if caps else "",
        "context": body,
    }


__all__ = ["match_capabilities", "load_pack_text", "load_capability_context"]
