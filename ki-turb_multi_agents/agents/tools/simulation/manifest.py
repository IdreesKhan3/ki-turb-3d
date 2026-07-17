"""Tool for reading a saved dataset manifest and helpers to summarize one."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from schemas import DatasetManifest

from . import _store

MANIFEST_TOOL_NAMES = frozenset({"read_dataset_manifest", "list_simulation_jobs"})


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "list_simulation_jobs",
            "description": (
                "List saved OpenLB/KI-TURB simulation jobs under simulations/. "
                "Use for questions like how many jobs are saved, which job ids exist, "
                "or which jobs have a manifest. Do NOT use load_dataset_manifest for counting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "with_manifest_only": {
                        "type": "boolean",
                        "description": "If true, only jobs that already have manifest.json",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max job ids to list in detail (default 50)",
                    },
                },
            },
        },
        {
            "name": "read_dataset_manifest",
            "description": (
                "Read the dataset manifest saved for a simulation job and summarize the "
                "files it produced (kinds, formats, time steps)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Simulation job id"}},
                "required": ["job_id"],
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    if name == "list_simulation_jobs":
        return _list_simulation_jobs(args or {}, project_root)
    if name != "read_dataset_manifest":
        return f"Error: Unknown manifest tool '{name}'"

    job_id = str(args.get("job_id", "")).strip()
    if not job_id:
        return "Error: job_id required"

    manifest = _store.load_manifest(project_root, job_id)
    if manifest is None:
        return f"Error: no manifest found for job '{job_id}'. Fetch outputs first."

    return (
        f"Manifest {manifest.manifest_id} for job {job_id} "
        f"(backend: {manifest.backend}, source: {manifest.source_simulation}).\n"
        f"{summarize_manifest(manifest)}"
    )


def _list_simulation_jobs(args: Dict[str, Any], project_root: Path) -> str:
    ids = _store.list_job_ids(project_root)
    with_manifest_only = bool(args.get("with_manifest_only"))
    try:
        limit = int(args.get("limit") or 50)
    except Exception:
        limit = 50
    limit = max(1, min(limit, 200))

    rows: List[str] = []
    with_manifest = 0
    for job_id in ids:
        has_manifest = (_store.job_dir(project_root, job_id) / _store.MANIFEST_FILENAME).is_file()
        if has_manifest:
            with_manifest += 1
        if with_manifest_only and not has_manifest:
            continue
        if len(rows) < limit:
            rows.append(f"- {job_id} (manifest={'yes' if has_manifest else 'no'})")

    listed = ids if not with_manifest_only else [
        j for j in ids
        if (_store.job_dir(project_root, j) / _store.MANIFEST_FILENAME).is_file()
    ]
    total = len(listed)
    lines = [
        f"Saved simulation jobs under simulations/: {total}",
        f"Jobs with manifest.json: {with_manifest} (of {len(ids)} total job dirs)",
        "Newest first:",
        *(rows if rows else ["- (none)"]),
    ]
    if total > limit:
        lines.append(f"… and {total - limit} more (raise limit to see them)")
    return "\n".join(lines)


def summarize_manifest(manifest: DatasetManifest) -> str:
    if not manifest.files:
        return "No output files recorded."
    kinds = Counter(f.kind for f in manifest.files)
    kind_summary = ", ".join(f"{kind}: {count}" for kind, count in sorted(kinds.items()))
    lines = [
        f"files: {len(manifest.files)} ({kind_summary})",
        f"base_dir: {manifest.base_dir}",
    ]
    if manifest.time_steps:
        lines.append(f"time_steps: {len(manifest.time_steps)}")
    return "\n".join(lines)
