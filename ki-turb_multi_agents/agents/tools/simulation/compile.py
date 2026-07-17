"""Tool for compiling a prepared solver case when the backend requires a build."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import integrations
from integrations.base import BackendError

from . import _store
from ._activity import emit_simulation_progress
from ._status import format_job_error, simulation_job_failed

COMPILE_TOOL_NAMES = frozenset({"compile_simulation"})


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "compile_simulation",
            "description": "Compile or build the prepared CFD solver case if the backend "
                           "requires compilation (e.g. OpenLB, Palabos). No-op otherwise.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        }
    ]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Mapping[str, Any]] = None,
) -> str:
    job_id = str(args.get("job_id", "")).strip()
    if not job_id:
        return "Error: job_id required"

    job = _store.load_job(project_root, job_id)
    if job is None:
        return f"Error: no job found with id '{job_id}'."

    emit_simulation_progress(
        session_context,
        phase="compile",
        job_id=job_id,
        progress_pct=5.0,
        message=f"Compiling {job.backend} solver",
        status="running",
    )

    backend = integrations.get_backend(job.backend)
    try:
        job = backend.compile_case(job)
    except BackendError as exc:
        emit_simulation_progress(
            session_context,
            phase="compile",
            job_id=job_id,
            progress_pct=100.0,
            message=str(exc),
            status="error",
        )
        return f"Error: {exc}"

    _store.save_job(project_root, job)
    if simulation_job_failed(job):
        emit_simulation_progress(
            session_context,
            phase="compile",
            job_id=job_id,
            progress_pct=100.0,
            message=job.message or "Compile failed",
            status="error",
        )
        return format_job_error(job, action="compile case")

    cache_hit = bool((job.metadata or {}).get("compile_result", {}).get("cache_hit"))
    emit_simulation_progress(
        session_context,
        phase="compile",
        job_id=job_id,
        progress_pct=100.0,
        message="Used cached build" if cache_hit else "Build finished",
        status="success",
    )
    return (
        f"Compiled simulation.\n"
        f"job_id: {job.job_id}\n"
        f"backend: {job.backend}\n"
        f"status: {job.status.value}\n"
        f"message: {job.message}"
    )
