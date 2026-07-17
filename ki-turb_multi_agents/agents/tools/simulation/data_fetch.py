"""Tool for collecting a completed simulation's outputs into a dataset manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import integrations
from integrations.base import BackendError
from schemas.simulation_job import JobStatus

from . import _store
from .manifest import summarize_manifest
from ._status import format_job_error, simulation_job_ready_for_fetch

DATA_FETCH_TOOL_NAMES = frozenset({"fetch_simulation_outputs"})


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "fetch_simulation_outputs",
            "description": (
                "Scan a simulation job's output directory, record every result file in a "
                "dataset manifest, and save it. Returns a summary of the files found."
            ),
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Simulation job id"}},
                "required": ["job_id"],
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    if name != "fetch_simulation_outputs":
        return f"Error: Unknown data-fetch tool '{name}'"

    job_id = str(args.get("job_id", "")).strip()
    if not job_id:
        return "Error: job_id required"

    job = _store.load_job(project_root, job_id)
    if job is None:
        return f"Error: no job found with id '{job_id}'."

    if not simulation_job_ready_for_fetch(job):
        hint = ""
        if job.status == JobStatus.RUNNING:
            hint = " Wait for supervise_simulation to finish before fetching outputs."
        return format_job_error(job, action="fetch outputs") + hint

    backend = integrations.get_backend(job.backend)
    try:
        manifest = backend.fetch_outputs(job)
    except BackendError as exc:
        return f"Error: {exc}"

    manifest_path = _store.save_manifest(project_root, job_id, manifest)
    job.metadata["manifest_path"] = str(manifest_path)
    if not job.status.is_terminal:
        job.mark(JobStatus.FETCHED, message="outputs collected into manifest")
    _store.save_job(project_root, job)

    return (
        f"Fetched outputs for job {job_id}.\n"
        f"manifest: {manifest_path}\n"
        f"{summarize_manifest(manifest)}"
    )
