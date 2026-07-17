"""Tools for controlling a simulation job: start, status, cancel.

Each tool loads the durable job record by ``job_id``, delegates to the job's
backend, and persists the updated record. Cross-call status polling relies on
process liveness, so a job launched in a prior session is reported as running
until its process exits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import integrations
from integrations.base import BackendError, BackendNotConfigured
from schemas.simulation_job import JobStatus

from . import _store
from ._activity import emit_simulation_progress, max_steps_from_job, progress_percent_from_job
from ._status import format_job_error, simulation_job_failed

RUN_CONTROL_TOOL_NAMES = frozenset(
    {"start_simulation", "check_simulation_status", "cancel_simulation", "supervise_simulation"}
)


def get_tool_definitions() -> List[Dict[str, Any]]:
    job_arg = {
        "type": "object",
        "properties": {"job_id": {"type": "string", "description": "Simulation job id"}},
        "required": ["job_id"],
    }
    return [
        {
            "name": "start_simulation",
            "description": "Launch a prepared simulation job by job_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "executable": {
                        "type": "string",
                        "description": "Solver executable path; overrides the backend env var.",
                    },
                },
                "required": ["job_id"],
            },
        },
        {
            "name": "check_simulation_status",
            "description": "Return the current status of a simulation job.",
            "parameters": job_arg,
        },
        {
            "name": "cancel_simulation",
            "description": "Cancel a running simulation job.",
            "parameters": job_arg,
        },
        {
            "name": "supervise_simulation",
            "description": (
                "Poll a running simulation job until it reaches a terminal status "
                "or the timeout elapses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum wait time in seconds (default 3600).",
                    },
                    "poll_seconds": {
                        "type": "number",
                        "description": "Polling interval in seconds (default 5).",
                    },
                },
                "required": ["job_id"],
            },
        },
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
        return f"Error: no job found with id '{job_id}'. Build a case first."

    backend = integrations.get_backend(job.backend, executable=args.get("executable"))

    if name == "start_simulation":
        emit_simulation_progress(
            session_context,
            phase="run",
            job_id=job_id,
            progress_pct=0.0,
            message="Launching solver process",
            status="running",
        )
        try:
            job = backend.run_case(job)
        except BackendNotConfigured as exc:
            emit_simulation_progress(
                session_context,
                phase="run",
                job_id=job_id,
                progress_pct=0.0,
                message=str(exc),
                status="error",
            )
            return f"Error: {exc}"
        except BackendError as exc:
            emit_simulation_progress(
                session_context,
                phase="run",
                job_id=job_id,
                progress_pct=0.0,
                message=str(exc),
                status="error",
            )
            return f"Error: {exc}"
        _store.save_job(project_root, job)
        if simulation_job_failed(job):
            emit_simulation_progress(
                session_context,
                phase="run",
                job_id=job_id,
                progress_pct=0.0,
                message=job.message or "Start failed",
                status="error",
            )
            return format_job_error(job, action="start simulation")
        emit_simulation_progress(
            session_context,
            phase="run",
            job_id=job_id,
            progress_pct=0.0,
            message="Solver running",
            status="running",
        )
        return _summary(job, header="Started simulation")

    if name == "check_simulation_status":
        job = backend.check_status(job)
        _store.save_job(project_root, job)
        _emit_job_progress(session_context, job)
        return _summary(job, header="Simulation status")

    if name == "cancel_simulation":
        job = backend.cancel_run(job)
        _store.save_job(project_root, job)
        emit_simulation_progress(
            session_context,
            phase="simulation",
            job_id=job_id,
            progress_pct=progress_percent_from_job(job)[0],
            message="Cancelled",
            status="warning",
        )
        return _summary(job, header="Cancelled simulation")

    if name == "supervise_simulation":
        timeout = float(args.get("timeout_seconds") or 3600.0)
        poll = max(0.5, float(args.get("poll_seconds") or 5.0))
        deadline = time.monotonic() + timeout
        max_steps = max_steps_from_job(job)
        emit_simulation_progress(
            session_context,
            phase="simulation",
            job_id=job_id,
            progress_pct=0.0,
            message="Monitoring simulation",
            status="running",
            step=0,
            max_steps=max_steps,
        )
        while time.monotonic() < deadline:
            job = _store.load_job(project_root, job_id)
            if job is None:
                return f"Error: no job found with id '{job_id}'."
            backend = integrations.get_backend(job.backend, executable=args.get("executable"))
            job = backend.check_status(job)
            _store.save_job(project_root, job)
            _emit_job_progress(session_context, job, max_steps=max_steps)
            if job.status.is_terminal:
                terminal_status = "success" if job.status == JobStatus.COMPLETED else "error"
                pct, step = progress_percent_from_job(job)
                emit_simulation_progress(
                    session_context,
                    phase="simulation",
                    job_id=job_id,
                    progress_pct=100.0 if job.status == JobStatus.COMPLETED else pct,
                    message=job.message or job.status.value,
                    status=terminal_status,
                    step=step,
                    max_steps=max_steps,
                )
                if job.status != JobStatus.COMPLETED:
                    return format_job_error(job, action="supervise simulation")
                return _summary(job, header="Simulation finished")
            time.sleep(poll)
        job = _store.load_job(project_root, job_id)
        status = job.status.value if job is not None else "unknown"
        emit_simulation_progress(
            session_context,
            phase="simulation",
            job_id=job_id,
            progress_pct=progress_percent_from_job(job)[0] if job else 0.0,
            message=f"Timed out after {int(timeout)}s (last status: {status})",
            status="warning",
        )
        return f"Error: simulation '{job_id}' did not finish within {int(timeout)}s (last status: {status})."

    return f"Error: Unknown run-control tool '{name}'"


def _emit_job_progress(
    session_context: Optional[Mapping[str, Any]],
    job: Any,
    *,
    max_steps: Optional[int] = None,
) -> None:
    if job.status.is_terminal:
        return
    max_steps = max_steps or max_steps_from_job(job)
    pct, step = progress_percent_from_job(job)
    emit_simulation_progress(
        session_context,
        phase="simulation",
        job_id=job.job_id,
        progress_pct=pct,
        message=f"{job.backend} · {job.status.value}",
        status="running",
        step=step,
        max_steps=max_steps,
    )


def _summary(job, *, header: str) -> str:
    lines = [
        f"{header}.",
        f"job_id: {job.job_id}",
        f"backend: {job.backend}",
        f"status: {job.status.value}",
    ]
    if job.progress is not None:
        pct = job.progress * 100.0 if job.progress <= 1.0 else job.progress
        lines.append(f"progress: {pct:.1f}%")
    measured = job.measured or {}
    if measured.get("step") is not None:
        lines.append(f"step: {measured['step']}")
    if job.external_id:
        lines.append(f"process: {job.external_id}")
    if job.return_code is not None:
        lines.append(f"return_code: {job.return_code}")
    if job.message:
        lines.append(f"message: {job.message}")
    return "\n".join(lines)
