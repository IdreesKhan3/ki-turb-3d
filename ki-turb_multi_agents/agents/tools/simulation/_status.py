"""Shared simulation job status helpers for tool responses."""
from __future__ import annotations

from schemas.simulation_job import JobStatus, SimulationJob

_FAILURE_STATUSES = frozenset({
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.REJECTED,
})

_FETCHABLE_STATUSES = frozenset({
    JobStatus.COMPLETED,
    JobStatus.FETCHED,
    JobStatus.PREPARED,
    JobStatus.BUILT,
})


def simulation_job_failed(job: SimulationJob) -> bool:
    return job.status in _FAILURE_STATUSES


def simulation_job_ready_for_fetch(job: SimulationJob) -> bool:
    return job.status in _FETCHABLE_STATUSES


def format_job_error(job: SimulationJob, *, action: str) -> str:
    detail = job.message.strip() or "no details recorded"
    return (
        f"Error: cannot {action} for job '{job.job_id}' "
        f"(status: {job.status.value}): {detail}"
    )


def tool_text_indicates_failure(text: str) -> bool:
    """Detect failed simulation tool responses that do not use an Error: prefix."""
    lowered = (text or "").strip().lower()
    if lowered.startswith(("error:", "tool error:")):
        return True
    for line in lowered.splitlines():
        if line.startswith("status:"):
            status = line.split(":", 1)[1].strip()
            if status in {"failed", "cancelled", "rejected"}:
                return True
    return False


__all__ = [
    "simulation_job_failed",
    "simulation_job_ready_for_fetch",
    "format_job_error",
    "tool_text_indicates_failure",
]
