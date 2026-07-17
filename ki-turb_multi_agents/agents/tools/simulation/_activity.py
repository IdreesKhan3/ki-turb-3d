"""Emit structured simulation progress events to the Autonomous Lab activity UI."""
from __future__ import annotations

from typing import Any, Mapping, Optional

ACTIVITY_CALLBACK_KEY = "_activity_callback"
ACTIVITY_RENDER_CALLBACK_KEY = "_activity_render_callback"
PROGRESS_QUEUE_KEY = "_simulation_progress_queue"
LATEST_PROGRESS_KEY = "_simulation_progress"


def _safe_callback(callback: Any, event: dict[str, Any]) -> None:
    """Invoke the UI callback without aborting long-running simulation tools."""
    try:
        callback(event)
    except Exception:
        return


def emit_simulation_progress(
    session_context: Optional[Mapping[str, Any]],
    *,
    phase: str,
    job_id: str,
    progress_pct: float,
    message: str = "",
    status: str = "running",
    step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> None:
    """Push a live progress update to the chat UI when a callback is registered."""
    if not session_context:
        return

    pct = max(0.0, min(100.0, float(progress_pct)))
    summary = message.strip()
    if step is not None and max_steps:
        step_pct = 100.0 * float(step) / float(max_steps) if max_steps else pct
        summary = (
            f"{message.strip() + ' · ' if summary else ''}"
            f"step {int(step):,} / {int(max_steps):,} ({step_pct:.1f}%)"
        ).strip(" ·")
    elif not summary:
        summary = f"{pct:.1f}% complete"

    event = {
        "type": "simulation_progress",
        "kind": "progress",
        "agent": "simulation",
        "status": status,
        "title": phase.replace("_", " ").title(),
        "summary": summary,
        "progress": pct,
        "job_id": job_id,
        "phase": phase,
    }

    # Always queue for the workflow UI; Streamlit may not be available during blocking tools.
    if hasattr(session_context, "setdefault"):
        queue = session_context.setdefault(PROGRESS_QUEUE_KEY, [])
        queue.append(event)
        session_context[LATEST_PROGRESS_KEY] = event

    callback = session_context.get(ACTIVITY_CALLBACK_KEY)
    if callable(callback):
        _safe_callback(callback, event)


def max_steps_from_job(job: Any) -> int:
    """Read max_steps from a SimulationJob's typed or legacy config."""
    requested = getattr(job, "requested_config", None) or {}
    runtime = requested.get("runtime") or {}
    resources = getattr(job, "resources", None) or {}
    for value in (runtime.get("max_steps"), resources.get("max_steps")):
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
    return 10_000


def progress_percent_from_job(job: Any) -> tuple[float, Optional[int]]:
    """Return (percent 0–100, latest step) from job progress or measured diagnostics."""
    measured = getattr(job, "measured", None) or {}
    step = measured.get("step")
    try:
        step = int(step) if step is not None else None
    except (TypeError, ValueError):
        step = None

    raw = getattr(job, "progress", None)
    if raw is not None:
        try:
            fraction = float(raw)
            if fraction <= 1.0:
                return fraction * 100.0, step
            return min(100.0, fraction), step
        except (TypeError, ValueError):
            pass

    max_steps = max_steps_from_job(job)
    if step is not None:
        return min(100.0, 100.0 * step / max_steps), step
    return 0.0, step


__all__ = [
    "ACTIVITY_CALLBACK_KEY",
    "ACTIVITY_RENDER_CALLBACK_KEY",
    "LATEST_PROGRESS_KEY",
    "PROGRESS_QUEUE_KEY",
    "emit_simulation_progress",
    "flush_simulation_progress",
    "max_steps_from_job",
    "progress_percent_from_job",
]


def flush_simulation_progress(session_context: Optional[Mapping[str, Any]]) -> None:
    """Replay queued progress events when Streamlit context is available again."""
    if not session_context or not hasattr(session_context, "pop"):
        return
    callback = session_context.get(ACTIVITY_CALLBACK_KEY)
    if not callable(callback):
        return
    queue = list(session_context.pop(PROGRESS_QUEUE_KEY, []) or [])
    latest = session_context.get(LATEST_PROGRESS_KEY)
    if latest and (not queue or queue[-1] is not latest):
        queue.append(latest)
    for event in queue:
        _safe_callback(callback, event)
