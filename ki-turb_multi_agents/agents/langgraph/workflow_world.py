"""Explicit workflow world state for solver/data agents.

Snapshot of job + session facts the planner/guards/verifier share.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.simulation_job import JobStatus


class WorkflowWorldState(BaseModel):
    """Typed facts about the current simulation/data world."""

    model_config = ConfigDict(extra="forbid")

    job_id: Optional[str] = None
    backend: Optional[str] = None
    job_status: Optional[str] = None
    has_job_record: bool = False
    has_case: bool = False
    has_executable: bool = False
    has_manifest: bool = False
    has_raw_outputs: bool = False
    has_processed_products: bool = False
    session_has_data: bool = False
    session_manifest_path: Optional[str] = None
    capabilities: Dict[str, bool] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    def capability(self, name: str) -> bool:
        return bool(self.capabilities.get(name))


def _status_value(job) -> Optional[str]:
    if job is None:
        return None
    status = getattr(job, "status", None)
    if isinstance(status, JobStatus):
        return status.value
    return str(status) if status is not None else None


def snapshot_world(
    project_root: str | Path,
    *,
    job_id: Optional[str] = None,
    session_context: Optional[Dict[str, Any]] = None,
) -> WorkflowWorldState:
    """Build world state from durable job files + live session context."""
    from agents.tools.simulation import _store as job_store

    root = Path(project_root).resolve()
    sess = session_context or {}
    warnings: List[str] = []

    resolved_job = (job_id or "").strip() or None
    if not resolved_job:
        for key in ("simulation_job_id", "sim_workflow_job"):
            value = str(sess.get(key) or "").strip()
            if value and not value.startswith("__"):
                resolved_job = value
                break

    job = job_store.load_job(root, resolved_job) if resolved_job else None
    if resolved_job and job is None:
        warnings.append(f"job record missing for {resolved_job}")

    job_dir = job_store.job_dir(root, resolved_job) if resolved_job else None
    manifest_path = (job_dir / job_store.MANIFEST_FILENAME) if job_dir else None
    has_manifest = bool(manifest_path and manifest_path.is_file())
    has_case = bool(job_dir and ((job_dir / "case.xml").is_file() or (job_dir / "case.json").is_file()))
    has_executable = False
    if job_dir and (job_dir / "executable").is_dir():
        has_executable = any((job_dir / "executable").iterdir())
    elif job is not None:
        exe = (job.metadata or {}).get("executable") or (job.resources or {}).get("executable")
        has_executable = bool(exe and Path(str(exe)).is_file())

    has_raw = False
    has_processed = False
    if job_dir:
        raw = job_dir / "raw"
        processed = job_dir / "processed"
        has_raw = raw.is_dir() and any(raw.iterdir())
        has_processed = processed.is_dir() and any(processed.rglob("*"))

    session_manifest = str(
        sess.get("manifest_path") or sess.get("dataset_manifest_path") or ""
    ).strip() or None
    session_has_data = bool(
        sess.get("data_directory")
        or sess.get("data_directories")
        or sess.get("all_loaded_files")
        or session_manifest
    )

    status = _status_value(job)
    status_enum = None
    if job is not None and isinstance(job.status, JobStatus):
        status_enum = job.status

    can_compile = bool(
        job is not None
        and status_enum
        in {
            JobStatus.CREATED,
            JobStatus.PENDING,
            JobStatus.PREPARED,
            JobStatus.VALIDATED,
            JobStatus.FAILED,
            JobStatus.BUILT,
            JobStatus.COMPILED,
            JobStatus.BUILDING,
            JobStatus.COMPILING,
        }
    )
    can_start = bool(
        job is not None
        and (
            status_enum in {JobStatus.BUILT, JobStatus.COMPILED}
            or (has_executable and status_enum in {JobStatus.PREPARED, JobStatus.BUILT, JobStatus.COMPILED})
        )
    )
    can_supervise = bool(
        job is not None
        and status_enum in {JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.SUBMITTED, JobStatus.CHECKPOINTING}
    )
    can_fetch = bool(
        job is not None
        and status_enum
        in {
            JobStatus.COMPLETED,
            JobStatus.FETCHED,
            JobStatus.ANALYSIS_READY,
            JobStatus.POSTPROCESSED,
            JobStatus.ANALYSED,
        }
    )
    can_postprocess = bool(has_manifest or (job is not None and status_enum in {JobStatus.FETCHED, JobStatus.COMPLETED}))
    can_load = bool(has_manifest or (session_manifest and Path(session_manifest).is_file()))
    can_analyze = bool(can_load or session_has_data or has_processed)

    return WorkflowWorldState(
        job_id=resolved_job,
        backend=(job.backend if job is not None else None),
        job_status=status,
        has_job_record=job is not None,
        has_case=has_case,
        has_executable=has_executable,
        has_manifest=has_manifest,
        has_raw_outputs=has_raw,
        has_processed_products=has_processed,
        session_has_data=session_has_data,
        session_manifest_path=session_manifest,
        capabilities={
            "build": True,
            "compile": can_compile,
            "start": can_start,
            "supervise": can_supervise,
            "fetch": can_fetch,
            "postprocess": can_postprocess,
            "load": can_load,
            "analyze": can_analyze,
            "status": job is not None,
            "cancel": can_supervise or (status_enum == JobStatus.RUNNING),
        },
        warnings=warnings,
    )


__all__ = ["WorkflowWorldState", "snapshot_world"]
