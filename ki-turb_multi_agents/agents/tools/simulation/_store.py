"""Persistence helpers for simulation jobs and dataset manifests.

Jobs live under ``<project_root>/simulations/<job_id>/`` with the durable job
record in ``job.json`` and any fetched dataset manifest in ``manifest.json``.
All writes are validated against the project path policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from schemas import DatasetManifest, SimulationJob
from ...security.path_policy import default_policy

SIMULATIONS_DIRNAME = "simulations"
JOB_FILENAME = "job.json"
MANIFEST_FILENAME = "manifest.json"


def simulations_root(project_root: Path) -> Path:
    return Path(project_root) / SIMULATIONS_DIRNAME


def job_dir(project_root: Path, job_id: str) -> Path:
    return simulations_root(project_root) / job_id


def _guarded_mkdir(project_root: Path, target: Path) -> Path:
    policy = default_policy(project_root, case_dir=simulations_root(project_root))
    resolved = policy.resolve_write(target)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_job(project_root: Path, job: SimulationJob) -> Path:
    directory = _guarded_mkdir(project_root, job_dir(project_root, job.job_id))
    path = directory / JOB_FILENAME
    default_policy(project_root, case_dir=simulations_root(project_root)).assert_write_allowed(path)
    path.write_text(job.to_json(), encoding="utf-8")
    return path


def load_job(project_root: Path, job_id: str) -> Optional[SimulationJob]:
    path = job_dir(project_root, job_id) / JOB_FILENAME
    if not path.is_file():
        return None
    return SimulationJob.from_json(path.read_text(encoding="utf-8"))


def save_manifest(project_root: Path, job_id: str, manifest: DatasetManifest) -> Path:
    directory = _guarded_mkdir(project_root, job_dir(project_root, job_id))
    path = directory / MANIFEST_FILENAME
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_manifest(project_root: Path, job_id: str) -> Optional[DatasetManifest]:
    path = job_dir(project_root, job_id) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    return DatasetManifest.from_json(path.read_text(encoding="utf-8"))


def list_job_ids(project_root: Path) -> list[str]:
    """Return job ids under simulations/, newest first (by directory mtime)."""
    root = simulations_root(project_root)
    if not root.is_dir():
        return []
    jobs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("job_")]
    jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in jobs]


def latest_job_id_with_manifest(project_root: Path) -> Optional[str]:
    """Newest job that already has a dataset manifest (ready to load)."""
    for job_id in list_job_ids(project_root):
        if (job_dir(project_root, job_id) / MANIFEST_FILENAME).is_file():
            return job_id
    return None


def latest_job_id(project_root: Path) -> Optional[str]:
    ids = list_job_ids(project_root)
    return ids[0] if ids else None
