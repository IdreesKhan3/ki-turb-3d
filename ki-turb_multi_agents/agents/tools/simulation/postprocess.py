"""Tool for turning raw simulation fields into KI-TURB-ready turbulence data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from schemas.simulation_job import JobStatus
from postprocessing.pipeline import postprocess_manifest

from . import _store

POSTPROCESS_TOOL_NAMES = frozenset({"postprocess_simulation_outputs"})


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [{
        "name": "postprocess_simulation_outputs",
        "description": "Compute canonical HIT spectra, isotropy, stresses, PDFs, structure functions, stationarity and uncertainty from fetched fields.",
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    }]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    job_id = str(args.get("job_id", "")).strip()
    if not job_id:
        return "Error: job_id required"
    job = _store.load_job(project_root, job_id)
    if job is None:
        return f"Error: no job found with id '{job_id}'."
    manifest = _store.load_manifest(project_root, job_id)
    if manifest is None:
        return "Error: fetch outputs before post-processing."

    typed_case = Path(job.paths.case_dir) / "requested_case.json"
    legacy_case = Path(job.paths.case_dir) / "case.json"
    case_path = typed_case if typed_case.is_file() else legacy_case
    processed_dir = Path(job.paths.processed_dir or (Path(job.paths.case_dir) / "processed"))

    job.mark(JobStatus.POSTPROCESSING, message="post-processing raw CFD fields")
    _store.save_job(project_root, job)
    try:
        manifest = postprocess_manifest(
            manifest,
            str(case_path),
            processed_dir=processed_dir,
        )
    except Exception as exc:
        job.mark(JobStatus.FAILED, message=f"post-processing failed: {exc}")
        _store.save_job(project_root, job)
        return f"Error: post-processing failed: {exc}"

    _store.save_manifest(project_root, job_id, manifest)
    scientific_status = str(manifest.postprocessing.get("validation_status", "unvalidated"))
    products_path = manifest.postprocessing.get("analysis_products_path")
    has_products = bool(products_path and Path(products_path).is_file())
    if not has_products or manifest.postprocessing.get("num_snapshots", 0) == 0:
        job.mark(JobStatus.INSUFFICIENT_DATA, message="no complete velocity snapshots were available")
    else:
        # ANALYSIS_READY means canonical products exist; it does not mean the run
        # has passed scientific review. The independent status is recorded below.
        job.mark(JobStatus.ANALYSIS_READY, message=f"analysis products generated; scientific_status={scientific_status}")
    job.metadata["analysis_products_path"] = str(products_path) if products_path else None
    job.metadata["scientific_status"] = scientific_status
    _store.save_job(project_root, job)

    # Preserve the old UI path without mixing raw and processed ownership.
    if job.paths.output_dir:
        compatibility = Path(job.paths.output_dir) / "processed"
        if not compatibility.exists():
            try:
                compatibility.symlink_to(processed_dir, target_is_directory=True)
            except OSError:
                pass

    return (
        "Post-processing completed.\n"
        f"job_id: {job_id}\n"
        f"status: {job.status.value}\n"
        f"scientific_status: {scientific_status}\n"
        f"analysis_products: {products_path or 'none'}\n"
        f"manifest files: {len(manifest.files)}\n"
        f"base_dir: {manifest.base_dir}"
    )
