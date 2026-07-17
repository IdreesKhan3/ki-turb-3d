"""Tests for the CFD/simulation schemas: manifest, job, and case round-trips."""

from schemas import (
    CFDCase,
    DatasetFile,
    DatasetManifest,
    JobStatus,
    SimulationJob,
)


def _manifest() -> DatasetManifest:
    return DatasetManifest(manifest_id="ds_1", base_dir="/tmp/out", backend="openlb")


def test_add_file_collects_and_sorts_time_steps():
    manifest = _manifest()
    manifest.add_file(DatasetFile(path="u_2000.vti", kind="field", time_step=2000))
    manifest.add_file(DatasetFile(path="u_1000.vti", kind="field", time_step=1000))
    manifest.add_file(DatasetFile(path="log.txt", kind="log"))

    assert manifest.time_steps == [1000, 2000]
    assert len(manifest.files) == 3


def test_add_file_deduplicates_time_steps():
    manifest = _manifest()
    manifest.add_file(DatasetFile(path="u_1000.vti", kind="field", time_step=1000))
    manifest.add_file(DatasetFile(path="p_1000.vti", kind="field", time_step=1000))

    assert manifest.time_steps == [1000]


def test_files_of_kind_filters():
    manifest = _manifest()
    manifest.add_file(DatasetFile(path="u.vti", kind="field"))
    manifest.add_file(DatasetFile(path="s.csv", kind="table"))

    fields = manifest.files_of_kind("field")
    assert len(fields) == 1
    assert fields[0].path == "u.vti"


def test_manifest_json_round_trip():
    manifest = _manifest()
    manifest.add_file(DatasetFile(path="u.vti", kind="field", format="vti", size_bytes=42))
    restored = DatasetManifest.from_json(manifest.to_json())

    assert restored.manifest_id == manifest.manifest_id
    assert restored.backend == "openlb"
    assert restored.files[0].size_bytes == 42


def test_simulation_job_mark_sets_timestamps():
    job = SimulationJob(job_id="job_1", backend="openlb")
    assert job.status is JobStatus.PENDING

    job.mark(JobStatus.SUBMITTED)
    job.mark(JobStatus.RUNNING)
    job.mark(JobStatus.COMPLETED, return_code=0)

    assert job.submitted_at is not None
    assert job.started_at is not None
    assert job.finished_at is not None
    assert job.status.is_terminal
    assert job.return_code == 0


def test_simulation_job_json_round_trip():
    job = SimulationJob(job_id="job_1", backend="palabos", case_name="hit")
    job.mark(JobStatus.RUNNING)
    restored = SimulationJob.from_json(job.to_json())

    assert restored.job_id == "job_1"
    assert restored.backend == "palabos"
    assert restored.status is JobStatus.RUNNING


def test_cfd_case_defaults_and_round_trip():
    case = CFDCase(name="hit")
    restored = CFDCase.from_json(case.to_json())

    assert restored.name == "hit"
    assert restored.mesh.resolution == (128, 128, 128)
    assert restored.schema_version == case.schema_version
