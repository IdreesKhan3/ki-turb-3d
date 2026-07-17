"""PDF/velocity discovery must not silently mix examples archives into OpenLB jobs."""
from __future__ import annotations

from pathlib import Path

from agents.tools._shared import (
    prefer_active_job_over_examples_mix,
    resolve_data_dir_and_find_files,
)
from agents.tools.physics.pdfs import _resolve_velocity_file_groups


def test_prefer_active_job_drops_examples_mix(tmp_path: Path):
    job = "job_test123"
    job_raw = tmp_path / "simulations" / job / "raw"
    job_raw.mkdir(parents=True)
    ex = tmp_path / "examples" / "DNS" / "128"
    ex.mkdir(parents=True)
    mixed = [str(job_raw), str(ex)]
    cleaned = prefer_active_job_over_examples_mix(
        mixed, tmp_path, {"simulation_job_id": job}
    )
    assert cleaned == [str(job_raw.resolve())] or cleaned == [str(job_raw)]


def test_resolve_velocity_groups_uses_job_not_project_rglob(tmp_path: Path):
    job = "job_pdf_only"
    job_raw = tmp_path / "simulations" / job / "raw"
    job_raw.mkdir(parents=True)
    (job_raw / "vel_000.h5").write_bytes(b"not-a-real-h5")
    ex = tmp_path / "examples" / "DNS" / "512"
    ex.mkdir(parents=True)
    (ex / "vel_dns.h5").write_bytes(b"not-a-real-h5")

    # Empty session dirs but active job — must not invent examples groups via rglob.
    groups = _resolve_velocity_file_groups(
        None,
        "",
        tmp_path,
        {"simulation_job_id": job},
        max_files_per_group=1,
    )
    # Fake h5 will not load later, but discovery paths must stay under the job.
    for paths in groups.values():
        for p in paths:
            assert "examples" not in str(p)
            assert job in str(p)


def test_no_directory_no_job_does_not_rglob_examples(tmp_path: Path):
    ex = tmp_path / "examples" / "LES" / "64"
    ex.mkdir(parents=True)
    (ex / "spectrum1.dat").write_text("k E\n", encoding="utf-8")
    files = resolve_data_dir_and_find_files("", "spectrum*.dat", tmp_path, {}, 10)
    assert files == []
