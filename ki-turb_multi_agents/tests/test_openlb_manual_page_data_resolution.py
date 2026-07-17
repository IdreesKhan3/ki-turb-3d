"""Manual pages must resolve OpenLB products under processed/ when session points at raw/."""

from __future__ import annotations

from pathlib import Path

from utils.file_detector import (
    detect_simulation_files,
    expand_analysis_search_dirs,
    list_velocity_field_files,
)
from agents.tools._session_loader import load_manifest_into_context
from pages.VolumeViewer3D.file_loading import collect_volume_files


def test_expand_analysis_search_dirs_includes_processed_stats():
    project_root = Path(__file__).resolve().parents[1]
    raw = project_root / "simulations/job_5fa8049d84b4/raw"
    if not raw.is_dir():
        return
    dirs = expand_analysis_search_dirs(raw)
    assert any(p.name == "stats" for p in dirs)
    assert any(p.name == "spectra" for p in dirs)


def test_detect_simulation_files_from_raw_finds_eps_validation():
    project_root = Path(__file__).resolve().parents[1]
    raw = project_root / "simulations/job_5fa8049d84b4/raw"
    if not raw.is_dir():
        return
    files = detect_simulation_files(str(raw))
    eps = files.get("spectral_turb_stats") or []
    assert eps
    assert any("eps_real_validation" in Path(p).name for p in eps)
    assert any("processed/stats" in str(p) for p in eps)


def test_manifest_load_indexes_processed_stats_for_manual_pages():
    project_root = Path(__file__).resolve().parents[1]
    manifest = project_root / "simulations/job_5fa8049d84b4/manifest.json"
    if not manifest.is_file():
        return
    ctx: dict = {}
    ok, _ = load_manifest_into_context(project_root, str(manifest), ctx)
    assert ok
    assert ctx.get("stats_data_directory")
    assert "processed/stats" in str(ctx["stats_data_directory"])
    indexed = (ctx.get("all_loaded_files") or {}).get("spectral_turb_stats") or []
    assert indexed
    assert any("eps_real_validation" in item["filename"] for item in indexed)


def test_structure_functions_page_dedupes_openlb_dirs():
    from pages.StructureFunctions.file_loading import _load_structure_groups
    from data_readers.text_reader import read_structure_function_txt

    project_root = Path(__file__).resolve().parents[1]
    job = project_root / "simulations/job_73df70275bab"
    if not (job / "processed/structure_functions").is_dir():
        return
    dirs = [
        str(job / "raw"),
        str(job / "processed"),
        str(job / "processed/structure_functions"),
    ]
    # Prefer product dir as session would after manifest load.
    import streamlit as st
    # unit path without streamlit session: call collector via preferred in list only
    groups = _load_structure_groups(dirs)
    assert groups is not None
    # One physical series, not raw_/processed_/structure_functions_ clones
    assert len(groups) == 1
    key = next(iter(groups))
    assert "structure_functions" in key
    files = groups[key]["files"]
    assert files
    data = read_structure_function_txt(files[0])
    assert data["u_rms"] > 0.0
    assert 2 in data["S_p"]
