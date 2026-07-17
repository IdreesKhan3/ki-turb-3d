"""Regression tests for spectrum file discovery after OpenLB postprocess."""

from __future__ import annotations

from pathlib import Path

from agents.tools._session_loader import load_manifest_into_context
from agents.tools._shared import resolve_data_dir_and_find_files
from agents.langgraph.router import RequestRouter


def test_spectra_found_from_processed_dir_when_raw_is_data_directory(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "simulations/job_c2830321a38f/manifest.json"
    if not manifest_path.is_file():
        return

    ctx: dict = {}
    ok, _ = load_manifest_into_context(project_root, str(manifest_path), ctx)
    assert ok
    files = resolve_data_dir_and_find_files(
        "",
        "spectrum*.dat",
        project_root,
        ctx,
        100,
    )
    assert len(files) >= 10
    assert all("processed/spectra" in str(f) for f in files)


def test_spectra_found_via_job_id_without_loaded_files_index():
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "simulations/job_c2830321a38f/manifest.json"
    if not manifest_path.is_file():
        return

    ctx = {
        "data_directory": str(project_root / "simulations/job_c2830321a38f/raw"),
        "simulation_job_id": "job_c2830321a38f",
    }
    files = resolve_data_dir_and_find_files(
        "",
        "spectrum*.dat",
        project_root,
        ctx,
        100,
    )
    assert len(files) >= 10


def test_energy_spectra_plan_prepends_manifest_load_for_active_job():
    router = RequestRouter(planner_agent=None)
    plan = router.plan(
        "compute and plot energy spectra from the previous simulation",
        {"simulation_job_id": "job_c2830321a38f"},
    )
    tools = [step.tool for step in plan.steps]
    assert "load_dataset_manifest" in tools
    assert "compute_spectra" in tools
    assert "plot_spectrum" in tools


def test_kolmogorov_line_edit_routes_to_energy_spectra_not_report_reorder():
    from agents.intent_detection import get_plot_routing

    query = (
        "move the -5/3 line above and on the left. "
        "also reduce its length to half and then repolot the corrected figure"
    )
    routing = get_plot_routing(query)
    assert routing["intent"] == "energy_spectra"
    assert routing["tool"] == "plot_spectrum"
    assert "reorder_report_section" not in (routing.get("prevent_tools") or [])
