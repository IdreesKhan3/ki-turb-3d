"""Tests for solver-neutral analysis product loading."""

from pathlib import Path

from analysis.manifest_index import MANIFEST_KIND_TO_SESSION_KEY
from analysis.product_loader import AnalysisProductLoader
from schemas import DatasetFile, DatasetManifest


def test_manifest_kind_mapping_covers_core_products():
    for kind in (
        "energy_spectrum",
        "spectral_isotropy",
        "flatness",
        "analysis_products",
        "velocity_field",
        "enstrophy_pdf",
        "joint_pdf",
        "rq_pdf",
        "tau_effective_field",
    ):
        assert kind in MANIFEST_KIND_TO_SESSION_KEY


def test_enrich_session_files_indexes_manifest_entries(tmp_path: Path):
    base = tmp_path / "run"
    processed = base / "processed" / "spectra"
    processed.mkdir(parents=True)
    spectrum = processed / "spectrum1.dat"
    spectrum.write_text("1 1\n", encoding="utf-8")
    manifest = DatasetManifest(
        manifest_id="test",
        base_dir=str(base),
        backend="openlb",
        files=[
            DatasetFile(path=str(spectrum.relative_to(base)), kind="energy_spectrum", format="dat"),
        ],
    )
    all_files: dict = {}
    loader = AnalysisProductLoader(tmp_path, {})
    loader._manifest = manifest
    hints = loader.enrich_session_files(all_files)
    assert all_files["spectrum"]
    assert hints["spectra_data_directory"] == str(processed)
