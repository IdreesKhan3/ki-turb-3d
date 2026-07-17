from pathlib import Path

import sys
import types
import pytest


class _StreamlitStub(types.ModuleType):
    session_state = {}

    def __getattr__(self, name):
        if name in {"cache_data", "cache_resource"}:
            def decorator(func=None, **kwargs):
                return func if func is not None else (lambda wrapped: wrapped)
            return decorator
        def noop(*args, **kwargs):
            return None
        return noop


sys.modules.setdefault("streamlit", _StreamlitStub("streamlit"))

from agents.hit_master_agent import HITMasterAgent
from agents.simulation_agent import SimulationSession
from schemas import DatasetFile, DatasetManifest
from schemas.hit_analysis_products import (
    EnergySpectrumProduct,
    HITAnalysisProducts,
    ProductProvenance,
    ReynoldsStressProduct,
)
from schemas.openlb_hit import OpenLBHITConfig


pytest.importorskip("matplotlib")


def _config() -> OpenLBHITConfig:
    return OpenLBHITConfig(
        name="offline_e2e",
        domain={"resolution": (32, 32, 32), "size": (1.0, 1.0, 1.0)},
        scaling={
            "characteristic_length": 1.0,
            "characteristic_velocity": 0.1,
            "reynolds_number": 100.0,
        },
        collision={"model": "BGK"},
        initial_condition={
            "type": "synthetic_spectrum",
            "wavenumber_min": 1,
            "wavenumber_max": 6,
        },
        forcing={"type": "none"},
    )


def test_offline_analysis_visualization_report_pipeline(tmp_path: Path):
    config = _config()
    run_dir = tmp_path / "run"
    for name in ("case", "build", "raw"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    session = SimulationSession(
        run_id="hit_test",
        status="fetched",
        run_dir=str(run_dir),
        case_dir=str(run_dir / "case"),
        build_dir=str(run_dir / "build"),
        output_dir=str(run_dir / "raw"),
        diagnostics_path=str(run_dir / "raw" / "diagnostics.jsonl"),
        config=config,
    )
    provenance = ProductProvenance(run_id="hit_test", source_steps=[100])
    products = HITAnalysisProducts(
        run_id="hit_test",
        spectra=[
            EnergySpectrumProduct(
                step=100,
                wavenumber=[1.0, 2.0, 3.0, 4.0],
                energy=[1.0, 0.3, 0.12, 0.06],
                provenance=provenance,
            )
        ],
        reynolds_stress=[
            ReynoldsStressProduct(
                step=100,
                r11=1.0,
                r22=1.0,
                r33=1.0,
                r12=0.0,
                r13=0.0,
                r23=0.0,
                provenance=provenance,
            )
        ],
    )
    manifest = DatasetManifest(
        manifest_id="ds_test",
        base_dir=str(run_dir / "raw"),
        backend="openlb",
        files=[
            DatasetFile(
                path="velocity_t100.vti",
                kind="velocity_field",
                format="vti",
                checksum="sha256:test",
            )
        ],
    )
    result = HITMasterAgent().finalize(session=session, products=products, manifest=manifest)
    assert result.status == "accepted"
    assert Path(result.analysis_products_path).is_file()
    assert Path(result.validation_path).is_file()
    assert Path(result.visualization_dashboard).is_file()
    assert Path(result.report_path).is_file()
