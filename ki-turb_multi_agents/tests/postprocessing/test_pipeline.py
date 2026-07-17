"""Post-processing pipeline and field-computation tests."""

import json
from pathlib import Path

import h5py
import numpy as np

from postprocessing.pipeline import postprocess_manifest
from postprocessing.readers import VelocitySnapshot
from postprocessing.spectra_from_fields import compute_energy_spectrum
from schemas import DatasetFile, DatasetManifest


def _write_velocity_h5(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("velocity", data=rng.standard_normal((n, n, n, 3)))


def test_energy_spectrum_conserves_energy():
    rng = np.random.default_rng(0)
    n = 16
    u = rng.standard_normal((n, n, n, 3))
    snap = VelocitySnapshot(step=0, time=None, velocity=u, dx=1.0 / n)
    result = compute_energy_spectrum([snap])[0]

    tke = 0.5 * float(np.mean(
        sum((u[..., i] - u[..., i].mean()) ** 2 for i in range(3))
    ))
    # Shell-summed spectrum should recover the turbulent kinetic energy.
    assert np.isclose(result["E"].sum(), tke, rtol=1e-6)


def test_pipeline_writes_kiturb_outputs(tmp_path):
    base = tmp_path / "output"
    base.mkdir()
    _write_velocity_h5(base / "velocity_1000.h5", 16, seed=1)
    _write_velocity_h5(base / "velocity_2000.h5", 16, seed=2)

    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps({
        "mesh": {"resolution": [16, 16, 16], "dx": 1.0 / 16},
        "solver": {"viscosity": 0.001},
        "runtime": {"max_steps": 2000},
    }), encoding="utf-8")

    manifest = DatasetManifest(manifest_id="m", base_dir=str(base))
    for name in ("velocity_1000.h5", "velocity_2000.h5"):
        manifest.add_file(DatasetFile(path=name, kind="velocity_field", format="h5"))

    manifest = postprocess_manifest(manifest, str(case_path))

    kinds = {f.kind for f in manifest.files}
    assert {"energy_spectrum", "normalized_spectrum", "spectral_isotropy",
            "flatness", "structure_functions", "turbulence_stats"} <= kinds
    assert (base / "processed" / "spectra" / "spectrum_data1_1000.dat").is_file()
    assert manifest.postprocessing["num_snapshots"] == 2
