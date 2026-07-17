"""Spectral isotropy time-average must follow isotropy_coeff IC definitions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core_physics.spectral_isotropy import avg_isotropy_coeff, read_isotropy_coeff_file, snapshot_ic_curve
from postprocessing.isotropy_from_fields import compute_spectral_isotropy
from postprocessing.readers import VelocitySnapshot


def test_avg_isotropy_not_identically_one_on_les_files():
    root = Path(__file__).resolve().parents[1] / "examples" / "LES" / "64"
    files = sorted(root.glob("isotropy_coeff_data4_*.dat"))[:8]
    assert files, "expected LES example isotropy_coeff files"
    data = [read_isotropy_coeff_file(f) for f in files]
    avg = avg_isotropy_coeff(data)
    assert avg is not None
    ic = np.asarray(avg["IC_mean"])
    assert ic.size > 0
    assert not np.allclose(ic[np.isfinite(ic)], 1.0)
    assert np.nanmax(np.asarray(avg["IC_std"])) > 0
    assert 0.5 < float(np.nanmedian(ic)) < 1.5


def test_dns512_ic_near_one_not_flat_zero():
    root = Path(__file__).resolve().parents[1] / "examples" / "DNS" / "512"
    files = sorted(root.glob("isotropy_coeff_data3_*.dat"))[::20]
    assert len(files) >= 5
    avg = avg_isotropy_coeff([read_isotropy_coeff_file(f) for f in files])
    assert avg is not None
    ic = np.asarray(avg["IC_mean"])
    finite = ic[np.isfinite(ic) & (ic > 0)]
    assert finite.size > 50
    assert 0.7 < float(np.median(finite)) < 1.3
    # Must not look like a flat zero curve with only a Nyquist spike
    assert float(np.mean(finite < 0.1)) < 0.2


def test_snapshot_ic_uses_standard_column():
    root = Path(__file__).resolve().parents[1] / "examples" / "LES" / "64"
    f = next(root.glob("isotropy_coeff_data4_*.dat"))
    rd = read_isotropy_coeff_file(f)
    k, ic = snapshot_ic_curve(rd, kind="standard")
    assert np.allclose(k, rd[:, 0])
    expected = np.divide(rd[:, 2], rd[:, 1], out=np.full(len(rd), np.nan), where=rd[:, 1] > 1e-15)
    finite = np.isfinite(expected) & np.isfinite(ic)
    assert np.allclose(ic[finite], expected[finite], rtol=1e-6, atol=1e-8)


def test_openlb_isotropy_columns_match_isotropy_coeff_layout():
    rng = np.random.default_rng(0)
    n = 16
    # Mildly anisotropic random field
    vel = rng.normal(size=(n, n, n, 3))
    vel[..., 1] *= 1.1
    snap = VelocitySnapshot(step=1000, time=None, velocity=vel, dx=1.0, spacing=(1.0, 1.0, 1.0))
    products = compute_spectral_isotropy([snap])
    assert len(products) == 1
    cols = products[0]["columns"]
    assert cols.shape[1] == 7
    e11, e22 = cols[:, 1], cols[:, 2]
    ic_std = cols[:, 5]
    expected = np.divide(e22, e11, out=np.full_like(e11, np.nan), where=e11 > 1e-15)
    finite = np.isfinite(expected) & np.isfinite(ic_std)
    assert np.allclose(ic_std[finite], expected[finite], rtol=1e-10, atol=1e-12)
    # Derivative IC present and finite for some shells
    assert np.any(np.isfinite(cols[:, 6]) & (cols[:, 6] > 0))
