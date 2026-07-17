"""Physics-constraint validation tests."""

from case_library import make_case
from physics_constraints import validate_case
from schemas import CFDCase


def _failed(report, name):
    return any(c.name == name and not c.passed for c in report.checks)


def test_valid_hit_case_passes():
    report = validate_case(make_case("hit", "openlb", name="hit"))
    assert report.passed


def test_valid_dhit_case_passes():
    report = validate_case(make_case("hit", "openlb", name="dhit", hit_mode="decaying"))
    assert report.passed


def test_fhit_missing_forcing_band_rejected():
    case = CFDCase.model_validate({
        "name": "bad",
        "flow": {
            "kind": "hit",
            "hit_mode": "forced",
            "forcing_type": "spectral_low_k",
        },
    })
    report = validate_case(case)
    assert not report.passed
    assert _failed(report, "hit_forcing_band")


def test_hit_non_cubic_box_rejected():
    case = CFDCase.model_validate({
        "name": "bad",
        "geometry": {"kind": "box", "size": [1.0, 2.0, 1.0]},
        "flow": {"kind": "hit"},
    })
    report = validate_case(case)
    assert not report.passed
    assert _failed(report, "hit_cube_domain")


def test_hit_non_periodic_boundary_rejected():
    case = CFDCase.model_validate({
        "name": "bad",
        "flow": {"kind": "hit", "forcing_type": "low_wavenumber"},
        "boundary_conditions": [{"region": "all", "type": "no_slip"}],
    })
    report = validate_case(case)
    assert not report.passed
    assert _failed(report, "hit_periodic_boundaries")


def test_lbm_high_mach_rejected():
    case = CFDCase.model_validate({
        "name": "m", "flow": {"kind": "hit"},
        "solver": {"kind": "lbm", "extra": {"mach_number": 0.2}},
    })
    report = validate_case(case)
    assert _failed(report, "lbm_target_mach") or _failed(report, "lbm_low_mach")


def test_lbm_low_tau_flagged():
    case = CFDCase.model_validate({
        "name": "t", "flow": {"kind": "hit"},
        "solver": {"kind": "lbm", "extra": {"relaxation_time": 0.4}},
    })
    report = validate_case(case)
    assert _failed(report, "lbm_tau")


def test_missing_velocity_output_rejected():
    case = CFDCase.model_validate({
        "name": "o", "flow": {"kind": "hit"},
        "outputs": {"write_velocity": False, "write_spectra": True},
    })
    report = validate_case(case)
    assert _failed(report, "velocity_output_required")
