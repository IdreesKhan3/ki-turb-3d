"""Solver-neutral UnitSystem contract and OpenLB filler."""
from __future__ import annotations

from schemas.openlb_hit import OpenLBHITConfig
from schemas.unit_system import UnitFrame, UnitSystem
from integrations.openlb.unit_system import unit_system_from_openlb_hit


def test_openlb_hit_fills_physical_analysis_units():
    cfg = OpenLBHITConfig.model_validate(
        {
            "name": "unit_test",
            "domain": {"resolution": [16, 16, 16], "size": [6.283185307179586] * 3},
            "scaling": {
                "characteristic_velocity": 0.1,
                "reynolds_number": 100.0,
                "relaxation_time": 0.53,
                "target_mach": 0.05,
            },
            "collision": {"model": "BGK"},
            "forcing": {"type": "none"},
            "runtime": {"max_steps": 1000, "output_interval": 100},
        }
    )
    # Clear tau so Mach stays consistent with target (same as calibration path).
    cfg.scaling.relaxation_time = None
    us = unit_system_from_openlb_hit(cfg)
    assert us.source_backend == "openlb"
    assert us.analysis_frame == UnitFrame.PHYSICAL
    us.require_analysis_ready()
    assert us.fields["velocity_field"].frame == UnitFrame.PHYSICAL
    assert us.fields["density_field"].frame == UnitFrame.LATTICE
    assert us.mach is not None and us.mach <= 0.1 + 1e-12
    labels = us.field_labels()
    assert labels["velocity_field"] == "physical_velocity"


def test_legacy_labels_roundtrip():
    us = UnitSystem.from_legacy_labels(
        {"velocity_field": "physical_velocity", "density_field": "lattice_density"},
        source_backend="openlb",
        dx=0.1,
        viscosity=0.01,
    )
    assert us.fields["velocity_field"].frame == UnitFrame.PHYSICAL
    assert us.fields["density_field"].frame == UnitFrame.LATTICE
    us.require_analysis_ready()


def test_cfd_case_accepts_unit_system():
    from schemas import CFDCase

    us = UnitSystem(
        source_backend="openlb",
        analysis_frame=UnitFrame.PHYSICAL,
        dx=1.0,
        viscosity=0.1,
    )
    case = CFDCase(name="t", units=us)
    assert case.units is not None
    assert case.units.dx == 1.0
