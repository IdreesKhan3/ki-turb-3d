"""Case-factory tests."""

import pytest

from case_library import make_case
from case_library.registry import has_factory
from schemas.cfd_case import FlowKind, GeometryKind, SolverKind


def test_openlb_hit_factory_builds_valid_case():
    case = make_case("hit", "openlb", name="hit_32", resolution=[32, 32, 32])
    assert case.name == "hit_32"
    assert case.geometry.kind == GeometryKind.BOX
    assert case.mesh.resolution == (32, 32, 32)
    assert case.solver.kind == SolverKind.LBM
    assert case.flow.kind == FlowKind.HIT
    assert case.flow.hit_mode.value == "forced"
    assert case.solver.extra["mach_number"] < 0.1
    assert case.solver.extra["relaxation_time"] > 0.5


def test_openlb_dhit_factory():
    case = make_case("hit", "openlb", name="dhit", hit_mode="decaying", resolution=[32, 32, 32])
    assert case.flow.hit_mode.value == "decaying"
    assert case.flow.forcing_type == "none"


def test_unknown_factory_raises():
    assert not has_factory("channel", "openlb")
    with pytest.raises(ValueError):
        make_case("channel", "openlb", name="x")
