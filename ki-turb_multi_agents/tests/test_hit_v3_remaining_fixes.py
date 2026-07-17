from __future__ import annotations

from pathlib import Path

from agents.security.openlb_permissions import (
    OpenLBPermission,
    require_openlb_permission,
)
from agents.tools.data.hit_data_collector import HITDataCollector
from case_library.flows.hit import make_openlb_hit_case
from postprocessing.pipeline import _case
from schemas.openlb_hit import OpenLBHITConfig


def test_default_hit_case_converts_to_typed_openlb_config() -> None:
    case = make_openlb_hit_case(name="default-hit", resolution=(16, 16, 16), hit_mode="decaying", char_velocity=0.1, reynolds_number=100)
    config = OpenLBHITConfig.from_cfd_case(case)
    derived = config.derive_scaling()
    assert config.initial_condition.wavenumber_min is not None
    assert config.initial_condition.wavenumber_max is not None
    assert derived.relaxation_time > 0.5
    assert derived.actual_mach <= config.acceptance.max_mach


def test_top_level_openlb_config_is_read_by_postprocessing(tmp_path: Path) -> None:
    config = OpenLBHITConfig.from_cfd_case(make_openlb_hit_case(name="typed-hit", resolution=(16, 16, 16), hit_mode="decaying", char_velocity=0.1, reynolds_number=100))
    path = tmp_path / "requested_case.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    dx, viscosity, _max_steps, loaded = _case(str(path))
    assert loaded is not None
    assert dx > 0
    assert viscosity is not None and viscosity > 0


def test_forcing_state_is_not_classified_as_forcing_field(tmp_path: Path) -> None:
    path = tmp_path / "forcing_state_10.txt"
    path.write_text("rng state", encoding="utf-8")
    kind, fmt, _metadata = HITDataCollector().classify(path)
    assert kind == "forcing_state"
    assert fmt == "txt"


def test_openlb_permission_enforcement() -> None:
    require_openlb_permission("simulation", OpenLBPermission.RUN)
