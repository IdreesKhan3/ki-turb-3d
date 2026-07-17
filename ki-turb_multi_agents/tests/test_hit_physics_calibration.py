"""Physics calibration for OpenLB HIT/FHIT build requests."""

from agents.langgraph.openlb_hit_build import parse_openlb_build_args
from agents.physics_constraint_agent import PhysicsConstraintAgent
from agents.tools.simulation.hit_calibration import build_args_to_openlb_config
from physics_constraints import validate_case
from case_library import make_case


def _validated_build_args(text: str) -> dict:
    args = parse_openlb_build_args(text)
    agent = PhysicsConstraintAgent()
    config, decision = agent.calibrate(build_args_to_openlb_config(args))
    assert decision.accepted, decision.errors
    kwargs = dict(args)
    if kwargs.get("resolution"):
        kwargs["resolution"] = tuple(kwargs["resolution"])
    case = make_case("hit", "openlb", **{
        key: value for key, value in kwargs.items()
        if key not in {"backend", "flow", "case"}
    })
    report = validate_case(case)
    assert report.passed, [check.message for check in report.errors()]
    return args


def test_calibrated_fhit_8_cube_passes_physics():
    args = _validated_build_args("from OpenLB run FHIT 8^3 grid Smagorinsky with spectral forcing")
    assert args["resolution"] == [8, 8, 8]
    assert args["ic_wavenumber_max"] <= 3
    assert args["forcing_wavenumber_max"] <= 3


def test_calibrated_fhit_16_cube_passes_physics():
    args = _validated_build_args("from OpenLB run FHIT 16^3 grid Smagorinsky with spectral forcing")
    assert args["resolution"] == [16, 16, 16]
    assert args["ic_wavenumber_max"] <= 4
    config, decision = PhysicsConstraintAgent().calibrate(build_args_to_openlb_config(args))
    assert decision.accepted
    assert config.acceptance.maximum_divergence_rms >= 0.25


def test_build_case_preserves_calibrated_acceptance():
    from agents.tools.simulation.case_builder import _build_case
    from schemas.openlb_hit import OpenLBHITConfig

    case = _build_case(
        "openlb",
        {
            "backend": "openlb",
            "flow": "hit",
            "name": "FHIT_16",
            "resolution": [16, 16, 16],
            "scheme": "Smagorinsky",
            "forcing_type": "spectral_low_k",
            "hit_mode": "forced",
        },
    )
    hit = OpenLBHITConfig.from_cfd_case(case)
    assert hit.acceptance.maximum_divergence_rms >= 0.25


def test_calibrated_fhit_64_grid_passes_physics():
    args = _validated_build_args("from OpenLB run FHIT 64 grid Smagorinsky with spectral forcing")
    assert args["resolution"] == [64, 64, 64]
    derived = build_args_to_openlb_config(args).derive_scaling()
    assert derived.actual_mach > 0


def test_user_locked_tau_is_preserved():
    args = parse_openlb_build_args("FHIT 32^3 Smagorinsky tau=0.55 spectral forcing")
    assert args["relaxation_time"] == 0.55


def test_high_mach_does_not_block_validation():
    from agents.tools.simulation.case_builder import _build_case

    case = _build_case(
        "openlb",
        {
            "backend": "openlb",
            "flow": "hit",
            "name": "high_mach",
            "resolution": [32, 32, 32],
            "scheme": "BGK",
            "hit_mode": "decaying",
            "mach_number": 0.2,
        },
    )
    report = validate_case(case)
    assert not any(check.name == "lbm_target_mach" and check.severity == "error" for check in report.errors())
