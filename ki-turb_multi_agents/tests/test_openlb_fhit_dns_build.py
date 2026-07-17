"""Deterministic parsing for FHIT DNS OpenLB build requests."""

from agents.langgraph.openlb_hit_build import parse_openlb_build_args


USER_QUERY = (
    "in openlb run fhit dns with 16^3 and tau 0.507 for 5000 iterations "
    "and save data after each 1000 iterations"
)


def test_fhit_dns_mrt_tau_without_equals_parses():
    query = (
        "in openlb run fhit dns mrt collision with 16^3 and tau 0.507 "
        "for 5000 iterations and save data after each 1000 iterations"
    )
    args = parse_openlb_build_args(query)
    assert args["scheme"] == "MRT"
    assert args["relaxation_time"] == 0.507
    assert args["resolution"] == [16, 16, 16]
    assert args["max_steps"] == 5000
    assert args["output_interval"] == 1000

    from agents.tools.simulation.hit_calibration import build_args_to_openlb_config

    cfg = build_args_to_openlb_config(args)
    derived = cfg.derive_scaling()
    assert abs(derived.relaxation_time - 0.507) < 1e-9
    assert cfg.scaling.characteristic_velocity == 0.1
    assert cfg.forcing.amplitude == 0.1 / (16**3)
    assert derived.actual_mach <= cfg.scaling.max_mach + 0.02


def test_fhit_dns_request_parses_and_validates():
    args = parse_openlb_build_args(USER_QUERY)
    assert args["hit_mode"] == "forced"
    assert args["resolution"] == [16, 16, 16]
    assert args["relaxation_time"] == 0.507
    assert args["max_steps"] == 5000
    assert args["output_interval"] == 1000
    assert args["turbulence_regime"] == "dns"
    assert args["scheme"] in {"BGK", "DNS"}

    from agents.physics_constraint_agent import PhysicsConstraintAgent
    from agents.tools.simulation.hit_calibration import build_args_to_openlb_config

    config, decision = PhysicsConstraintAgent().calibrate(build_args_to_openlb_config(args))
    assert decision.accepted, decision.errors
    assert abs(config.scaling.relaxation_time - 0.507) < 1e-9
    derived = config.derive_scaling()
    assert derived.actual_mach <= config.scaling.max_mach + 0.02


def test_any_catalog_collision_builds_without_regime_gate():
    from agents.physics_constraint_agent import PhysicsConstraintAgent
    from agents.tools.simulation.hit_calibration import build_args_to_openlb_config

    for scheme in (
        "BGK", "MRT", "TRT", "RLB", "regularized",
        "Smagorinsky", "WALE", "ShearSmagorinsky", "Krause", "DynamicSmagorinsky",
    ):
        args = {
            "backend": "openlb",
            "flow": "hit",
            "name": f"FHIT_{scheme}",
            "resolution": [16, 16, 16],
            "scheme": scheme,
            "hit_mode": "forced",
            "forcing_type": "spectral_low_k",
            "turbulence_regime": "les" if scheme == "BGK" else "dns",
            "max_steps": 100,
            "output_interval": 50,
        }
        config, decision = PhysicsConstraintAgent().calibrate(build_args_to_openlb_config(args))
        assert decision.accepted, f"{scheme}: {decision.errors}"
