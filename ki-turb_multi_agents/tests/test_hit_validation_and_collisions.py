"""Validation and collision-operator behaviour for OpenLB HIT."""

from agents.langgraph.openlb_hit_build import parse_openlb_build_args
from agents.physics_constraint_agent import PhysicsConstraintAgent
from agents.tools.physics.hit_validation_agent import HITValidationAgent
from agents.tools.simulation.hit_supervisor import HITDiagnostics, HITSupervisor
from schemas.hit_analysis_products import HITAnalysisProducts, TimeHistoryProduct
from schemas.openlb_hit import HITAcceptanceThresholds, OpenLBHITConfig


def test_hit_validation_only_errors_on_divergence():
    thresholds = HITAcceptanceThresholds(maximum_divergence_rms=0.01, max_mach=0.05, minimum_kmax_eta=2.0)
    products = HITAnalysisProducts(
        run_id="test",
        spectra=[{"step": 1, "wavenumber": [1.0], "energy": [1.0]}],
        time_history=TimeHistoryProduct(
            step=[1],
            mach_max=[0.2],
            divergence_rms=[0.005],
        ),
    )
    report = HITValidationAgent(thresholds).validate(products)
    assert report.passed
    assert any(check.name == "measured_mach" and check.severity == "warning" for check in report.checks)
    assert any(check.name == "measured_divergence" and check.severity == "error" for check in report.checks)


def test_hit_validation_fails_on_divergence_only():
    thresholds = HITAcceptanceThresholds(maximum_divergence_rms=0.01)
    products = HITAnalysisProducts(
        run_id="test",
        spectra=[{"step": 1, "wavenumber": [1.0], "energy": [1.0]}],
        time_history=TimeHistoryProduct(
            step=[1],
            mach_max=[0.01],
            divergence_rms=[0.05],
        ),
    )
    report = HITValidationAgent(thresholds).validate(products)
    assert not report.passed
    assert any(check.name == "measured_divergence" and not check.passed for check in report.checks)


def test_supervisor_aborts_on_divergence_not_mach():
    supervisor = HITSupervisor(HITAcceptanceThresholds(maximum_divergence_rms=0.01, max_mach=0.05))
    healthy = supervisor.evaluate(
        HITDiagnostics(step=10, mach_max=0.2, divergence_rms=0.005, mass=1.0)
    )
    assert healthy.healthy
    assert not healthy.should_abort
    assert any("Mach" in warning for warning in healthy.warnings)

    unhealthy = supervisor.evaluate(
        HITDiagnostics(step=11, mach_max=0.01, divergence_rms=0.05, mass=1.0)
    )
    assert not unhealthy.healthy
    assert unhealthy.should_abort


def test_all_openlb_hit_collision_operators_are_parsed():
    cases = {
        "FHIT 32^3 BGK spectral forcing": "BGK",
        "DHIT 32^3 RLB": "RLB",
        "FHIT 32^3 MRT spectral forcing": "MRT",
        "FHIT 32^3 TRT spectral forcing": "TRT",
        "FHIT 32^3 Smagorinsky spectral forcing": "Smagorinsky",
        "FHIT 32^3 WALE spectral forcing": "WALE",
        "DHIT 32^3 ConsistentStrainSmagorinsky": "ConsistentStrainSmagorinsky",
        "FHIT 32^3 ShearSmagorinsky spectral forcing": "ShearSmagorinsky",
        "DHIT 32^3 Krause": "Krause",
        "DHIT 32^3 DynamicSmagorinsky": "DynamicSmagorinsky",
    }
    for text, expected in cases.items():
        args = parse_openlb_build_args(text)
        assert args["scheme"] == expected, text


def test_collision_operator_reaches_typed_config():
    config = OpenLBHITConfig(
        domain={"resolution": (16, 16, 16), "size": (6.28, 6.28, 6.28)},
        scaling={"characteristic_length": 6.28, "characteristic_velocity": 0.1, "reynolds_number": 100.0},
        collision={"model": "Krause"},
        initial_condition={"type": "synthetic_spectrum", "wavenumber_min": 1, "wavenumber_max": 3},
        forcing={"type": "none"},
    )
    decision = PhysicsConstraintAgent().validate(config)
    assert decision.accepted
    assert decision.config.collision.model.value == "Krause"
