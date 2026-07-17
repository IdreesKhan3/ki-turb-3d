"""Tests for OpenLB HIT capability catalog."""

from integrations.openlb_hit_catalog import (
    ALL_COLLISIONS,
    collision_allowed_for_regime,
    normalize_collision,
    normalize_turbulence_regime,
    xml_collision_name,
)


def test_all_tgv_collision_models_registered():
    for name in (
        "BGK", "DNS", "RLB", "MRT", "TRT",
        "Smagorinsky", "WALE", "ConsistentStrainSmagorinsky",
        "ShearSmagorinsky", "Krause", "SmagorinskyMRT", "DynSmagorinsky",
        "regularized", "regularised",
    ):
        assert normalize_collision(name) in ALL_COLLISIONS


def test_collision_is_authoritative_not_regime_label():
    assert collision_allowed_for_regime("BGK", "les")
    assert collision_allowed_for_regime("BGK", "dns")
    assert collision_allowed_for_regime("Smagorinsky", "dns")
    assert collision_allowed_for_regime("MRT", "les")
    assert normalize_turbulence_regime("les", "BGK") == "dns"
    assert normalize_turbulence_regime("dns", "Smagorinsky") == "les"
    assert normalize_collision("regularized") == "rlb"


def test_xml_collision_names():
    assert xml_collision_name("wale") == "WALE"
    assert xml_collision_name("smagorinskimrt") == "SmagorinskyMRT"
    assert xml_collision_name("regularised") == "RLB"
