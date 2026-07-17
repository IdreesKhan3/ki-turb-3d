import pytest

from integrations.openlb.capability_validator import (
    OpenLBHITCapabilityValidator,
    UnsupportedOpenLBCapability,
)
from integrations.openlb.config_translator import OpenLBHITConfigTranslator
from schemas.openlb_hit import OpenLBHITConfig


def _config(collision: str, forcing: str = "none") -> OpenLBHITConfig:
    forcing_config = {"type": forcing}
    if forcing != "none":
        forcing_config.update(
            {"wavenumber_min": 1, "wavenumber_max": 2, "amplitude": 0.01}
        )
    return OpenLBHITConfig(
        domain={"resolution": (32, 32, 32), "size": (1.0, 1.0, 1.0)},
        scaling={
            "characteristic_length": 1.0,
            "characteristic_velocity": 0.1,
            "reynolds_number": 100.0,
        },
        collision={"model": collision},
        initial_condition={
            "type": "synthetic_spectrum",
            "wavenumber_min": 1,
            "wavenumber_max": 6,
        },
        forcing=forcing_config,
    )


def test_supported_collision_is_not_substituted():
    config = _config("BGK", "none")
    effective = OpenLBHITConfigTranslator().effective_configuration(config)
    assert effective["collision"]["model"] == "BGK"
    assert effective["forcing"]["type"] == "none"


def test_unsupported_collision_forcing_combination_is_rejected():
    config = _config("RLB", "spectral_random")
    with pytest.raises(UnsupportedOpenLBCapability):
        OpenLBHITCapabilityValidator().assert_supported(config)


def test_forced_mrt_uses_exact_supported_identity():
    config = _config("MRT", "spectral_random")
    effective = OpenLBHITConfigTranslator().effective_configuration(config)
    assert effective["collision"]["model"] == "MRT"
    assert effective["forcing"]["type"] == "spectral_random"


def test_unsupported_model_is_rejected_instead_of_falling_back():
    config = _config("SmagorinskyMRT", "none")
    with pytest.raises(UnsupportedOpenLBCapability):
        OpenLBHITConfigTranslator().effective_configuration(config)
