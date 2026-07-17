from pathlib import Path
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from integrations.openlb.config_translator import OpenLBHITConfigTranslator
from schemas.openlb_hit import OpenLBHITConfig


def _config() -> OpenLBHITConfig:
    return OpenLBHITConfig(
        name="contract_test",
        domain={"resolution": (32, 32, 32), "size": (1.0, 1.0, 1.0)},
        scaling={
            "characteristic_length": 1.0,
            "characteristic_velocity": 0.1,
            "reynolds_number": 100.0,
            "target_mach": 0.05,
        },
        collision={"model": "BGK"},
        initial_condition={
            "type": "synthetic_spectrum",
            "wavenumber_min": 1,
            "wavenumber_max": 6,
            "target_urms": 0.1,
        },
        forcing={"type": "none"},
        runtime={"max_steps": 100, "output_interval": 10},
    )


def test_translator_writes_requested_effective_and_xml(tmp_path: Path):
    config = _config()
    paths = OpenLBHITConfigTranslator().write_case(config, tmp_path)
    assert set(paths) == {"requested", "effective", "xml"}
    assert all(path.is_file() for path in paths.values())

    root = ElementTree.parse(paths["xml"]).getroot()
    tags = {element.tag for element in root.iter()}
    required = {
        "Name",
        "Lx",
        "Ly",
        "Lz",
        "Nx",
        "Ny",
        "Nz",
        "Lattice",
        "Collision",
        "Tau",
        "Mach",
        "Reynolds",
        "InitialCondition",
        "ForcingType",
        "MaxSteps",
        "OutputInterval",
        "WriteVelocity",
    }
    assert required <= tags


def test_unknown_parameters_are_rejected():
    payload = _config().model_dump()
    payload["not_a_real_hit_parameter"] = 123
    with pytest.raises(ValidationError):
        OpenLBHITConfig.model_validate(payload)


def test_effective_configuration_preserves_model_identity():
    config = _config()
    effective = OpenLBHITConfigTranslator().effective_configuration(config)
    assert effective["collision"]["model"] == config.collision.model.value
    assert effective["forcing"]["type"] == config.forcing.type.value
    assert effective["requested_equals_effective"] is True
