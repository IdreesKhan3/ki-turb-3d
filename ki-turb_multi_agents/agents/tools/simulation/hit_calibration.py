"""Convert and calibrate OpenLB HIT build arguments via PhysicsConstraintAgent."""
from __future__ import annotations

from typing import Any

from case_library import make_case
from schemas.openlb_hit import OpenLBHITConfig

_COLLISION_TO_SCHEME = {
    "SmagorinskyBGK": "Smagorinsky",
    "SmagorinskyMRT": "SmagorinskyMRT",
    "DynamicSmagorinsky": "DynamicSmagorinsky",
    "ConsistentStrainSmagorinsky": "ConsistentStrainSmagorinsky",
    "ShearSmagorinsky": "ShearSmagorinsky",
    "Krause": "Krause",
    "BGK": "BGK",
    "WALE": "WALE",
    "MRT": "MRT",
    "TRT": "TRT",
    "RLB": "RLB",
}

_FORCING_TO_BUILD = {
    "spectral_random": "spectral_low_k",
    "ornstein_uhlenbeck": "ornstein_uhlenbeck",
    "constant_energy_input": "linear",
    "none": "none",
}

_FACTORY_KEYS = (
    "name", "hit_mode", "turbulence_regime", "reynolds_number", "viscosity", "max_steps",
    "output_interval", "mach_number", "relaxation_time", "scheme", "smagorinsky_constant",
    "trt_magic_parameter", "density", "char_velocity", "box_length", "initial_condition",
    "ic_wavenumber_min", "ic_wavenumber_max", "ic_seed", "ic_spectrum_exponent",
    "forcing_type", "forcing_scheme", "forcing_pattern", "forcing_wavenumber_min",
    "forcing_wavenumber_max", "forcing_amplitude", "target_urms", "forcing_update_interval",
    "target_urms", "target_re_lambda", "statistically_stationary",
    "sample_start_step", "write_pressure",
)


def build_args_to_openlb_config(args: dict[str, Any]) -> OpenLBHITConfig:
    """Build a typed OpenLB config from build_simulation_case kwargs."""
    flow = str(args.get("flow", "hit")).lower()
    backend = str(args.get("backend", "openlb")).lower()
    kwargs = {key: args[key] for key in _FACTORY_KEYS if args.get(key) is not None and key != "forcing_scheme"}
    if args.get("forcing_scheme") and not args.get("forcing_type"):
        kwargs["forcing_type"] = args["forcing_scheme"]
    if args.get("resolution"):
        kwargs["resolution"] = tuple(args["resolution"])
    if kwargs.get("char_velocity") is not None and kwargs.get("reynolds_number") is not None:
        kwargs.pop("viscosity", None)
    case = make_case(flow, backend, **kwargs)
    return OpenLBHITConfig.from_cfd_case(case)


def openlb_config_to_build_args(config: OpenLBHITConfig, seed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map a calibrated typed config back to build_simulation_case kwargs."""
    seed = dict(seed or {})
    scaling = config.scaling
    forcing = config.forcing
    initial = config.initial_condition
    collision = config.collision.model.value

    forcing_type = forcing.type.value if hasattr(forcing.type, "value") else str(forcing.type)
    result: dict[str, Any] = {
        "backend": "openlb",
        "flow": "hit",
        "name": config.name,
        "resolution": list(config.domain.resolution),
        "scheme": _COLLISION_TO_SCHEME.get(collision, collision),
        "reynolds_number": scaling.reynolds_number,
        "mach_number": scaling.target_mach,
        "char_velocity": scaling.characteristic_velocity,
        "relaxation_time": scaling.relaxation_time,
        "max_steps": config.runtime.max_steps,
        "output_interval": config.runtime.output_interval,
        "ic_wavenumber_min": initial.wavenumber_min,
        "ic_wavenumber_max": initial.wavenumber_max,
        "forcing_wavenumber_min": forcing.wavenumber_min,
        "forcing_wavenumber_max": forcing.wavenumber_max,
        "forcing_amplitude": forcing.amplitude,
        "forcing_type": _FORCING_TO_BUILD.get(forcing_type, forcing_type),
        "target_urms": initial.target_urms,
        "smagorinsky_constant": config.collision.smagorinsky_constant,
        "turbulence_regime": (config.metadata or {}).get("turbulence_regime"),
    }
    if seed.get("hit_mode"):
        result["hit_mode"] = seed["hit_mode"]
    merged = dict(seed)
    merged.update({key: value for key, value in result.items() if value is not None})
    return merged


__all__ = ["build_args_to_openlb_config", "openlb_config_to_build_args"]
