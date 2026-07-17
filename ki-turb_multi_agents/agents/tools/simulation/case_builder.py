"""Tool for creating a CFD simulation case.

Builds a validated :class:`~schemas.cfd_case.CFDCase` from tool arguments, asks
the selected backend to write its input files, and persists a durable job record
that the run and fetch tools reference by ``job_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from schemas import CFDCase
import integrations
from integrations.base import new_job_id
from case_library import registry as case_registry
from physics_constraints import validate_case
from schemas.openlb_hit import OpenLBHITConfig

from . import _store
from schemas.cfd_case import normalize_hit_mode

CASE_BUILDER_TOOL_NAMES = frozenset({"build_simulation_case"})

_HIT_PARAMS = {
    "turbulence_regime": {
        "type": "string",
        "description": "dns (direct) or les (subgrid models: Smagorinsky, WALE, ...).",
    },
    "hit_mode": {
        "type": "string",
        "description": "OpenLB HIT regime: decaying (DHIT) or forced (FHIT).",
    },
    "initial_condition": {"type": "string", "description": "e.g. divergence_free_random"},
    "ic_wavenumber_min": {"type": "integer"},
    "ic_wavenumber_max": {"type": "integer"},
    "ic_seed": {"type": "integer"},
    "ic_spectrum_exponent": {"type": "number"},
    "forcing_type": {
        "type": "string",
        "description": "Alias: forcing_scheme. none | linear | spectral_low_k | abc | constant | ornstein_uhlenbeck",
    },
    "forcing_scheme": {"type": "string", "description": "Same as forcing_type."},
    "forcing_pattern": {
        "type": "string",
        "description": "random_phase | fixed_phase | sine | cosine | ou_process | abc_time",
    },
    "forcing_wavenumber_min": {"type": "integer"},
    "forcing_wavenumber_max": {"type": "integer"},
    "forcing_amplitude": {"type": "number"},
    "forcing_update_interval": {"type": "integer"},
    "target_urms": {"type": "number"},
    "target_re_lambda": {"type": "number"},
    "statistically_stationary": {"type": "boolean"},
    "box_length": {"type": "number"},
    "smagorinsky_constant": {"type": "number"},
    "trt_magic_parameter": {"type": "number", "description": "TRT magic parameter (default 0.25)."},
    "density": {"type": "number"},
    "char_velocity": {"type": "number"},
    "write_pressure": {"type": "boolean"},
    "sample_start_step": {"type": "integer"},
}


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "build_simulation_case",
            "description": (
                "Create a CFD simulation case for a backend (openlb, palabos, ansys) and "
                "write its input files. For OpenLB HIT (DHIT/FHIT), pass any combination "
                "of physics parameters below or supply a full ``case`` CFDCase dict. "
                "Returns a job_id used by the other simulation tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "backend": {"type": "string", "description": "openlb | palabos | ansys"},
                    "name": {"type": "string", "description": "Case name"},
                    "case": {
                        "type": "object",
                        "description": "Full CFDCase dict; overrides convenience fields.",
                    },
                    "resolution": {
                        "type": "array",
                        "description": "Mesh resolution [nx, ny, nz]",
                        "items": {"type": "integer"},
                    },
                    "flow": {"type": "string", "description": "hit | channel | cylinder | custom"},
                    "reynolds_number": {"type": "number"},
                    "viscosity": {"type": "number"},
                    "max_steps": {"type": "integer"},
                    "output_interval": {"type": "integer"},
                    "scheme": {
                        "type": "string",
                        "description": (
                            "OpenLB collision: BGK, DNS, RLB, MRT, TRT (DNS); "
                            "Smagorinsky, WALE, ConsistentStrainSmagorinsky, ShearSmagorinsky, "
                            "Krause, SmagorinskyMRT, DynSmagorinsky (LES)"
                        ),
                    },
                    "mach_number": {"type": "number"},
                    "relaxation_time": {"type": "number"},
                    **_HIT_PARAMS,
                },
                "required": ["backend", "name"],
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    if name != "build_simulation_case":
        return f"Error: Unknown case-builder tool '{name}'"

    backend_name = str(args.get("backend", "")).strip().lower()
    if backend_name not in integrations.available_backends():
        return (
            f"Error: unknown backend '{args.get('backend')}'. "
            f"Available: {', '.join(integrations.available_backends())}"
        )

    try:
        case = _build_case(backend_name, args)
    except (ValidationError, ValueError) as exc:
        return f"Error: invalid case configuration: {exc}"

    validation = validate_case(case)
    if not validation.passed:
        errors = "\n".join(
            f"- {c.name}: {c.message} value={c.value} limit={c.limit}"
            for c in validation.errors()
        )
        return "Error: physics validation failed.\n" + errors

    backend = integrations.get_backend(backend_name)
    job_id = new_job_id()
    target_dir = _store.job_dir(project_root, job_id)
    job = backend.prepare_case(case, target_dir, job_id=job_id)

    validation_path = Path(job.paths.case_dir) / "validation_report.json"
    validation_path.write_text(validation.model_dump_json(indent=2), encoding="utf-8")
    job.metadata["validation_report_path"] = str(validation_path)
    _store.save_job(project_root, job)

    warnings = [c for c in validation.checks if c.severity == "warning" and not c.passed]
    warning_line = f"\nwarnings: {len(warnings)}" if warnings else ""
    hit_mode = case.flow.hit_mode.value if case.flow.hit_mode else "inferred"
    return (
        f"Prepared and validated {backend_name} case '{case.name}'.\n"
        f"job_id: {job.job_id}\n"
        f"hit_mode: {hit_mode}\n"
        f"case_dir: {job.paths.case_dir}\n"
        f"output_dir: {job.paths.output_dir}\n"
        f"status: {job.status.value}"
        f"{warning_line}\n"
        f"Next: compile_simulation (if the backend compiles) then start_simulation "
        f"with job_id={job.job_id}."
    )


def _apply_calibrated_hit(case: CFDCase, calibrated_hit: OpenLBHITConfig) -> CFDCase:
    case.hit = calibrated_hit
    if case.flow is not None:
        case.flow.target_urms = calibrated_hit.initial_condition.target_urms
        case.flow.forcing_amplitude = calibrated_hit.forcing.amplitude
        case.flow.ic_wavenumber_min = calibrated_hit.initial_condition.wavenumber_min
        case.flow.ic_wavenumber_max = calibrated_hit.initial_condition.wavenumber_max
    if case.solver is not None:
        case.solver.reynolds_number = calibrated_hit.scaling.reynolds_number
        case.solver.viscosity = calibrated_hit.scaling.physical_viscosity
        extra = dict(case.solver.extra or {})
        extra["mach_number"] = calibrated_hit.scaling.target_mach
        extra["char_velocity"] = calibrated_hit.scaling.characteristic_velocity
        case.solver.extra = extra
    return case


def _build_case(backend_name: str, args: Dict[str, Any]) -> CFDCase:
    """Build a case from an explicit dict, a curated factory, or convenience fields."""
    from agents.physics_constraint_agent import PhysicsConstraintAgent

    backend = backend_name.strip().lower()

    if isinstance(args.get("case"), dict):
        case_data = dict(args["case"])
        case_data.setdefault("name", args.get("name", "case"))
        case = CFDCase.model_validate(case_data)
        # LLM/manifest case dumps often keep schema default div limit (1e-6) — calibrate.
        if case.flow and str(getattr(case.flow.kind, "value", case.flow.kind)).lower() == "hit" and backend == "openlb":
            try:
                calibrated, _decision = PhysicsConstraintAgent().calibrate(
                    OpenLBHITConfig.from_cfd_case(case)
                )
                case = _apply_calibrated_hit(case, calibrated)
            except Exception:
                # Leave the case as-is; validate_case reports physics errors.
                pass
        return case

    flow = str(args.get("flow", "")).strip().lower()
    calibrated_hit: OpenLBHITConfig | None = None
    if flow == "hit" and backend == "openlb":
        from agents.langgraph.openlb_hit_build import infer_locked_build_fields, normalize_build_args
        from agents.physics_constraint_agent import _OPENLB_HIT_CONFIG_KEY

        args = normalize_build_args(dict(args))
        args = PhysicsConstraintAgent().calibrate_build_args(
            args,
            locked=infer_locked_build_fields(args),
        )
        payload = args.pop(_OPENLB_HIT_CONFIG_KEY, None)
        if payload:
            calibrated_hit = OpenLBHITConfig.model_validate(payload)

    if flow and case_registry.has_factory(flow, backend):
        case = case_registry.make_case(flow, backend, **_factory_kwargs(args))
    else:
        case = _build_case_from_fields(args)

    if calibrated_hit is not None:
        case = _apply_calibrated_hit(case, calibrated_hit)

    return case


def _factory_kwargs(args: Dict[str, Any]) -> Dict[str, Any]:
    from agents.physics_constraint_agent import _OPENLB_HIT_CONFIG_KEY

    keys = (
        "name", "hit_mode", "turbulence_regime", "reynolds_number", "viscosity", "max_steps",
        "output_interval", "mach_number", "relaxation_time", "scheme", "smagorinsky_constant",
        "trt_magic_parameter", "density", "char_velocity", "box_length", "initial_condition",
        "ic_wavenumber_min", "ic_wavenumber_max", "ic_seed", "ic_spectrum_exponent",
        "forcing_type", "forcing_scheme", "forcing_pattern", "forcing_wavenumber_min",
        "forcing_wavenumber_max", "forcing_amplitude", "forcing_update_interval",
        "target_urms", "target_re_lambda", "statistically_stationary",
        "sample_start_step", "write_pressure",
    )
    kwargs = {k: args[k] for k in keys if args.get(k) is not None and k != "forcing_scheme"}
    kwargs.pop(_OPENLB_HIT_CONFIG_KEY, None)
    if args.get("hit_mode") is not None:
        kwargs["hit_mode"] = normalize_hit_mode(args["hit_mode"])
    if args.get("forcing_scheme") and not args.get("forcing_type"):
        kwargs["forcing_type"] = args["forcing_scheme"]
    if args.get("resolution"):
        kwargs["resolution"] = tuple(args["resolution"])
    return kwargs


def _build_case_from_fields(args: Dict[str, Any]) -> CFDCase:
    payload: Dict[str, Any] = {"name": args.get("name", "case")}

    if args.get("resolution"):
        payload["mesh"] = {"resolution": tuple(args["resolution"])}

    if args.get("box_length") is not None:
        side = float(args["box_length"])
        payload["geometry"] = {"kind": "box", "size": [side, side, side]}

    solver: Dict[str, Any] = {}
    for key in ("reynolds_number", "viscosity", "scheme"):
        if args.get(key) is not None:
            solver[key] = args[key]
    extra = {
        k: args[k]
        for k in (
            "mach_number", "relaxation_time", "smagorinsky_constant", "trt_magic_parameter",
            "density", "char_velocity", "turbulence_regime",
        )
        if args.get(k) is not None
    }
    if extra:
        solver["extra"] = extra
    if solver:
        payload["solver"] = solver

    runtime = {k: args[k] for k in ("max_steps", "output_interval") if args.get(k) is not None}
    if runtime:
        payload["runtime"] = runtime

    flow: Dict[str, Any] = {}
    if args.get("flow"):
        flow["kind"] = str(args["flow"]).lower()
    flow_keys = (
        "hit_mode", "initial_condition", "ic_wavenumber_min", "ic_wavenumber_max",
        "ic_seed", "ic_spectrum_exponent", "forcing_type", "forcing_pattern",
        "forcing_wavenumber_min", "forcing_wavenumber_max", "forcing_amplitude",
        "forcing_update_interval", "target_urms", "target_re_lambda", "statistically_stationary",
    )
    for key in flow_keys:
        if args.get(key) is not None:
            flow[key] = normalize_hit_mode(args[key]) if key == "hit_mode" else args[key]
    if args.get("forcing_scheme") and "forcing_type" not in flow:
        flow["forcing_type"] = args["forcing_scheme"]
    if flow:
        payload["flow"] = flow

    outputs: Dict[str, Any] = {}
    for key in ("sample_start_step", "write_pressure"):
        if args.get(key) is not None:
            outputs[key] = args[key]
    if outputs:
        payload["outputs"] = outputs

    return CFDCase.model_validate(payload)
