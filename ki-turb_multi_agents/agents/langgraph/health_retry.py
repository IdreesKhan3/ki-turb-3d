"""Recover from OpenLB simulation health rejections by retuning and re-running."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import WorkflowPlan, WorkflowStep
from .request_intent import ACTIVE_SIMULATION_JOB_ID

MAX_HEALTH_RETRIES = 3

_LES_SCHEMES = re.compile(
    r"smagorinsky|wale|krause|dynsmagorinsky|dynamicsmagorinsky",
    re.I,
)


def is_recoverable_health_rejection(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if "health rejection" in lowered:
        return True
    if re.search(r"\bstatus:\s*rejected\b", lowered):
        return True
    if "divergence" in lowered and ("exceed" in lowered or "rms" in lowered):
        return True
    if "mach" in lowered and ("exceed" in lowered or "too high" in lowered):
        return True
    return False


def build_params_from_job(project_root: Path, job_id: str) -> Dict[str, Any]:
    root = Path(project_root)
    job_dir = root / "simulations" / job_id
    requested = _read_json(job_dir / "requested_case.json")
    effective = _read_json(job_dir / "effective_openlb.json")
    params: Dict[str, Any] = {
        "backend": "openlb",
        "flow": "hit",
        "name": str(
            (effective or {}).get("case_name")
            or (requested or {}).get("name")
            or "HIT_retry"
        ),
    }
    if requested:
        domain = requested.get("domain") or {}
        scaling = requested.get("scaling") or {}
        collision = requested.get("collision") or {}
        ic = requested.get("initial_condition") or {}
        forcing = requested.get("forcing") or {}
        runtime = requested.get("runtime") or {}
        acceptance = requested.get("acceptance") or {}
        res = domain.get("resolution")
        if isinstance(res, list) and len(res) >= 3:
            params["resolution"] = [int(res[0]), int(res[1]), int(res[2])]
        if scaling.get("reynolds_number") is not None:
            params["reynolds_number"] = float(scaling["reynolds_number"])
        if scaling.get("physical_viscosity") is not None:
            params["viscosity"] = float(scaling["physical_viscosity"])
        if scaling.get("characteristic_velocity") is not None:
            params["char_velocity"] = float(scaling["characteristic_velocity"])
        if scaling.get("target_mach") is not None:
            params["mach_number"] = float(scaling["target_mach"])
        if scaling.get("relaxation_time") is not None:
            params["relaxation_time"] = float(scaling["relaxation_time"])
        if scaling.get("density") is not None:
            params["density"] = float(scaling["density"])
        model = collision.get("model")
        if model:
            params["scheme"] = str(model)
        if collision.get("smagorinsky_constant") is not None:
            params["smagorinsky_constant"] = float(collision["smagorinsky_constant"])
        if ic.get("target_urms") is not None:
            params["target_urms"] = float(ic["target_urms"])
        if ic.get("wavenumber_min") is not None:
            params["ic_wavenumber_min"] = int(ic["wavenumber_min"])
        if ic.get("wavenumber_max") is not None:
            params["ic_wavenumber_max"] = int(ic["wavenumber_max"])
        ftype = forcing.get("type")
        if ftype is not None:
            params["forcing_type"] = ftype if isinstance(ftype, str) else str(ftype)
            if str(params["forcing_type"]).lower() in {"none", "forcingtype.none"}:
                params["forcing_type"] = "none"
                params["hit_mode"] = "decaying"
            else:
                params["hit_mode"] = params.get("hit_mode") or "forced"
        if forcing.get("wavenumber_min") is not None:
            params["forcing_wavenumber_min"] = int(forcing["wavenumber_min"])
        if forcing.get("wavenumber_max") is not None:
            params["forcing_wavenumber_max"] = int(forcing["wavenumber_max"])
        if forcing.get("amplitude") is not None:
            params["forcing_amplitude"] = float(forcing["amplitude"])
        if runtime.get("max_steps") is not None:
            params["max_steps"] = int(runtime["max_steps"])
        if runtime.get("output_interval") is not None:
            params["output_interval"] = int(runtime["output_interval"])
        if acceptance.get("turbulence_regime"):
            params["turbulence_regime"] = str(acceptance["turbulence_regime"])
    if effective:
        if effective.get("collision") and "scheme" not in params:
            params["scheme"] = str(effective["collision"])
        if effective.get("characteristic_velocity") is not None:
            params["char_velocity"] = float(effective["characteristic_velocity"])
        if effective.get("mach") is not None:
            params["mach_number"] = float(effective["mach"])
        if effective.get("reynolds") is not None:
            params["reynolds_number"] = float(effective["reynolds"])
        if effective.get("tau") is not None and "relaxation_time" not in params:
            params["relaxation_time"] = float(effective["tau"])
        if str(effective.get("forcing") or "").lower() == "none":
            params["hit_mode"] = "decaying"
            params["forcing_type"] = "none"
    if "hit_mode" not in params:
        params["hit_mode"] = "decaying" if str(params.get("forcing_type") or "").lower() == "none" else "forced"
    return params


def load_job_measured(project_root: Path, job_id: str) -> Dict[str, Any]:
    job = _read_json(Path(project_root) / "simulations" / job_id / "job.json") or {}
    measured = job.get("measured")
    return dict(measured) if isinstance(measured, dict) else {}


def retune_build_params(
    params: Dict[str, Any],
    rejection_message: str,
    *,
    attempt: int,
    measured: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(params or {})
    out["backend"] = "openlb"
    out["flow"] = "hit"
    measured = measured or {}
    msg = (rejection_message or "").lower()

    u = float(out.get("char_velocity") or out.get("target_urms") or 0.1)
    mach_max = measured.get("mach_max")
    try:
        mach_max_f = float(mach_max) if mach_max is not None else None
    except (TypeError, ValueError):
        mach_max_f = None

    if mach_max_f is not None and mach_max_f > 0.08 and u > 0:
        u = u * min(0.05 / mach_max_f, 0.5)
    elif u >= 0.5:
        u = 0.1
    else:
        u = u * 0.5
    u = float(max(0.05, min(u, 0.1)))
    if attempt >= 2:
        u = 0.05

    out["char_velocity"] = u
    out["target_urms"] = u
    out["mach_number"] = min(float(out.get("mach_number") or 0.05), 0.05)

    re_val = out.get("reynolds_number")
    try:
        re_f = float(re_val) if re_val is not None else None
    except (TypeError, ValueError):
        re_f = None
    if re_f is not None and re_f > 200:
        out["reynolds_number"] = 100.0

    scheme = str(out.get("scheme") or "")
    if _LES_SCHEMES.search(scheme) or str(out.get("turbulence_regime") or "").lower() == "les":
        out["scheme"] = "MRT" if str(out.get("hit_mode") or "") == "forced" else "BGK"
        out["turbulence_regime"] = "dns"
    elif "divergence" in msg and attempt >= 1 and not scheme:
        out["scheme"] = "BGK"
        out["turbulence_regime"] = "dns"

    if out.get("relaxation_time") is not None:
        try:
            tau = float(out["relaxation_time"])
            if tau < 0.505:
                out["relaxation_time"] = 0.51
        except (TypeError, ValueError):
            pass

    base = str(out.get("name") or "HIT").split("_retry")[0]
    out["name"] = f"{base}_retry{attempt + 1}"
    out["health_retry_attempt"] = attempt + 1
    out["health_retry_from"] = rejection_message[:240]
    return out


def lifecycle_retry_steps(build_args: Dict[str, Any]) -> List[WorkflowStep]:
    job_ref = {"job_id": ACTIVE_SIMULATION_JOB_ID}
    clean_args = {
        key: value
        for key, value in dict(build_args or {}).items()
        if key not in {"health_retry_attempt", "health_retry_from", "validate", "stop_after"}
    }
    summary = {
        k: clean_args.get(k)
        for k in (
            "name", "hit_mode", "scheme", "char_velocity", "target_urms",
            "mach_number", "reynolds_number", "turbulence_regime", "resolution",
        )
        if k in clean_args
    }
    return [
        WorkflowStep(
            role="simulation",
            instruction=(
                "Rebuild after simulation health rejection with retuned lattice-stable "
                f"parameters: {json.dumps(summary, default=str)}"
            ),
            tool="build_simulation_case",
            tool_args=clean_args,
        ),
        WorkflowStep(
            role="simulation",
            instruction="Compile the retuned OpenLB solver case.",
            tool="compile_simulation",
            tool_args=dict(job_ref),
        ),
        WorkflowStep(
            role="simulation",
            instruction="Launch the retuned OpenLB simulation job.",
            tool="start_simulation",
            tool_args=dict(job_ref),
        ),
        WorkflowStep(
            role="simulation",
            instruction="Monitor the retuned simulation until it completes or fails.",
            tool="supervise_simulation",
            tool_args=dict(job_ref),
        ),
    ]


def splice_health_retry_plan(
    plan: WorkflowPlan,
    supervise_index: int,
    build_args: Dict[str, Any],
) -> Tuple[WorkflowPlan, int]:
    retry = lifecycle_retry_steps(build_args)
    steps = list(plan.steps[:supervise_index]) + retry + list(plan.steps[supervise_index + 1 :])
    rationale = (plan.rationale or "").rstrip()
    attempt = int(build_args.get("health_retry_attempt") or 1)
    rationale = f"{rationale} | health_retry#{attempt}".strip(" |")
    return WorkflowPlan(steps=steps, rationale=rationale, kind=plan.kind), supervise_index


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


__all__ = [
    "MAX_HEALTH_RETRIES",
    "is_recoverable_health_rejection",
    "build_params_from_job",
    "load_job_measured",
    "retune_build_params",
    "lifecycle_retry_steps",
    "splice_health_retry_plan",
]
