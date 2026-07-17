"""Manifest-driven raw-field to validated KI-TURB analysis workflow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from schemas import DatasetManifest
from schemas.openlb_hit import OpenLBHITConfig
from schemas.unit_system import UnitSystem
from .flatness_from_fields import compute_flatness
from .hit_products_adapter import build_hit_analysis_products, save_products_and_register
from .isotropy_from_fields import compute_spectral_isotropy
from .pdfs_from_fields import compute_pdfs
from .readers import load_velocity_snapshots
from .real_stats_from_fields import compute_real_turbulence_stats
from .spectra_from_fields import average_stationary_spectra, compute_energy_spectrum
from .stationarity import detect_stationarity
from .structure_functions_from_fields import compute_structure_functions
from .uncertainty import summarize_uncertainty
from .writers import write_kiturb_outputs


def _resolve_unit_system(
    path: Optional[str],
    manifest: Optional[DatasetManifest],
) -> Tuple[float, Optional[float], int, Optional[OpenLBHITConfig], Optional[UnitSystem]]:
    """Return (dx, viscosity, max_steps, hit_config, unit_system). Prefer UnitSystem when present."""
    if manifest is not None and manifest.unit_system is not None:
        us = manifest.unit_system
        payload = dict(manifest.requested_config or manifest.case or {})
        typed = None
        candidate = payload if {"domain", "scaling", "collision"} <= set(payload) else payload.get("hit")
        if isinstance(candidate, dict) and {"domain", "scaling", "collision"} <= set(candidate):
            typed = OpenLBHITConfig.model_validate(candidate)
            max_steps = typed.runtime.max_steps
        else:
            max_steps = int((payload.get("runtime") or {}).get("max_steps") or 0)
        return float(us.dx or 0.0), us.viscosity, max_steps, typed, us

    payload: Dict[str, Any] = {}
    if path and Path(path).is_file():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload.get("units"), dict) and payload["units"].get("dx") is not None:
            try:
                us = UnitSystem.model_validate(payload["units"])
                hit = payload.get("hit")
                typed = OpenLBHITConfig.model_validate(hit) if isinstance(hit, dict) else None
                max_steps = typed.runtime.max_steps if typed else int((payload.get("runtime") or {}).get("max_steps") or 0)
                return float(us.dx or 0.0), us.viscosity, max_steps, typed, us
            except Exception:
                pass
    elif manifest is not None:
        payload = dict(manifest.requested_config or manifest.case or {})

    typed: Optional[OpenLBHITConfig] = None
    candidate = payload.get("hit") if isinstance(payload.get("hit"), dict) else payload
    if isinstance(candidate, dict) and {"domain", "scaling", "collision"} <= set(candidate):
        typed = OpenLBHITConfig.model_validate(candidate)
        derived = typed.derive_scaling()
        try:
            from integrations.openlb.unit_system import unit_system_from_openlb_hit

            us = unit_system_from_openlb_hit(typed)
        except Exception:
            us = None
        return derived.dx, derived.physical_viscosity, typed.runtime.max_steps, typed, us

    hit = payload.get("hit") or {}
    domain = hit.get("domain") or {}
    scaling = hit.get("scaling") or {}
    mesh = payload.get("mesh") or {}
    solver = payload.get("solver") or {}
    runtime = payload.get("runtime") or {}
    size = domain.get("size") or [1.0, 1.0, 1.0]
    resolution = domain.get("resolution") or mesh.get("resolution") or [1, 1, 1]
    dx = mesh.get("dx") or float(size[0]) / int(resolution[0])
    viscosity = scaling.get("physical_viscosity") or solver.get("viscosity")
    max_steps = int(runtime.get("max_steps") or (hit.get("runtime") or {}).get("max_steps") or 0)
    return float(dx), viscosity, max_steps, None, None


def postprocess_manifest(
    manifest: DatasetManifest,
    case_json_path: Optional[str] = None,
    *,
    processed_dir: Optional[str | Path] = None,
) -> DatasetManifest:
    manifest.require_complete()
    dx, viscosity, max_step, config, unit_system = _resolve_unit_system(case_json_path, manifest)
    if unit_system is not None:
        manifest.unit_system = unit_system
        manifest.units = {**manifest.units, **unit_system.field_labels()}
    snapshots = load_velocity_snapshots(manifest, dx=dx)
    target = Path(processed_dir) if processed_dir else Path(manifest.base_dir) / "processed"
    target.mkdir(parents=True, exist_ok=True)

    if not snapshots:
        manifest.status = "insufficient_data"
        manifest.postprocessing.update(
            status="insufficient_data",
            validation_status="insufficient_data",
            reason="no valid velocity snapshots",
            num_snapshots=0,
        )
        empty = build_hit_analysis_products(manifest, {
            "stationarity": {"stationary": False, "reason": "no valid velocity snapshots"},
            "validation_status": "insufficient_data",
            "warnings": ["no valid velocity snapshots"],
        })
        save_products_and_register(manifest, empty, target / "hit_analysis_products.json")
        (Path(manifest.base_dir) / "dataset_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        return manifest
    if viscosity is None or float(viscosity) <= 0:
        raise ValueError("positive viscosity is required for HIT post-processing")
    viscosity = float(viscosity)

    spectra = compute_energy_spectrum(snapshots, viscosity)
    stats = compute_real_turbulence_stats(snapshots, viscosity)
    products: Dict[str, Any] = {
        "spectra": spectra,
        "spectral_isotropy": compute_spectral_isotropy(snapshots),
        "real_stats": stats,
        "flatness": compute_flatness(snapshots),
        "structure_functions": compute_structure_functions(snapshots),
        "pdfs": compute_pdfs(snapshots, viscosity=viscosity),
    }
    # Prefer physical time only when every snapshot has it and the series is monotonic.
    # OpenLB manifests often mix a few physical_time stamps with step-only entries.
    physical = [snapshot.time for snapshot in snapshots]
    if all(t is not None for t in physical):
        time_values = np.asarray(physical, dtype=float)
    else:
        time_values = np.asarray([snapshot.step for snapshot in snapshots], dtype=float)
    if time_values.size >= 2 and np.any(np.diff(time_values) <= 0):
        time_values = np.asarray([snapshot.step for snapshot in snapshots], dtype=float)
    tke = np.array([row["TKE"] for row in stats], dtype=float)
    dissipation = np.array([row["eps_real"] for row in stats], dtype=float)
    stationarity = (
        detect_stationarity(
            time_values,
            tke,
            dissipation=dissipation,
            minimum_samples=min(8, max(3, len(snapshots))),
            cv_limit=config.acceptance.stationary_cv_limit if config else 0.05,
        )
        if len(snapshots) >= 3
        else None
    )
    selected = list(range(len(snapshots)))
    if stationarity and stationarity.stationary:
        selected = list(range(stationarity.start_index or 0, len(snapshots)))
    products["stationarity"] = stationarity.model_dump() if stationarity else {
        "stationary": False,
        "reason": "insufficient samples",
    }
    products["stationary_spectrum"] = average_stationary_spectra([spectra[index] for index in selected])
    products["uncertainty"] = [
        summarize_uncertainty([stats[index][metric] for index in selected], metric=metric).model_dump()
        for metric in ("TKE", "eps_real", "re_lambda", "kmax_eta")
        if selected
    ]
    physically_realizable = all(stats[index]["lumley_realizable"] for index in selected)
    divergence_values = [stats[index].get("divergence_rms") for index in selected if stats[index].get("divergence_rms") is not None]
    divergence_ok = bool(divergence_values) and max(divergence_values) <= (
        config.acceptance.maximum_divergence_rms if config else 1e-3
    )
    products["validation_status"] = "passed" if divergence_ok else "insufficient_data"
    products["diagnostics"] = {
        "lumley_realizable": physically_realizable,
        "stationary": bool(stationarity and stationarity.stationary),
        "divergence_ok": divergence_ok,
    }

    manifest.postprocessing.update(
        num_snapshots=len(snapshots),
        stationarity=products["stationarity"],
        uncertainty=products["uncertainty"],
        validation_status=products["validation_status"],
    )
    manifest = write_kiturb_outputs(manifest, products, target, max_step)
    canonical = build_hit_analysis_products(manifest, products, selected_indices=selected)
    save_products_and_register(manifest, canonical, target / "hit_analysis_products.json")
    manifest.status = "analysed" if canonical.spectra or canonical.reynolds_stress else "insufficient_data"
    manifest.postprocessing["status"] = manifest.status
    (Path(manifest.base_dir) / "dataset_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return manifest
