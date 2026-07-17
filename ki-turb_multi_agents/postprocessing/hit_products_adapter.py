"""Bridge computed HIT dictionaries and manifests into canonical analysis products."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from schemas import DatasetFile, DatasetManifest
from schemas.hit_analysis_products import (
    EnergyBalanceAssessment,
    EnergySpectrumProduct,
    FlatnessProduct,
    HITAnalysisProducts,
    PDFProduct,
    ProductProvenance,
    ResolutionAssessment,
    ReynoldsStressProduct,
    SpectralIsotropyProduct,
    StationarityAssessment,
    StatisticalSummary,
    StructureFunctionProduct,
    TimeHistoryProduct,
)
from .energy_balance import energy_balance_history


def _tolist(value: Any) -> List[float]:
    return np.asarray(value, dtype=float).tolist()


def _source_files(manifest: DatasetManifest) -> List[str]:
    return [item.path for item in manifest.files if item.kind == "velocity_field" and item.complete]


def _diagnostics(manifest: DatasetManifest) -> Dict[int, Dict[str, Any]]:
    base = Path(manifest.base_dir)
    records: Dict[int, Dict[str, Any]] = {}
    for item in manifest.files:
        if item.kind != "diagnostics" or not item.complete:
            continue
        path = base / item.path
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("step"), (int, float)):
                records[int(payload["step"])] = payload
    return records


def build_hit_analysis_products(
    manifest: DatasetManifest,
    computed: Dict[str, Any],
    *,
    selected_indices: Optional[Sequence[int]] = None,
) -> HITAnalysisProducts:
    """Create the canonical solver-neutral product bundle from computed records."""
    run_id = manifest.run_id or manifest.source_job_id or manifest.source_simulation
    source_files = _source_files(manifest)
    stats = list(computed.get("real_stats") or [])
    spectra = list(computed.get("spectra") or [])
    isotropy = list(computed.get("spectral_isotropy") or [])
    diagnostics = _diagnostics(manifest)
    indices = list(selected_indices or range(len(stats)))
    source_steps = [int(row.get("iter", 0)) for row in stats]
    provenance = ProductProvenance(
        run_id=run_id,
        source_steps=source_steps,
        source_files=source_files,
        algorithm="kiturb_periodic_hit_pipeline",
        algorithm_version="3",
        normalization="Parseval shell-summed energy spectrum",
        units=dict(manifest.units),
    )

    spectrum_products: List[EnergySpectrumProduct] = []
    for row in spectra:
        spectrum_products.append(EnergySpectrumProduct(
            step=int(row["step"]),
            time=row.get("time"),
            wavenumber=_tolist(row["k"]),
            energy=_tolist(row["E"]),
            compensated_energy=_tolist(row["compensated"]) if row.get("compensated") is not None else None,
            k_eta=_tolist(row["k_eta"]) if row.get("k_eta") is not None else None,
            inertial_slope=row.get("inertial_slope"),
            provenance=provenance.model_copy(deep=True),
        ))

    isotropy_products: List[SpectralIsotropyProduct] = []
    for row in isotropy:
        deviation = np.asarray(row.get("component_deviation", []), dtype=float)
        finite = deviation[np.isfinite(deviation)]
        isotropy_products.append(SpectralIsotropyProduct(
            step=int(row["step"]),
            time=row.get("time"),
            wavenumber=_tolist(row["k"]),
            e11=_tolist(row["E11"]),
            e22=_tolist(row["E22"]),
            e33=_tolist(row["E33"]),
            maximum_component_deviation=float(np.max(finite)) if finite.size else None,
            provenance=provenance.model_copy(deep=True),
        ))

    stress_products = [ReynoldsStressProduct(
        step=int(row["iter"]), time=row.get("time"),
        r11=float(row["R11"]), r22=float(row["R22"]), r33=float(row["R33"]),
        r12=float(row["R12"]), r13=float(row["R13"]), r23=float(row["R23"]),
        b11=float(row["b11"]), b22=float(row["b22"]), b33=float(row["b33"]),
        b12=float(row["b12"]), b13=float(row["b13"]), b23=float(row["b23"]),
        invariant_ii=float(row["II_b"]), invariant_iii=float(row["III_b"]),
        provenance=provenance.model_copy(deep=True),
    ) for row in stats]

    structure_products = [StructureFunctionProduct(
        step=int(row["step"]), time=row.get("time"), separation=_tolist(row["r"]),
        orders=[int(v) for v in row["orders"]],
        longitudinal={str(k): _tolist(v) for k, v in row["longitudinal"].items()},
        transverse={str(k): _tolist(v) for k, v in row["transverse"].items()},
        signed_longitudinal_third=_tolist(row["signed_longitudinal_third"]),
        provenance=provenance.model_copy(deep=True),
    ) for row in (computed.get("structure_functions") or [])]

    pdf_products: List[PDFProduct] = []
    for row in computed.get("pdfs") or []:
        for variable, x_key, y_key in (
            ("normalized_velocity", "velocity_bin", "velocity_pdf"),
            ("normalized_velocity_gradient", "gradient_bin", "gradient_pdf"),
            ("normalized_dissipation", "dissipation_bin", "dissipation_pdf"),
        ):
            if x_key in row and y_key in row:
                pdf_products.append(PDFProduct(
                    step=int(row["step"]), time=row.get("time"), variable=variable,
                    bin_center=_tolist(row[x_key]), density=_tolist(row[y_key]),
                    provenance=provenance.model_copy(deep=True),
                ))

    flatness_products = [FlatnessProduct(
        step=int(row["step"]), time=row.get("time"), separation=_tolist(row["r"]),
        flatness=_tolist(row["flatness"]), provenance=provenance.model_copy(deep=True),
    ) for row in (computed.get("flatness") or [])]

    time_history: Optional[TimeHistoryProduct] = None
    energy_balance: Optional[EnergyBalanceAssessment] = None
    if stats:
        steps = [int(row["iter"]) for row in stats]
        times = [
            float(row["time"] if row.get("time") is not None else diagnostics.get(step, {}).get("physical_time", step))
            for row, step in zip(stats, steps)
        ]
        forcing_power = [float(diagnostics.get(step, {}).get("forcing_power", 0.0)) for step in steps]
        dissipation = [float(row["eps_real"]) for row in stats]
        time_history = TimeHistoryProduct(
            time=times,
            step=steps,
            tke=[float(row["TKE"]) for row in stats],
            dissipation=dissipation,
            forcing_power=forcing_power,
            re_lambda=[float(row["re_lambda"]) for row in stats],
            mach_max=[float(diagnostics.get(step, {}).get("mach_max", 0.0)) for step in steps],
            density_min=[float(diagnostics.get(step, {}).get("density_min", 1.0)) for step in steps],
            density_max=[float(diagnostics.get(step, {}).get("density_max", 1.0)) for step in steps],
            divergence_rms=[float(row["divergence_rms"]) for row in stats],
            kmax_eta=[float(row["kmax_eta"]) for row in stats],
            provenance=provenance.model_copy(deep=True),
        )
        if len(times) >= 2 and np.all(np.diff(times) > 0):
            spectral_by_step = {int(row["step"]): row.get("epsilon_spectral") for row in spectra}
            spectral_values = [spectral_by_step.get(step, np.nan) for step in steps]
            energy_balance = energy_balance_history(
                times,
                time_history.tke,
                dissipation,
                forcing=forcing_power,
                dissipation_spectral_values=spectral_values,
                provenance=provenance.model_copy(deep=True),
            )

    stationarity_payload = computed.get("stationarity") or {}
    stationarity = StationarityAssessment.model_validate(stationarity_payload) if stationarity_payload else None
    kmax_eta_values = [float(stats[index]["kmax_eta"]) for index in indices if 0 <= index < len(stats)]
    resolution = ResolutionAssessment(
        passed=True,
        kmax_eta_mean=float(np.mean(kmax_eta_values)) if kmax_eta_values else None,
        kmax_eta_min=float(np.min(kmax_eta_values)) if kmax_eta_values else None,
        samples=len(kmax_eta_values),
        reason="kmax*eta diagnostic computed from periodic physical-space dissipation (not an acceptance gate)",
    )
    uncertainty = [StatisticalSummary.model_validate(item) for item in computed.get("uncertainty") or []]
    warnings = list(computed.get("warnings") or [])
    status = str(computed.get("validation_status") or "unvalidated")
    return HITAnalysisProducts(
        run_id=run_id,
        time_history=time_history,
        spectra=spectrum_products,
        spectral_isotropy=isotropy_products,
        reynolds_stress=stress_products,
        structure_functions=structure_products,
        pdfs=pdf_products,
        flatness=flatness_products,
        stationarity=stationarity,
        resolution=resolution,
        energy_balance=energy_balance,
        uncertainty=uncertainty,
        validation_status=status.upper(),
        warnings=warnings,
        unit_system=manifest.unit_system,
        metadata={"stationary_indices": indices, "source_manifest_id": manifest.manifest_id},
    )


def save_products_and_register(
    manifest: DatasetManifest,
    products: HITAnalysisProducts,
    path: str | Path,
) -> Path:
    destination = products.save(path)
    import hashlib
    checksum = "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()
    try:
        manifest_path_value = str(destination.relative_to(Path(manifest.base_dir)))
    except ValueError:
        manifest_path_value = str(destination)
    manifest.add_file(DatasetFile(
        path=manifest_path_value,
        kind="analysis_products",
        format="json",
        size_bytes=destination.stat().st_size,
        checksum=checksum,
        complete=True,
        source_steps=list(products.time_history.step if products.time_history else []),
        metadata={"validation_status": products.validation_status},
    ))
    manifest.postprocessing["analysis_products_path"] = str(destination)
    return destination


def load_products_from_manifest(manifest: DatasetManifest) -> HITAnalysisProducts:
    candidates = [item for item in manifest.files if item.kind == "analysis_products" and item.complete]
    if not candidates:
        raw = manifest.postprocessing.get("analysis_products_path")
        if not raw:
            raise FileNotFoundError("manifest does not reference HITAnalysisProducts")
        path = Path(raw)
    else:
        path = Path(manifest.base_dir) / candidates[-1].path
    return HITAnalysisProducts.load(path)


__all__ = ["build_hit_analysis_products", "save_products_and_register", "load_products_from_manifest"]
