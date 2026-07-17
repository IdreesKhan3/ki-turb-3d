"""Scientific acceptance checks for processed HIT products.

Only velocity divergence is used as a hard acceptance gate.  All other
quantities (Re, eta, kmax, Mach, energy balance, isotropy, stationarity, …)
are recorded as diagnostics so agents can report them on request without
blocking the workflow.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from schemas import ConstraintCheck, DatasetManifest, ValidationReport
from schemas.hit_analysis_products import HITAnalysisProducts
from schemas.openlb_hit import HITAcceptanceThresholds, OpenLBHITConfig


class HITValidationAgent:
    def __init__(self, thresholds: Optional[HITAcceptanceThresholds] = None) -> None:
        self.thresholds = thresholds or HITAcceptanceThresholds()

    def validate(
        self,
        products: HITAnalysisProducts,
        *,
        config: Optional[OpenLBHITConfig] = None,
        manifest: Optional[DatasetManifest] = None,
    ) -> ValidationReport:
        thresholds = config.acceptance if config is not None else self.thresholds
        report = ValidationReport()
        diagnostics: List[ConstraintCheck] = []

        report.add(
            ConstraintCheck(
                name="analysis_has_snapshots",
                passed=bool(products.spectra or products.reynolds_stress or products.time_history),
                severity="error",
                message="analysis must contain at least one physical product",
            )
        )

        if products.stationarity is not None:
            diagnostics.append(
                ConstraintCheck(
                    name="statistically_stationary",
                    passed=products.stationarity.stationary,
                    severity="warning",
                    message=products.stationarity.reason or "stationarity assessment (informational)",
                    value=products.stationarity.model_dump(mode="json"),
                )
            )
        elif config and config.forcing.type.value != "none" and config.analysis.stationarity:
            diagnostics.append(
                ConstraintCheck(
                    name="stationarity_available",
                    passed=False,
                    severity="warning",
                    message="forced HIT stationarity was requested but not computed",
                )
            )

        if products.resolution is not None:
            value = products.resolution.kmax_eta_min
            diagnostics.append(
                ConstraintCheck(
                    name="resolution_kmax_eta",
                    passed=value is not None and value >= thresholds.minimum_kmax_eta,
                    severity="warning",
                    message="kmax*eta diagnostic (not an acceptance gate)",
                    value=value,
                    limit=f">= {thresholds.minimum_kmax_eta}",
                )
            )

        if products.energy_balance is not None:
            relative = products.energy_balance.relative_error_mean
            diagnostics.append(
                ConstraintCheck(
                    name="energy_balance",
                    passed=relative is not None
                    and relative <= thresholds.maximum_energy_balance_relative_error,
                    severity="warning",
                    message="energy-balance diagnostic (not an acceptance gate)",
                    value=relative,
                    limit=f"<= {thresholds.maximum_energy_balance_relative_error}",
                )
            )

        if products.spectral_isotropy:
            deviations = [
                item.maximum_component_deviation
                for item in products.spectral_isotropy
                if item.maximum_component_deviation is not None
            ]
            maximum = max(deviations) if deviations else self._infer_spectral_deviation(products)
            diagnostics.append(
                ConstraintCheck(
                    name="spectral_component_isotropy",
                    passed=maximum is not None
                    and maximum <= thresholds.maximum_component_energy_deviation,
                    severity="warning",
                    message="spectral isotropy diagnostic (not an acceptance gate)",
                    value=maximum,
                    limit=f"<= {thresholds.maximum_component_energy_deviation}",
                )
            )

        if products.reynolds_stress:
            deviations = [
                self._stress_diagonal_deviation(item.r11, item.r22, item.r33)
                for item in products.reynolds_stress
            ]
            maximum = max(deviations)
            diagnostics.append(
                ConstraintCheck(
                    name="reynolds_stress_isotropy",
                    passed=maximum <= thresholds.maximum_component_energy_deviation,
                    severity="warning",
                    message="Reynolds-stress isotropy diagnostic (not an acceptance gate)",
                    value=maximum,
                    limit=f"<= {thresholds.maximum_component_energy_deviation}",
                )
            )
            realizable = all(self._stress_realizable(item) for item in products.reynolds_stress)
            diagnostics.append(
                ConstraintCheck(
                    name="reynolds_stress_realizability",
                    passed=realizable,
                    severity="warning",
                    message="Reynolds-stress realizability diagnostic (not an acceptance gate)",
                )
            )

        if products.spectra:
            slopes = [item.inertial_slope for item in products.spectra if item.inertial_slope is not None]
            if slopes:
                mean_slope = float(np.mean(slopes))
                diagnostics.append(
                    ConstraintCheck(
                        name="inertial_range_slope",
                        passed=-2.0 <= mean_slope <= -1.35,
                        severity="warning",
                        message="inertial-range slope diagnostic (not an acceptance gate)",
                        value=mean_slope,
                        limit="[-2.0, -1.35]",
                    )
                )

        divergence_checked = False
        if products.time_history is not None:
            history = products.time_history
            if history.mach_max:
                maximum_mach = max(history.mach_max)
                diagnostics.append(
                    ConstraintCheck(
                        name="measured_mach",
                        passed=maximum_mach <= thresholds.max_mach,
                        severity="warning",
                        message="measured Mach diagnostic (not an acceptance gate)",
                        value=maximum_mach,
                        limit=f"<= {thresholds.max_mach}",
                    )
                )
            if history.divergence_rms:
                maximum_divergence = max(history.divergence_rms)
                divergence_checked = True
                report.add(
                    ConstraintCheck(
                        name="measured_divergence",
                        passed=maximum_divergence <= thresholds.maximum_divergence_rms,
                        severity="error",
                        message="measured velocity divergence must remain below tolerance",
                        value=maximum_divergence,
                        limit=f"<= {thresholds.maximum_divergence_rms}",
                    )
                )

        if not divergence_checked:
            diagnostics.append(
                ConstraintCheck(
                    name="measured_divergence",
                    passed=True,
                    severity="warning",
                    message="no divergence time history available; divergence acceptance was not evaluated",
                )
            )

        if manifest is not None:
            velocity_files = manifest.files_of_kind("velocity_field")
            diagnostics.append(
                ConstraintCheck(
                    name="manifest_velocity_fields",
                    passed=bool(velocity_files),
                    severity="warning",
                    message="velocity fields in manifest (informational)",
                    value=len(velocity_files),
                )
            )
            checksums_present = all(bool(item.checksum) for item in manifest.files)
            diagnostics.append(
                ConstraintCheck(
                    name="manifest_checksums",
                    passed=checksums_present,
                    severity="warning",
                    message="dataset checksum coverage (informational)",
                )
            )

        for check in diagnostics:
            report.add(check)

        report.metadata["status"] = "PASSED" if report.passed else "FAILED"
        report.metadata["diagnostics_only"] = [
            check.name for check in diagnostics if check.severity == "warning"
        ]
        products.validation_status = report.metadata["status"]
        return report

    @staticmethod
    def _stress_diagonal_deviation(r11: float, r22: float, r33: float) -> float:
        values = np.asarray([r11, r22, r33], dtype=float)
        mean = abs(float(np.mean(values)))
        if mean <= 1.0e-15:
            return float("inf")
        return float(np.max(np.abs(values - mean)) / mean)

    @staticmethod
    def _stress_realizable(item) -> bool:
        tensor = np.asarray(
            [
                [item.r11, item.r12, item.r13],
                [item.r12, item.r22, item.r23],
                [item.r13, item.r23, item.r33],
            ],
            dtype=float,
        )
        eigenvalues = np.linalg.eigvalsh(tensor)
        return bool(np.all(eigenvalues >= -1.0e-12))

    @staticmethod
    def _infer_spectral_deviation(products: HITAnalysisProducts) -> Optional[float]:
        deviations: List[float] = []
        for item in products.spectral_isotropy:
            arrays = [np.asarray(item.e11), np.asarray(item.e22), np.asarray(item.e33)]
            if not arrays[0].size or not all(array.shape == arrays[0].shape for array in arrays):
                continue
            stacked = np.vstack(arrays)
            mean = np.mean(stacked, axis=0)
            mask = np.abs(mean) > 1.0e-15
            if np.any(mask):
                deviations.append(float(np.max(np.abs(stacked[:, mask] - mean[mask]) / np.abs(mean[mask]))))
        return max(deviations) if deviations else None


__all__ = ["HITValidationAgent"]
