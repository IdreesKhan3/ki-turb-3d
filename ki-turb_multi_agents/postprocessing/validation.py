"""Sanity checks on post-processing products before they reach KI-TURB."""

from __future__ import annotations

from typing import List

import numpy as np

from schemas import ConstraintCheck, ValidationReport

from .readers import VelocitySnapshot


def validate_products(snapshots: List[VelocitySnapshot], products: dict) -> ValidationReport:
    report = ValidationReport()

    report.add(ConstraintCheck(
        name="snapshots_present",
        passed=len(snapshots) > 0,
        severity="error",
        message="No velocity snapshots were read from the raw output.",
        value=len(snapshots),
    ))

    spectra = products.get("spectra") or []
    energy_finite = all(np.all(np.isfinite(s["E"])) for s in spectra)
    report.add(ConstraintCheck(
        name="spectra_finite",
        passed=(not spectra) or energy_finite,
        severity="error",
        message="Energy spectra contain non-finite values.",
    ))

    isotropy = products.get("spectral_isotropy") or []
    if isotropy:
        ic = isotropy[-1]["columns"][:, 5]
        ic = ic[np.isfinite(ic) & (ic > 0)]
        near_isotropic = bool(ic.size and 0.5 < float(np.median(ic)) < 3.0)
        report.add(ConstraintCheck(
            name="isotropy_reasonable",
            passed=near_isotropic,
            severity="warning",
            message="Median IC_standard=E22/E11 far from isotropic (~1).",
            value=float(np.median(ic)) if ic.size else None,
        ))

    return report
