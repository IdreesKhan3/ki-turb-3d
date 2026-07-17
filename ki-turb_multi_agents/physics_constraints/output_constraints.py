"""Constraints ensuring requested analyses have the raw data they need."""

from __future__ import annotations

from schemas import CFDCase, ConstraintCheck, ValidationReport


class OutputConstraintValidator:
    name = "outputs"

    def validate(self, case: CFDCase) -> ValidationReport:
        report = ValidationReport()
        out = case.outputs

        wants_analysis = (
            out.write_spectra
            or out.write_isotropy
            or out.write_flatness
            or out.write_structure_functions
            or out.write_pdfs
        )

        report.add(ConstraintCheck(
            name="velocity_output_required",
            passed=(not wants_analysis) or out.write_velocity,
            severity="error",
            message="Velocity fields are required to compute spectra, isotropy, "
                    "flatness, PDFs, and structure functions.",
            value=out.write_velocity,
        ))

        report.add(ConstraintCheck(
            name="sample_interval_positive",
            passed=out.sample_interval > 0,
            severity="error",
            message="Output sample interval must be positive.",
            value=out.sample_interval,
        ))

        return report
