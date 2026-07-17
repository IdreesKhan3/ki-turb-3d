"""Physics constraints specific to lattice-Boltzmann (LBM) solvers."""

from __future__ import annotations

from schemas import CFDCase, ConstraintCheck, ValidationReport
from schemas.cfd_case import FlowKind


class LBMConstraintValidator:
    name = "lbm"

    def validate(self, case: CFDCase) -> ValidationReport:
        report = ValidationReport()
        extra = case.solver.extra or {}
        tau = extra.get("relaxation_time")
        target_mach = extra.get("mach_number")
        mach = target_mach

        if case.flow.kind == FlowKind.HIT:
            try:
                from schemas.openlb_hit import OpenLBHITConfig

                derived = OpenLBHITConfig.from_cfd_case(case).derive_scaling()
                mach = derived.actual_mach
            except Exception:
                mach = target_mach

        if target_mach is not None:
            report.add(ConstraintCheck(
                name="lbm_target_mach",
                passed=float(target_mach) < 0.1,
                severity="warning",
                message="requested target Mach (informational; not an acceptance gate)",
                value=target_mach,
                limit="< 0.1",
            ))

        if mach is not None:
            report.add(ConstraintCheck(
                name="lbm_derived_mach",
                passed=float(mach) < 0.1,
                severity="warning",
                message="derived Mach number (informational; not an acceptance gate)",
                value=mach,
                limit="< 0.1",
            ))

        if tau is not None:
            report.add(ConstraintCheck(
                name="lbm_tau",
                passed=tau > 0.5,
                severity="warning",
                message="relaxation time tau (informational; not an acceptance gate)",
                value=tau,
                limit="> 0.5",
            ))

        if case.solver.viscosity is not None:
            report.add(ConstraintCheck(
                name="lbm_viscosity",
                passed=case.solver.viscosity > 0,
                severity="warning",
                message="kinematic viscosity (informational; not an acceptance gate)",
                value=case.solver.viscosity,
            ))

        return report
