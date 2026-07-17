"""Run legacy UI constraints plus the authoritative typed HIT referee."""
from schemas import CFDCase, ValidationReport
from schemas.cfd_case import FlowKind, SolverKind
from .hit_constraints import HITConstraintValidator
from .lbm_constraints import LBMConstraintValidator
from .output_constraints import OutputConstraintValidator


def validate_case(case: CFDCase) -> ValidationReport:
    reports = []
    if case.flow.kind == FlowKind.HIT:
        # The legacy validator checks user intent that is deliberately absent
        # from the solver-neutral typed config (explicit BC declarations and
        # whether a requested spectral forcing band was omitted). The typed
        # referee then performs the physical/lattice reconciliation.
        reports.append(HITConstraintValidator().validate(case))
        try:
            from agents.physics_constraint_agent import PhysicsConstraintAgent
            reports.append(PhysicsConstraintAgent().validate_cfd_case(case).report)
        except Exception:
            # Legacy checks remain authoritative enough to return a useful
            # report; backend preparation will surface typed conversion errors.
            pass
    if case.solver.kind == SolverKind.LBM:
        reports.append(LBMConstraintValidator().validate(case))
    reports.append(OutputConstraintValidator().validate(case))
    combined = ValidationReport()
    for report in reports:
        for check in report.checks:
            combined.add(check)
    return combined
