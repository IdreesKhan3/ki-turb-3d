"""Physics constraints for homogeneous isotropic turbulence (HIT) cases."""

from __future__ import annotations

from integrations.openlb_hit_catalog import (
    DECAYING_FORCING,
    FORCING_PATTERNS,
    FORCING_SCHEMES,
    ALL_COLLISIONS,
    normalize_collision,
    normalize_forcing_pattern,
    normalize_forcing_scheme,
    normalize_turbulence_regime,
)
from schemas import CFDCase, ConstraintCheck, ValidationReport
from schemas.cfd_case import GeometryKind, HITMode

from .common import check_resolution

_ACTIVE_FORCING = FORCING_SCHEMES - {"none", "off", ""}


def _resolved_hit_mode(case: CFDCase) -> HITMode:
    if case.flow.hit_mode is not None:
        return case.flow.hit_mode
    forcing = normalize_forcing_scheme(case.flow.forcing_type)
    if forcing in DECAYING_FORCING:
        return HITMode.DECAYING
    return HITMode.FORCED


class HITConstraintValidator:
    name = "hit"

    def validate(self, case: CFDCase) -> ValidationReport:
        report = ValidationReport()

        nx, ny, nz = case.mesh.resolution
        lx, ly, lz = case.geometry.size
        hit_mode = _resolved_hit_mode(case)
        forcing = normalize_forcing_scheme(case.flow.forcing_type)
        pattern = normalize_forcing_pattern(case.flow.forcing_pattern)
        collision = normalize_collision(case.solver.scheme)
        extra = case.solver.extra or {}
        regime = normalize_turbulence_regime(None, case.solver.scheme)

        report.add(check_resolution(case.mesh.resolution))

        report.add(ConstraintCheck(
            name="hit_uses_box_geometry",
            passed=case.geometry.kind == GeometryKind.BOX,
            severity="error",
            message="HIT requires a periodic box geometry.",
            value=case.geometry.kind.value,
        ))

        report.add(ConstraintCheck(
            name="hit_cube_domain",
            passed=abs(lx - ly) < 1e-12 and abs(ly - lz) < 1e-12,
            severity="error",
            message="HIT should use a cubic domain: Lx = Ly = Lz.",
            value=case.geometry.size,
        ))

        report.add(ConstraintCheck(
            name="hit_equal_resolution",
            passed=nx == ny == nz,
            severity="error",
            message="HIT should use equal resolution Nx = Ny = Nz.",
            value=case.mesh.resolution,
        ))

        bcs = [bc.type.lower() for bc in case.boundary_conditions]
        report.add(ConstraintCheck(
            name="hit_periodic_boundaries",
            passed=(not bcs) or all(t == "periodic" for t in bcs),
            severity="error",
            message="HIT requires periodic boundary conditions.",
            value=bcs,
        ))

        report.add(ConstraintCheck(
            name="hit_collision_known",
            passed=collision in ALL_COLLISIONS,
            severity="error",
            message=f"Unknown OpenLB HIT collision model: {case.solver.scheme}",
            value=case.solver.scheme,
        ))

        report.add(ConstraintCheck(
            name="hit_collision_regime_label",
            passed=True,
            severity="warning",
            message="turbulence_regime is derived from the requested collision model (informational only)",
            value={"regime": regime, "collision": collision, "scheme": case.solver.scheme},
        ))

        report.add(ConstraintCheck(
            name="hit_mode_consistent",
            passed=(
                (hit_mode == HITMode.DECAYING and forcing in DECAYING_FORCING)
                or (hit_mode == HITMode.FORCED and forcing in _ACTIVE_FORCING)
            ),
            severity="error",
            message="HIT mode must match forcing_scheme (decaying → none; forced → active scheme).",
            value={"hit_mode": hit_mode.value, "forcing_scheme": forcing},
        ))

        if pattern and pattern not in FORCING_PATTERNS:
            report.add(ConstraintCheck(
                name="hit_forcing_pattern_known",
                passed=False,
                severity="error",
                message=f"Unknown forcing pattern: {case.flow.forcing_pattern}",
                value=pattern,
            ))

        if hit_mode == HITMode.FORCED:
            kmin = case.flow.forcing_wavenumber_min
            kmax = case.flow.forcing_wavenumber_max
            report.add(ConstraintCheck(
                name="hit_forcing_band",
                passed=(
                    forcing in {"linear", "constant", "abc", "ornstein_uhlenbeck"}
                    or (kmin is not None and kmax is not None and 0 < kmin <= kmax)
                ),
                severity="error",
                message="Spectral forcing requires 0 < ForcingKMin <= ForcingKMax.",
                value={"kmin": kmin, "kmax": kmax, "forcing": forcing},
            ))
            amp = case.flow.forcing_amplitude
            report.add(ConstraintCheck(
                name="hit_forcing_amplitude",
                passed=amp is None or amp > 0,
                severity="error",
                message="Forced HIT forcing amplitude must be positive when set.",
                value=amp,
            ))

        ic_kmin = case.flow.ic_wavenumber_min
        ic_kmax = case.flow.ic_wavenumber_max
        if ic_kmin is not None or ic_kmax is not None:
            report.add(ConstraintCheck(
                name="hit_ic_band",
                passed=(
                    ic_kmin is not None and ic_kmax is not None and 0 < ic_kmin <= ic_kmax
                ),
                severity="error",
                message="Initial-condition wavenumber band requires 0 < ICKMin <= ICKMax.",
                value={"ic_kmin": ic_kmin, "ic_kmax": ic_kmax},
            ))

        if collision == "trt":
            magic = extra.get("trt_magic_parameter")
            if magic is not None:
                report.add(ConstraintCheck(
                    name="hit_trt_magic_parameter",
                    passed=0 < float(magic) < 1,
                    severity="error",
                    message="TRT magic parameter must be in (0, 1).",
                    value=magic,
                ))

        return report
