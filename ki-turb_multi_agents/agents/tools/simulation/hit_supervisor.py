"""Simulation-health supervisor for OpenLB HIT runs.

Only divergence and non-finite diagnostics abort a run.  Mach, density, mass
drift, kmax*eta, and related quantities are reported as warnings.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.openlb_hit import HITAcceptanceThresholds

from .execution_runners import ExecutionHandle, ExecutionRunner, ExecutionStatus


class HITDiagnostics(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    physical_time: Optional[float] = None
    progress: Optional[float] = None
    mass: Optional[float] = None
    density_min: Optional[float] = None
    density_max: Optional[float] = None
    velocity_max: Optional[float] = None
    mach_max: Optional[float] = None
    tke: Optional[float] = None
    dissipation: Optional[float] = None
    energy_input: Optional[float] = None
    divergence_rms: Optional[float] = None
    re_lambda: Optional[float] = None
    forcing_power: Optional[float] = None
    kmax_eta: Optional[float] = None
    disk_free_bytes: Optional[int] = None


class HealthDecision(BaseModel):
    model_config = ConfigDict(extra="allow")

    healthy: bool
    should_abort: bool = False
    should_checkpoint: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    diagnostics: Optional[HITDiagnostics] = None


class SupervisorResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    state: str
    execution_status: Optional[ExecutionStatus] = None
    health: Optional[HealthDecision] = None
    samples_seen: int = 0
    message: str = ""


class HITSupervisor:
    """Read structured diagnostics and enforce divergence/runtime limits."""

    _KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*([-+0-9.eE]+)")

    def __init__(
        self,
        thresholds: Optional[HITAcceptanceThresholds] = None,
        *,
        minimum_disk_free_bytes: int = 512 * 1024 * 1024,
        heartbeat_timeout_seconds: float = 600.0,
    ) -> None:
        self.thresholds = thresholds or HITAcceptanceThresholds()
        self.minimum_disk_free_bytes = minimum_disk_free_bytes
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._initial_mass: Optional[float] = None

    def evaluate(self, diagnostics: HITDiagnostics) -> HealthDecision:
        errors: List[str] = []
        warnings: List[str] = []

        numeric = diagnostics.model_dump(exclude_none=True)
        for name, value in numeric.items():
            if name in {"step", "disk_free_bytes"}:
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                errors.append(f"{name} is NaN or infinite")

        if diagnostics.divergence_rms is not None:
            if diagnostics.divergence_rms > self.thresholds.maximum_divergence_rms:
                errors.append(
                    f"divergence RMS {diagnostics.divergence_rms:g} exceeds "
                    f"{self.thresholds.maximum_divergence_rms:g}"
                )

        if diagnostics.mach_max is not None and diagnostics.mach_max > self.thresholds.max_mach:
            warnings.append(
                f"maximum Mach {diagnostics.mach_max:g} exceeds {self.thresholds.max_mach:g}"
            )
        if diagnostics.density_min is not None and diagnostics.density_max is not None:
            allowed = self.thresholds.maximum_density_deviation
            if diagnostics.density_min < 1.0 - allowed or diagnostics.density_max > 1.0 + allowed:
                warnings.append(
                    f"density range [{diagnostics.density_min:g}, {diagnostics.density_max:g}] "
                    f"exceeds ±{allowed:g} around unity"
                )
        if diagnostics.mass is not None:
            if self._initial_mass is None:
                self._initial_mass = diagnostics.mass
            reference = max(abs(self._initial_mass), 1.0e-15)
            drift = abs(diagnostics.mass - self._initial_mass) / reference
            if drift > self.thresholds.maximum_mass_drift_fraction:
                warnings.append(
                    f"mass drift {drift:g} exceeds {self.thresholds.maximum_mass_drift_fraction:g}"
                )
        if diagnostics.kmax_eta is not None and diagnostics.kmax_eta < self.thresholds.minimum_kmax_eta:
            warnings.append(
                f"kmax*eta {diagnostics.kmax_eta:g} is below {self.thresholds.minimum_kmax_eta:g}"
            )
        if diagnostics.disk_free_bytes is not None and diagnostics.disk_free_bytes < self.minimum_disk_free_bytes:
            errors.append("available disk space is below the configured safety limit")
        if diagnostics.tke is not None and diagnostics.tke < 0:
            warnings.append("TKE is negative")
        if diagnostics.dissipation is not None and diagnostics.dissipation < 0:
            warnings.append("dissipation is negative")

        return HealthDecision(
            healthy=not errors,
            should_abort=bool(errors),
            should_checkpoint=bool(errors),
            errors=errors,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def read_jsonl(self, path: str | Path, *, start_line: int = 0) -> List[HITDiagnostics]:
        diagnostics_path = Path(path)
        if not diagnostics_path.is_file():
            return []
        samples: List[HITDiagnostics] = []
        with diagnostics_path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index < start_line or not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    samples.append(HITDiagnostics.model_validate(payload))
                except (json.JSONDecodeError, ValueError):
                    continue
        return samples

    def parse_log_line(self, line: str) -> Optional[HITDiagnostics]:
        """Best-effort parser for legacy key/value progress lines."""
        values: Dict[str, float] = {
            key.lower(): float(value) for key, value in self._KEY_VALUE.findall(line)
        }
        if "step" not in values and "iter" not in values:
            return None
        aliases = {
            "step": int(values.get("step", values.get("iter", 0))),
            "physical_time": values.get("time"),
            "progress": values.get("progress"),
            "mass": values.get("mass"),
            "density_min": values.get("density_min", values.get("rhomin")),
            "density_max": values.get("density_max", values.get("rhomax")),
            "velocity_max": values.get("velocity_max", values.get("umax")),
            "mach_max": values.get("mach_max", values.get("mamax")),
            "tke": values.get("tke"),
            "dissipation": values.get("dissipation", values.get("eps")),
            "energy_input": values.get("energy_input"),
            "divergence_rms": values.get("divergence_rms"),
            "re_lambda": values.get("re_lambda"),
            "forcing_power": values.get("forcing_power"),
            "kmax_eta": values.get("kmax_eta"),
        }
        return HITDiagnostics.model_validate(aliases)

    def supervise_once(
        self,
        runner: ExecutionRunner,
        handle: ExecutionHandle,
        diagnostics_path: str | Path,
        *,
        last_line: int = 0,
    ) -> SupervisorResult:
        execution = runner.status(handle)
        samples = self.read_jsonl(diagnostics_path, start_line=last_line)
        health = self.evaluate(samples[-1]) if samples else None
        if health and health.should_abort and execution.state in {"running", "queued"}:
            if health.should_checkpoint:
                try:
                    runner.checkpoint(handle)
                    time.sleep(1.0)
                except Exception:
                    pass
            runner.cancel(handle)
            return SupervisorResult(
                state="rejected",
                execution_status=execution,
                health=health,
                samples_seen=len(samples),
                message="run cancelled after physical-health failure",
            )
        return SupervisorResult(
            state=execution.state,
            execution_status=execution,
            health=health,
            samples_seen=len(samples),
        )

    def supervise_until_terminal(
        self,
        runner: ExecutionRunner,
        handle: ExecutionHandle,
        diagnostics_path: str | Path,
        *,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> SupervisorResult:
        started = time.monotonic()
        seen = 0
        latest: Optional[SupervisorResult] = None
        while True:
            latest = self.supervise_once(runner, handle, diagnostics_path, last_line=seen)
            seen += latest.samples_seen
            if latest.state in {"completed", "failed", "cancelled", "rejected", "unknown"}:
                return latest
            if timeout is not None and time.monotonic() - started > timeout:
                runner.cancel(handle)
                latest.state = "cancelled"
                latest.message = "supervision timeout exceeded"
                return latest
            time.sleep(poll_interval)


__all__ = ["HITDiagnostics", "HealthDecision", "SupervisorResult", "HITSupervisor"]

# Compatibility helper used by backend polling.
class DiagnosticsAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")
    healthy: bool = True
    progress: Optional[float] = None
    latest: Optional[Dict[str, object]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

def _assess(self, path: str | Path) -> DiagnosticsAssessment:
    samples = self.read_jsonl(path)
    if not samples:
        return DiagnosticsAssessment(healthy=True)
    latest = samples[-1]
    decision = self.evaluate(latest)
    return DiagnosticsAssessment(healthy=decision.healthy, progress=latest.progress,
        latest=latest.model_dump(exclude_none=True), errors=decision.errors, warnings=decision.warnings)
HITSupervisor.assess = _assess
