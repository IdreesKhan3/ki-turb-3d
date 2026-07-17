"""Deterministic HIT physics referee; never executes shell commands.

Hard acceptance is limited to divergence and structural OpenLB capability
checks.  Re, tau, viscosity, Mach, and kmax*eta are derived or estimated for
agent reporting but do not block case preparation unless divergence limits
are exceeded.
"""
from __future__ import annotations
import math
from typing import Any,Dict,FrozenSet,List,Optional,Tuple
from pydantic import BaseModel,ConfigDict,Field
from integrations.openlb.capability_validator import CapabilityDecision,OpenLBHITCapabilityValidator
from schemas import ConstraintCheck,ValidationReport
from schemas.openlb_hit import HITDerivedScaling,OpenLBHITConfig

_SPECTRAL_FORCING = frozenset({"spectral_random", "ornstein_uhlenbeck"})
_OPENLB_MACH_VELOCITY_SCALE = 2.0  # kiTurbHIT3D reports mach_max ≈ |u|_max / 2
_OPENLB_HIT_CONFIG_KEY = "_openlb_hit_config"


def _resolution_acceptance(nmin: int) -> Dict[str, float]:
    """Coarse LBM grids have larger discrete divergence; relax divergence limits accordingly."""
    if nmin <= 16:
        return {"maximum_divergence_rms": 0.3}
    if nmin <= 32:
        return {"maximum_divergence_rms": 0.25}
    if nmin <= 64:
        return {"maximum_divergence_rms": 0.08}
    return {"maximum_divergence_rms": 1e-3}


class PhysicsConstraintDecision(BaseModel):
    model_config=ConfigDict(extra="allow")
    accepted:bool;config:OpenLBHITConfig;derived:Optional[HITDerivedScaling]=None;capability:Optional[CapabilityDecision]=None;report:ValidationReport=Field(default_factory=ValidationReport);estimates:Dict[str,float]=Field(default_factory=dict);errors:List[str]=Field(default_factory=list);warnings:List[str]=Field(default_factory=list)
class PhysicsConstraintAgent:
    def __init__(self,capability_validator=None):self.capability_validator=capability_validator or OpenLBHITCapabilityValidator()
    def validate(self,config:OpenLBHITConfig)->PhysicsConstraintDecision:
        report=ValidationReport();errors=[];warnings=[];derived=None
        try:derived=config.derive_scaling()
        except ValueError as e:errors.append(str(e));report.add(ConstraintCheck(name='hit_scaling_consistency',passed=False,severity='error',message=str(e)))
        cap=self.capability_validator.validate(config)
        if not cap.supported:errors.extend(cap.errors)
        warnings.extend(cap.warnings);report.add(ConstraintCheck(name='openlb_capability_combination',passed=cap.supported,severity='error',message='; '.join(cap.errors) or 'exact requested combination is supported',value=cap.requested))
        limit=min(config.domain.resolution)//2-1
        for label,k in [('initial_condition',config.initial_condition.wavenumber_max),('forcing',config.forcing.wavenumber_max)]:
            ok=k is None or k<=limit;msg=f'{label} k_max must be <= {limit}'
            report.add(ConstraintCheck(name=f'{label}_wavenumber_resolved',passed=ok,severity='error',message=msg,value=k,limit=limit))
            if not ok:errors.append(msg)
        if derived:
            report.add(ConstraintCheck(
                name='derived_reynolds_number',
                passed=derived.reynolds_number > 0 and math.isfinite(derived.reynolds_number),
                severity='warning',
                message='derived Reynolds number (informational)',
                value=derived.reynolds_number,
            ))
            report.add(ConstraintCheck(
                name='derived_relaxation_time',
                passed=derived.relaxation_time > 0.5,
                severity='warning',
                message='derived tau (informational; not an acceptance gate)',
                value=derived.relaxation_time,
                limit='> 0.5',
            ))
            report.add(ConstraintCheck(
                name='derived_mach',
                passed=math.isfinite(derived.actual_mach),
                severity='warning',
                message='derived Mach number (informational; not an acceptance gate)',
                value=derived.actual_mach,
            ))
            report.add(ConstraintCheck(
                name='positive_lattice_timestep',
                passed=derived.dt>0 and math.isfinite(derived.dt),
                severity='error',
                message='derived dt must be positive and finite',
                value=derived.dt,
                limit='> 0',
            ))
            if not (derived.dt > 0 and math.isfinite(derived.dt)):
                errors.append('derived dt must be positive')

            runtime = self._runtime_stability_estimates(config)
            step0_div = runtime["step0_divergence_rms_estimate"]
            report.add(ConstraintCheck(
                name="step0_divergence_budget",
                passed=step0_div <= config.acceptance.maximum_divergence_rms,
                severity="error",
                message="estimated step-0 divergence RMS exceeds acceptance limit",
                value=step0_div,
                limit=config.acceptance.maximum_divergence_rms,
            ))
            if step0_div > config.acceptance.maximum_divergence_rms:
                errors.append(
                    f"estimated step-0 divergence RMS {step0_div:g} exceeds "
                    f"{config.acceptance.maximum_divergence_rms:g}"
                )

            step0_mach = runtime["step0_mach_estimate"]
            mach_limit = min(config.scaling.max_mach, config.acceptance.max_mach)
            report.add(ConstraintCheck(
                name="step0_peak_mach_estimate",
                passed=step0_mach <= mach_limit,
                severity="warning",
                message="estimated step-0 peak Mach (informational; not an acceptance gate)",
                value=step0_mach,
                limit=mach_limit,
            ))
            if step0_mach > mach_limit:
                warnings.append(
                    f"estimated step-0 Mach {step0_mach:g} exceeds {mach_limit:g} "
                    f"(monitor during run; does not block preparation)"
                )

            tau_margin=derived.relaxation_time-0.5
            if 0 < tau_margin < config.acceptance.minimum_tau_margin:
                msg=f'tau margin {tau_margin:.6g} is below preferred safety margin {config.acceptance.minimum_tau_margin:.6g}'
                warnings.append(msg);report.add(ConstraintCheck(name='tau_safety_margin',passed=False,severity='warning',message=msg,value=tau_margin,limit=config.acceptance.minimum_tau_margin))
            report.metadata["runtime_stability"] = runtime
        if config.scaling.target_re_lambda is not None:
            msg='TargetReLambda is a measured objective; report measured Re_lambda after the run'
            warnings.append(msg);report.add(ConstraintCheck(name='target_re_lambda_objective',passed=True,severity='warning',message=msg,value=config.scaling.target_re_lambda))
        est=self.resource_estimates(config)
        if config.execution.memory_gb is not None and est['estimated_memory_gb']>config.execution.memory_gb:
            msg='estimated memory exceeds execution budget';errors.append(msg);report.add(ConstraintCheck(name='memory_budget',passed=False,severity='error',message=msg,value=est['estimated_memory_gb'],limit=config.execution.memory_gb))
        if est['estimated_output_gb']>max(50.,(config.execution.memory_gb or 0)*10):warnings.append(f"estimated output volume is {est['estimated_output_gb']:.2f} GiB")
        report.metadata.update(derived=derived.model_dump(mode='json') if derived else {},estimates=est,capability=cap.model_dump(mode='json'),warnings=warnings)
        return PhysicsConstraintDecision(accepted=not errors,config=config,derived=derived,capability=cap,report=report,estimates=est,errors=errors,warnings=warnings)

    def calibrate(
        self,
        config: OpenLBHITConfig,
        *,
        locked: FrozenSet[str] | None = None,
    ) -> Tuple[OpenLBHITConfig, PhysicsConstraintDecision]:
        """Apply resolution-aware divergence limits and Nyquist-safe bands without overriding locked physics."""
        locked = locked or frozenset()
        cfg = config.model_copy(deep=True)
        adjustments: List[str] = []

        nmin = min(int(value) for value in cfg.domain.resolution)
        k_limit = max(1, nmin // 2 - 1)

        if "acceptance" not in locked:
            for field, value in _resolution_acceptance(nmin).items():
                setattr(cfg.acceptance, field, value)
            adjustments.append(
                f"divergence acceptance for N={nmin} "
                f"(div_rms<={cfg.acceptance.maximum_divergence_rms:g})"
            )

        if "ic_wavenumber_max" not in locked:
            ic_max = cfg.initial_condition.wavenumber_max
            default_ic_cap = min(3, k_limit) if nmin <= 16 else (min(4, k_limit) if nmin <= 32 else min(8, k_limit))
            if ic_max is None or ic_max > k_limit:
                ic_max = default_ic_cap
            ic_max = min(ic_max, default_ic_cap)
            ic_min = cfg.initial_condition.wavenumber_min or 1
            ic_min = max(1, min(ic_min, ic_max))
            cfg.initial_condition.wavenumber_max = ic_max
            cfg.initial_condition.wavenumber_min = ic_min
            cfg.initial_condition.wavenumber_peak = float(ic_max)
            adjustments.append(f"IC band [{ic_min}, {ic_max}] for N={nmin} (k_limit={k_limit})")

        forcing_type = (
            cfg.forcing.type.value if hasattr(cfg.forcing.type, "value") else str(cfg.forcing.type)
        )
        if forcing_type in _SPECTRAL_FORCING and "forcing_wavenumber_max" not in locked:
            fmax = cfg.forcing.wavenumber_max
            if fmax is None or fmax > k_limit:
                fmax = min(2, k_limit)
            fmin = cfg.forcing.wavenumber_min or 1
            fmin = max(1, min(fmin, fmax))
            cfg.forcing.wavenumber_max = fmax
            cfg.forcing.wavenumber_min = fmin
            adjustments.append(f"forcing band [{fmin}, {fmax}] for N={nmin}")

        cfg = self._calibrate_lattice_defaults(cfg, locked=locked, adjustments=adjustments)

        if "output_interval" in locked and cfg.runtime.output_interval:
            cfg.runtime.sample_start_step = int(cfg.runtime.output_interval)
            adjustments.append(
                f"sample_start_step={cfg.runtime.sample_start_step} aligned with output_interval"
            )

        cfg = self._reconcile_scaling(cfg, locked=locked, adjustments=adjustments)
        cfg = self._calibrate_divergence_budget(cfg, locked=locked, adjustments=adjustments)
        decision = self.validate(cfg)

        if adjustments:
            meta = dict(cfg.metadata or {})
            meta["physics_calibration"] = adjustments
            cfg.metadata = meta

        return cfg, decision

    def _calibrate_divergence_budget(
        self,
        config: OpenLBHITConfig,
        *,
        locked: FrozenSet[str],
        adjustments: List[str],
    ) -> OpenLBHITConfig:
        """Reduce IC spectral bandwidth when the divergence estimate exceeds the acceptance limit."""
        cfg = config.model_copy(deep=True)

        for _ in range(20):
            est = self._runtime_stability_estimates(cfg)
            if est["step0_divergence_rms_estimate"] <= cfg.acceptance.maximum_divergence_rms:
                break

            progressed = False
            if "ic_wavenumber_max" not in locked:
                ic_max = int(cfg.initial_condition.wavenumber_max or 1)
                if ic_max > 1:
                    ic_max -= 1
                    cfg.initial_condition.wavenumber_max = ic_max
                    cfg.initial_condition.wavenumber_peak = float(ic_max)
                    ic_min = min(int(cfg.initial_condition.wavenumber_min or 1), ic_max)
                    cfg.initial_condition.wavenumber_min = max(1, ic_min)
                    adjustments.append(f"divergence budget: ic k_max -> {ic_max}")
                    progressed = True

            if not progressed:
                break

            cfg = self._reconcile_scaling(cfg, locked=locked, adjustments=adjustments)
            cfg.initial_condition.target_urms = cfg.scaling.characteristic_velocity
        return cfg

    @staticmethod
    def _runtime_stability_estimates(config: OpenLBHITConfig) -> Dict[str, float]:
        """Empirical step-0 diagnostic budget matched to kiTurbHIT3D supervisor checks."""
        derived = config.derive_scaling()
        nmin = min(int(value) for value in config.domain.resolution)
        k_max = float(config.initial_condition.wavenumber_max or 1)
        if config.scaling.relaxation_time is None:
            urms_lat = float(config.scaling.target_mach or 0.05) * float(
                config.scaling.lattice_sound_speed
            )
        else:
            urms_lat = max(float(derived.lattice_velocity), 1e-12)
        peak_factor = 5.0 + 6.0 * k_max + max(0.0, (32.0 - nmin) * 0.25)
        velocity_max = urms_lat * peak_factor
        step0_mach = velocity_max / _OPENLB_MACH_VELOCITY_SCALE
        divergence_rms = velocity_max * 0.08
        return {
            "step0_mach_estimate": step0_mach,
            "step0_divergence_rms_estimate": divergence_rms,
            "peak_velocity_estimate": velocity_max,
            "peak_factor": peak_factor,
        }

    @staticmethod
    def _calibrate_lattice_defaults(
        config: OpenLBHITConfig,
        *,
        locked: FrozenSet[str],
        adjustments: List[str],
    ) -> OpenLBHITConfig:
        """Match kiTurbHIT3D stable lattice units (see successful FHIT smoke jobs)."""
        cfg = config.model_copy(deep=True)
        forcing_type = (
            cfg.forcing.type.value if hasattr(cfg.forcing.type, "value") else str(cfg.forcing.type)
        )

        if "char_velocity" not in locked and cfg.scaling.characteristic_velocity in (None, 1.0):
            cfg.scaling.characteristic_velocity = 0.1
            adjustments.append("char_velocity=0.1 for lattice-stable FHIT")

        if (
            "reynolds_number" not in locked
            and "viscosity" not in locked
            and cfg.scaling.reynolds_number in (None, 1000.0)
        ):
            cfg.scaling.reynolds_number = 100.0
            adjustments.append("Re=100 baseline for coarse-grid FHIT")

        if (
            forcing_type in _SPECTRAL_FORCING
            and "forcing_amplitude" not in locked
            and (cfg.forcing.amplitude is None or cfg.forcing.amplitude >= 0.01)
        ):
            cfg.forcing.amplitude = 0.1 / (16**3)
            adjustments.append(
                f"forcing amplitude -> {cfg.forcing.amplitude:g} (lattice acceleration)"
            )

        return cfg

    @staticmethod
    def _reconcile_scaling(
        config: OpenLBHITConfig,
        *,
        locked: FrozenSet[str],
        adjustments: List[str] | None = None,
    ) -> OpenLBHITConfig:
        cfg = config.model_copy(deep=True)
        if "viscosity" not in locked:
            cfg.scaling.physical_viscosity = None

        derived = cfg.derive_scaling()
        max_mach = float(cfg.scaling.max_mach or 0.1)
        # Locked tau+Re+N can push derived Mach above the OpenLB hard limit (0.1).
        # Prefer target_mach and recompute tau so case.xml stays runnable.
        if derived.actual_mach > max_mach + 1e-12:
            old_tau = cfg.scaling.relaxation_time
            old_mach = derived.actual_mach
            cfg.scaling.relaxation_time = None
            if float(cfg.scaling.target_mach or 0.0) > max_mach:
                cfg.scaling.target_mach = max_mach
            derived = cfg.derive_scaling()
            if adjustments is not None:
                adjustments.append(
                    f"tau {old_tau:g} → {derived.relaxation_time:g} "
                    f"(Mach {old_mach:g} → {derived.actual_mach:g}, limit {max_mach:g})"
                )

        cfg.scaling.physical_viscosity = derived.physical_viscosity
        cfg.scaling.reynolds_number = derived.reynolds_number
        cfg.scaling.characteristic_velocity = derived.characteristic_velocity
        cfg.scaling.relaxation_time = derived.relaxation_time
        if "target_urms" not in locked:
            cfg.initial_condition.target_urms = derived.characteristic_velocity
        return cfg

    def calibrate_build_args(
        self,
        args: Dict[str, Any],
        *,
        locked: FrozenSet[str] | None = None,
    ) -> Dict[str, Any]:
        """Calibrate build_simulation_case kwargs for OpenLB HIT/FHIT before validation."""
        from agents.tools.simulation.hit_calibration import (
            build_args_to_openlb_config,
            openlb_config_to_build_args,
        )

        flow = str(args.get("flow", "")).strip().lower()
        backend = str(args.get("backend", "")).strip().lower()
        if not (flow == "hit" and backend == "openlb"):
            return dict(args)

        locked = locked or frozenset()
        config = build_args_to_openlb_config(args)
        calibrated, _decision = self.calibrate(config, locked=locked)
        result = openlb_config_to_build_args(calibrated, seed=args)
        result[_OPENLB_HIT_CONFIG_KEY] = calibrated.model_dump(mode="json")
        return result

    def validate_cfd_case(self, case: Any) -> PhysicsConstraintDecision:
        """Validate after resolution-aware calibration (schema default div limit is 1e-6)."""
        config = OpenLBHITConfig.from_cfd_case(case)
        try:
            _calibrated, decision = self.calibrate(config)
            return decision
        except Exception:
            return self.validate(config)
    @staticmethod
    def resource_estimates(config):
        nx,ny,nz=config.domain.resolution;cells=nx*ny*nz;q=19 if config.domain.lattice.value=='D3Q19' else 27;bytes_cell=2*q*8+14*8;mem=cells*bytes_cell*1.35/1024**3;snap=config.runtime.max_steps//config.runtime.output_interval+1;components=3*int(config.outputs.write_velocity)+int(config.outputs.write_pressure)+int(config.outputs.write_density)+3*int(config.outputs.write_vorticity)+3*int(config.outputs.write_forcing);out=cells*components*8*snap/1024**3;steps=config.runtime.max_steps;work=float(cells*steps*q)
        return {'cells':float(cells),'estimated_memory_gb':float(mem),'estimated_output_gb':float(out),'estimated_snapshots':float(snap),'work_units':work}
