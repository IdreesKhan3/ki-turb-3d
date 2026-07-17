"""Typed OpenLB HIT case factory with legacy mirrors for existing UI code."""
from __future__ import annotations
import math
from typing import Optional, Sequence
from schemas import (BoundaryCondition, CFDCase, ExecutionConfig, FlowConfig, Geometry,
                     Mesh, OutputConfig, RuntimeParams, SolverConfig)
from schemas.cfd_case import FlowKind, GeometryKind, HITMode, SolverKind, normalize_hit_mode
from schemas.openlb_hit import OpenLBHITConfig


def make_openlb_hit_case(
    name:str, resolution:Sequence[int]=(64,64,64), *, hit_mode:str="forced",
    reynolds_number:float=1000.0, viscosity:Optional[float]=None,
    max_steps:int=10000, output_interval:int=1000, mach_number:float=0.05,
    relaxation_time:Optional[float]=None, scheme:str="SmagorinskyBGK",
    smagorinsky_constant:float=0.1, density:float=1.0, char_velocity:float=1.0,
    box_length:float=2*math.pi, initial_condition:str="synthetic_spectrum",
    initial_condition_file:Optional[str]=None, ic_wavenumber_min:Optional[int]=None,
    ic_wavenumber_max:Optional[int]=None, ic_seed:int=12345, ic_spectrum_exponent:float=-2.0,
    forcing_type:Optional[str]=None, forcing_wavenumber_min:Optional[int]=None,
    forcing_wavenumber_max:Optional[int]=None, forcing_amplitude:float=0.1,
    forcing_update_interval:int=1, forcing_pattern:str="random_phase",
    forcing_correlation_time:Optional[float]=None, target_injection_rate:Optional[float]=None,
    target_tke:Optional[float]=None, turbulence_regime:Optional[str]=None,
    target_urms:Optional[float]=None, target_re_lambda:Optional[float]=None,
    statistically_stationary:bool=True, sample_start_step:Optional[int]=None,
    trt_magic_parameter:float=0.25, write_pressure:bool=False,
    write_density:bool=True, write_vorticity:bool=True, write_forcing:bool=False,
    checkpoint_interval:Optional[int]=None, diagnostics_interval:int=100,
    execution_mode:str="local", build_profile:str="serial", num_procs:int=1,
    num_threads:int=1,
)->CFDCase:
    resolution=tuple(int(n) for n in resolution)
    mode=HITMode(normalize_hit_mode(hit_mode))
    if viscosity is None: viscosity=char_velocity*box_length/reynolds_number
    if forcing_type is None: forcing_type="spectral_random" if mode==HITMode.FORCED else "none"
    force_alias={"spectral_low_k":"spectral_random","low_wavenumber":"spectral_random","ou":"ornstein_uhlenbeck","linear":"constant_energy_input"}
    forcing_type=force_alias.get(forcing_type.lower(),forcing_type.lower())
    if forcing_type=="ornstein_uhlenbeck" and forcing_correlation_time is None: forcing_correlation_time=1.0
    if forcing_type=="constant_energy_input" and target_injection_rate is None: target_injection_rate=forcing_amplitude
    if forcing_type=="constant_tke" and target_tke is None: target_tke=0.5*(target_urms or char_velocity)**2
    if sample_start_step is None: sample_start_step=max_steps//2 if mode==HITMode.FORCED else 0
    collision_alias={
        "smagorinsky": "SmagorinskyBGK",
        "dns": "BGK",
        "regularized": "RLB",
        "regularised": "RLB",
        "dynsmagorinsky": "DynamicSmagorinsky",
        "dynamicsmagorinsky": "DynamicSmagorinsky",
        "smagorinskimrt": "SmagorinskyMRT",
        "consistentstrainsmagorinsky": "ConsistentStrainSmagorinsky",
        "shearsmagorinsky": "ShearSmagorinsky",
        "krause": "Krause",
    }
    collision=collision_alias.get(scheme.replace("_","").lower(),scheme)
    from integrations.openlb_hit_catalog import normalize_turbulence_regime
    effective_regime = normalize_turbulence_regime(None, collision)
    ic_alias={"divergence_free_random":"synthetic_spectrum","synthetic":"synthetic_spectrum"}
    ic_type=ic_alias.get(initial_condition.lower(),initial_condition.lower())
    k_limit=max(1,min(resolution)//2-1)
    ic_kmin=ic_wavenumber_min if ic_wavenumber_min is not None else 1
    ic_kmax=ic_wavenumber_max if ic_wavenumber_max is not None else min(8,k_limit)
    ic_kmax=min(ic_kmax,k_limit)
    ic_kmin=max(1,min(ic_kmin,ic_kmax))
    force_kmin=forcing_wavenumber_min
    force_kmax=forcing_wavenumber_max
    if forcing_type!="none":
        if force_kmin is None: force_kmin=1
        if force_kmax is None: force_kmax=min(2,k_limit)
        force_kmax=min(force_kmax,k_limit)
        force_kmin=max(1,min(force_kmin,force_kmax))

    hit=OpenLBHITConfig(
        name=name, description=f"OpenLB {mode.value} homogeneous isotropic turbulence",
        domain={"resolution":resolution,"size":(box_length,box_length,box_length),"lattice":"D3Q19"},
        scaling={"density":density,"characteristic_length":box_length,"characteristic_velocity":char_velocity,
                 "physical_viscosity":viscosity,"reynolds_number":reynolds_number,
                 "target_re_lambda":target_re_lambda,"relaxation_time":relaxation_time,
                 "target_mach":mach_number,
                 "max_mach":max(0.1, float(mach_number or 0.1))},
        collision={"model":collision,"smagorinsky_constant":smagorinsky_constant,
                   "trt_magic_parameter":trt_magic_parameter},
        initial_condition={"type":ic_type,"seed":ic_seed,"wavenumber_min":ic_kmin,
                           "wavenumber_peak":float(ic_kmax),"wavenumber_max":ic_kmax,
                           "spectrum_exponent":ic_spectrum_exponent,"target_urms":target_urms,
                           "source_file":initial_condition_file},
        forcing={"type":forcing_type,"wavenumber_min":force_kmin if forcing_type!="none" else None,
                 "wavenumber_max":force_kmax if forcing_type!="none" else None,
                 "amplitude":forcing_amplitude if forcing_type in {"spectral_random","ornstein_uhlenbeck"} else None,
                 "target_injection_rate":target_injection_rate,"target_tke":target_tke,
                 "correlation_time":forcing_correlation_time,"update_interval":forcing_update_interval},
        runtime={"max_steps":max_steps,"output_interval":output_interval,"diagnostics_interval":diagnostics_interval,
                 "checkpoint_interval":checkpoint_interval,"sample_start_step":sample_start_step},
        execution={"mode":execution_mode,"build_profile":build_profile,"num_procs":num_procs,"num_threads":num_threads},
        checkpoint={"enabled":bool(checkpoint_interval),"interval":checkpoint_interval},
        outputs={"write_velocity":True,"write_pressure":write_pressure,"write_density":write_density,
                 "write_vorticity":write_vorticity,"write_forcing":write_forcing,"format":"vtm"},
        metadata={"turbulence_regime":effective_regime,"statistically_stationary":statistically_stationary,
                  "forcing_pattern_legacy":forcing_pattern})
    derived=hit.derive_scaling()
    # Legacy mirrors remain read-only compatibility data; the adapter consumes ``case.hit``.
    return CFDCase(
        name=name, description=hit.description, hit=hit,
        geometry=Geometry(kind=GeometryKind.BOX,size=hit.domain.size),
        mesh=Mesh(resolution=resolution,dx=derived.dx),
        solver=SolverConfig(kind=SolverKind.LBM,scheme=collision,reynolds_number=derived.reynolds_number,
            viscosity=derived.physical_viscosity,forcing=None if mode==HITMode.DECAYING else forcing_type,
            extra={"mach_number":mach_number,"actual_mach":derived.actual_mach,
                   "relaxation_time":derived.relaxation_time,
                   "lattice":"D3Q19","smagorinsky_constant":smagorinsky_constant,
                   "trt_magic_parameter":trt_magic_parameter,"density":density,
                   "char_velocity":char_velocity,"turbulence_regime":effective_regime}),
        flow=FlowConfig(kind=FlowKind.HIT,hit_mode=mode,initial_condition=ic_type,
            ic_wavenumber_min=ic_kmin,ic_wavenumber_max=ic_kmax,ic_seed=ic_seed,
            ic_spectrum_exponent=ic_spectrum_exponent,forcing_type=forcing_type,
            forcing_wavenumber_min=force_kmin if forcing_type!="none" else None,
            forcing_wavenumber_max=force_kmax if forcing_type!="none" else None,
            forcing_amplitude=forcing_amplitude,forcing_update_interval=forcing_update_interval,
            forcing_pattern=forcing_pattern,target_urms=target_urms,target_re_lambda=target_re_lambda,
            statistically_stationary=statistically_stationary,
            extra={"source_file":initial_condition_file,"correlation_time":forcing_correlation_time,
                   "target_injection_rate":target_injection_rate,"target_tke":target_tke}),
        boundary_conditions=[BoundaryCondition(region="all",type="periodic")],
        runtime=RuntimeParams(dt=derived.dt,max_steps=max_steps,output_interval=output_interval,
            checkpoint_interval=checkpoint_interval,num_procs=num_procs,extra={"diagnostics_interval":diagnostics_interval}),
        outputs=OutputConfig(write_velocity=True,write_pressure=write_pressure,write_vorticity=write_vorticity,
            sample_start_step=sample_start_step,sample_interval=output_interval,
            extra={"write_density":write_density,"write_forcing":write_forcing,"format":"vtm"}),
        execution=ExecutionConfig(mode=execution_mode,num_procs=num_procs,num_threads=num_threads,
            extra={"build_profile":build_profile}),
        metadata={"case_family":"FHIT" if mode==HITMode.FORCED else "DHIT","solver_target":"openlb",
                  "requested_parameters":hit.model_dump(mode="json"),"derived_parameters":derived.model_dump()})
