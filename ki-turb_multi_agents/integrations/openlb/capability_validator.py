"""Versioned, strict OpenLB HIT capability validation."""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any,Dict,List,Optional,Set
from pydantic import BaseModel,ConfigDict,Field
from schemas.openlb_hit import *
class CapabilityStatus(str,Enum):SUPPORTED="supported"; EXPERIMENTAL="experimental"; UNSUPPORTED="unsupported"
class UnsupportedOpenLBCapability(ValueError):pass
class CapabilityDecision(BaseModel):
    model_config=ConfigDict(extra="allow")
    supported:bool; experimental:bool=False; errors:List[str]=Field(default_factory=list); warnings:List[str]=Field(default_factory=list); requested:Dict[str,Any]=Field(default_factory=dict); effective:Dict[str,Any]=Field(default_factory=dict)
class OpenLBHITCapabilities(BaseModel):
    model_config=ConfigDict(extra="allow")
    schema_version:int=2; solver:str="openlb"; solver_version:str="legacy-patched"; app:str="kiTurbHIT3D"
    lattices:Dict[str,CapabilityStatus]=Field(default_factory=dict); collision_models:Dict[str,CapabilityStatus]=Field(default_factory=dict)
    forcing_models:Dict[str,CapabilityStatus]=Field(default_factory=dict); build_profiles:Dict[str,CapabilityStatus]=Field(default_factory=dict)
    collision_forcing:Dict[str,List[str]]=Field(default_factory=dict); output_formats:Dict[str,CapabilityStatus]=Field(default_factory=dict)
    initial_conditions:Dict[str,CapabilityStatus]=Field(default_factory=dict); constraints:Dict[str,Any]=Field(default_factory=dict)
    @classmethod
    def conservative_default(cls):
        none=[HITForcingType.NONE.value]
        all_forcing=[m.value for m in HITForcingType]
        supported={
            HITCollisionModel.BGK.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.TRT.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.MRT.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.RLB.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.SMAGORINSKY_BGK.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.WALE.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.SHEAR_SMAGORINSKY.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.CONSISTENT_STRAIN_SMAGORINSKY.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.KRAUSE.value:CapabilityStatus.SUPPORTED,
            HITCollisionModel.SMAGORINSKY_MRT.value:CapabilityStatus.UNSUPPORTED,
            HITCollisionModel.DYNAMIC_SMAGORINSKY.value:CapabilityStatus.EXPERIMENTAL,
        }
        mapping={
            HITCollisionModel.BGK.value:all_forcing,
            HITCollisionModel.TRT.value:all_forcing,
            HITCollisionModel.MRT.value:all_forcing,
            HITCollisionModel.RLB.value:none,
            HITCollisionModel.SMAGORINSKY_BGK.value:all_forcing,
            HITCollisionModel.WALE.value:all_forcing,
            HITCollisionModel.SHEAR_SMAGORINSKY.value:all_forcing,
            HITCollisionModel.CONSISTENT_STRAIN_SMAGORINSKY.value:none,
            HITCollisionModel.KRAUSE.value:none,
            HITCollisionModel.SMAGORINSKY_MRT.value:[],
            HITCollisionModel.DYNAMIC_SMAGORINSKY.value:none,
        }
        return cls(
          lattices={"D3Q19":CapabilityStatus.SUPPORTED,"D3Q27":CapabilityStatus.UNSUPPORTED},
          collision_models=supported,
          forcing_models={m.value:CapabilityStatus.SUPPORTED for m in HITForcingType},
          collision_forcing=mapping,
          build_profiles={"serial":CapabilityStatus.SUPPORTED,"openmp":CapabilityStatus.SUPPORTED,"mpi":CapabilityStatus.SUPPORTED,"mpi_openmp":CapabilityStatus.SUPPORTED,"debug":CapabilityStatus.SUPPORTED,"sanitizer":CapabilityStatus.SUPPORTED,"cuda":CapabilityStatus.UNSUPPORTED,"hip":CapabilityStatus.UNSUPPORTED},
          output_formats={"vtm":CapabilityStatus.SUPPORTED,"vti":CapabilityStatus.SUPPORTED,"hdf5":CapabilityStatus.UNSUPPORTED,"xdmf_hdf5":CapabilityStatus.UNSUPPORTED},
          initial_conditions={"synthetic_spectrum":CapabilityStatus.SUPPORTED,"restart":CapabilityStatus.EXPERIMENTAL,"imported_field":CapabilityStatus.EXPERIMENTAL},
          constraints={"uniform_spacing":True,"periodic":True,"forcing_nyquist_fraction":0.5})
    @classmethod
    def load(cls,path):return cls.model_validate_json(Path(path).read_text())
    def save(self,path):p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(self.model_dump_json(indent=2));return p
class OpenLBHITCapabilityValidator:
    def __init__(self,capabilities:Optional[OpenLBHITCapabilities]=None):self.capabilities=capabilities or OpenLBHITCapabilities.conservative_default()
    def validate(self,config:OpenLBHITConfig)->CapabilityDecision:
        c=self.capabilities; errors=[]; warnings=[]
        requested={"lattice":config.domain.lattice.value,"collision":config.collision.model.value,"forcing":config.forcing.type.value,"initial_condition":config.initial_condition.type.value,"build_profile":config.execution.build_profile.value,"output_format":config.outputs.format}
        for label,key,mapping in [("lattice",requested["lattice"],c.lattices),("collision",requested["collision"],c.collision_models),("forcing",requested["forcing"],c.forcing_models),("initial condition",requested["initial_condition"],c.initial_conditions),("build profile",requested["build_profile"],c.build_profiles),("output format",requested["output_format"],c.output_formats)]:
            status=mapping.get(key,CapabilityStatus.UNSUPPORTED)
            if status==CapabilityStatus.UNSUPPORTED:errors.append(f"{label} '{key}' is unsupported")
            elif status==CapabilityStatus.EXPERIMENTAL:warnings.append(f"{label} '{key}' is experimental")
        if requested['forcing'] not in set(c.collision_forcing.get(requested['collision'],[])):errors.append(f"forcing '{requested['forcing']}' is not available with exact collision '{requested['collision']}'")
        dx=[l/n for l,n in zip(config.domain.size,config.domain.resolution)]
        if max(dx)-min(dx)>1e-12*max(dx):errors.append("OpenLB HIT requires uniform grid spacing")
        if not all(config.domain.periodic):errors.append("OpenLB HIT requires periodic boundaries")
        k_limit=min(config.domain.resolution)//2-1
        for label,k in [("initial-condition",config.initial_condition.wavenumber_max),("forcing",config.forcing.wavenumber_max)]:
            if k is not None and k>k_limit:errors.append(f"{label} k_max={k} exceeds resolvable limit {k_limit}")
        return CapabilityDecision(supported=not errors,experimental=bool(warnings),errors=errors,warnings=warnings,requested=requested,effective=dict(requested))
    def assert_supported(self,config):
        d=self.validate(config)
        if not d.supported:raise UnsupportedOpenLBCapability('; '.join(d.errors))
        return d
