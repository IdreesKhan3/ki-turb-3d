"""Backend-neutral CFD case schema with an optional typed HIT contract."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from .openlb_hit import OpenLBHITConfig
from .unit_system import UnitSystem

SCHEMA_VERSION = 2

class GeometryKind(str, Enum):
    BOX="box"; CHANNEL="channel"; CYLINDER="cylinder"; CUSTOM_MESH="custom_mesh"
class Geometry(BaseModel):
    model_config=ConfigDict(extra="forbid")
    kind: GeometryKind=GeometryKind.BOX
    size: Tuple[float,float,float]=(1.0,1.0,1.0)
    mesh_file: Optional[str]=None
    notes: Optional[str]=None
class Mesh(BaseModel):
    model_config=ConfigDict(extra="forbid")
    resolution: Tuple[int,int,int]=(128,128,128)
    dx: Optional[float]=None
    refinement_level:int=0
    extra:Dict[str,Any]=Field(default_factory=dict)
class SolverKind(str, Enum):
    LBM="lbm"; NAVIER_STOKES="navier_stokes"; SPECTRAL="spectral"
class SolverConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    kind:SolverKind=SolverKind.LBM
    scheme:Optional[str]=None
    reynolds_number:Optional[float]=None
    viscosity:Optional[float]=None
    forcing:Optional[str]=None
    # Deprecated compatibility channel. New OpenLB HIT cases use ``hit``.
    extra:Dict[str,Any]=Field(default_factory=dict)
class BoundaryCondition(BaseModel):
    model_config=ConfigDict(extra="forbid")
    region:str
    type:str
    value:Optional[Any]=None
class FlowKind(str, Enum):
    HIT="hit"; CHANNEL="channel"; CYLINDER="cylinder"; CUSTOM="custom"
class HITMode(str, Enum):
    DECAYING="decaying"; FORCED="forced"

def normalize_hit_mode(value: Any) -> str:
    """Map FHIT/DHIT aliases to schema values."""
    aliases = {"fhit": HITMode.FORCED.value, "dhit": HITMode.DECAYING.value}
    key = str(value or HITMode.FORCED.value).strip().lower()
    return aliases.get(key, key)
class FlowConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    kind:FlowKind=FlowKind.HIT
    hit_mode:Optional[HITMode]=None
    initial_condition:Optional[str]=None
    ic_wavenumber_min:Optional[int]=None
    ic_wavenumber_max:Optional[int]=None
    ic_seed:Optional[int]=None
    ic_spectrum_exponent:Optional[float]=None
    forcing_type:Optional[str]=None
    forcing_wavenumber_min:Optional[int]=None
    forcing_wavenumber_max:Optional[int]=None
    forcing_amplitude:Optional[float]=None
    forcing_update_interval:Optional[int]=None
    forcing_pattern:Optional[str]=None
    target_urms:Optional[float]=None
    target_re_lambda:Optional[float]=None
    statistically_stationary:bool=True
    extra:Dict[str,Any]=Field(default_factory=dict)
class OutputConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    write_velocity:bool=True
    write_pressure:bool=False
    write_vorticity:bool=False
    write_spectra:bool=True
    write_isotropy:bool=True
    write_flatness:bool=True
    write_structure_functions:bool=True
    write_pdfs:bool=True
    sample_start_step:Optional[int]=None
    sample_interval:int=1000
    extra:Dict[str,Any]=Field(default_factory=dict)
class ExecutionConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    mode:str="local"
    num_procs:int=1
    num_threads:int=1
    memory_gb:Optional[float]=None
    walltime:Optional[str]=None
    container_image:Optional[str]=None
    queue:Optional[str]=None
    extra:Dict[str,Any]=Field(default_factory=dict)
class RuntimeParams(BaseModel):
    model_config=ConfigDict(extra="forbid")
    dt:Optional[float]=None
    max_steps:Optional[int]=None
    max_time:Optional[float]=None
    output_interval:int=1000
    checkpoint_interval:Optional[int]=None
    num_procs:int=1
    extra:Dict[str,Any]=Field(default_factory=dict)
class CFDCase(BaseModel):
    model_config=ConfigDict(extra="forbid")
    schema_version:int=SCHEMA_VERSION
    name:str
    description:Optional[str]=None
    geometry:Geometry=Field(default_factory=Geometry)
    mesh:Mesh=Field(default_factory=Mesh)
    solver:SolverConfig=Field(default_factory=SolverConfig)
    boundary_conditions:List[BoundaryCondition]=Field(default_factory=list)
    runtime:RuntimeParams=Field(default_factory=RuntimeParams)
    flow:FlowConfig=Field(default_factory=FlowConfig)
    outputs:OutputConfig=Field(default_factory=OutputConfig)
    execution:ExecutionConfig=Field(default_factory=ExecutionConfig)
    hit:Optional[OpenLBHITConfig]=None
    # Solver-neutral unit contract (filled by backend adapters).
    units:Optional[UnitSystem]=None
    metadata:Dict[str,Any]=Field(default_factory=dict)
    def typed_hit(self)->OpenLBHITConfig:
        return OpenLBHITConfig.from_cfd_case(self)
    def to_json(self)->str: return self.model_dump_json(indent=2)
    @classmethod
    def from_json(cls,data:str)->"CFDCase": return cls.model_validate_json(data)
