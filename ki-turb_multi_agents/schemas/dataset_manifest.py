"""Authoritative, solver-neutral manifest for raw and processed CFD data."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .unit_system import UnitSystem

SCHEMA_VERSION=2
STANDARD_FILE_KINDS=("velocity_field","pressure_field","density_field","vorticity_field","forcing_field","checkpoint","diagnostics","energy_spectrum","normalized_spectrum","spectral_isotropy","component_spectra","flatness","structure_functions","velocity_pdf","gradient_pdf","dissipation_pdf","enstrophy_pdf","joint_pdf","rq_pdf","turbulence_stats","reynolds_stress","anisotropy_invariants","dissipation_validation","energy_balance","analysis_products","tau_effective_field","figure","log","metadata")
def _utcnow(): return datetime.now(timezone.utc)

class DatasetFile(BaseModel):
    model_config=ConfigDict(extra="allow")
    path:str
    kind:str
    variable:Optional[str]=None
    format:Optional[str]=None
    time_step:Optional[int]=None
    time_value:Optional[float]=None
    size_bytes:Optional[int]=None
    checksum:Optional[str]=None
    complete:bool=True
    components:Optional[int]=None
    shape:Optional[Tuple[int,...]]=None
    spacing:Optional[Tuple[float,float,float]]=None
    origin:Optional[Tuple[float,float,float]]=None
    units:Optional[str]=None
    source_steps:List[int]=Field(default_factory=list)
    source_files:List[str]=Field(default_factory=list)
    metadata:Dict[str,Any]=Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid(self):
        if self.components is not None and self.components <= 0: raise ValueError("components must be positive")
        if self.size_bytes is not None and self.size_bytes < 0: raise ValueError("size_bytes cannot be negative")
        return self

class DatasetManifest(BaseModel):
    model_config=ConfigDict(extra="allow")
    schema_version:int=SCHEMA_VERSION
    manifest_id:str
    base_dir:str
    status:str="fetched"
    source_job_id:Optional[str]=None
    source_simulation:Optional[str]=None
    backend:Optional[str]=None
    solver_version:Optional[str]=None
    run_id:Optional[str]=None
    # Legacy flat kind→label map (kept for older readers).
    units:Dict[str,str]=Field(default_factory=dict)
    # Authoritative solver-neutral unit contract.
    unit_system:Optional[UnitSystem]=None
    time_steps:List[int]=Field(default_factory=list)
    files:List[DatasetFile]=Field(default_factory=list)
    created_at:datetime=Field(default_factory=_utcnow)
    requested_config:Dict[str,Any]=Field(default_factory=dict)
    effective_config:Dict[str,Any]=Field(default_factory=dict)
    measured:Dict[str,Any]=Field(default_factory=dict)
    case:Dict[str,Any]=Field(default_factory=dict)
    validation:Dict[str,Any]=Field(default_factory=dict)
    provenance:Dict[str,Any]=Field(default_factory=dict)
    postprocessing:Dict[str,Any]=Field(default_factory=dict)
    metadata:Dict[str,Any]=Field(default_factory=dict)

    def add_file(self,file:DatasetFile)->"DatasetManifest":
        # One authoritative entry per relative path.
        self.files=[f for f in self.files if f.path != file.path]
        self.files.append(file)
        if file.time_step is not None and file.time_step not in self.time_steps:
            self.time_steps.append(file.time_step); self.time_steps.sort()
        return self
    def files_of_kind(self,kind:str)->List[DatasetFile]: return [f for f in self.files if f.kind==kind]
    def files_of_variable(self,variable:str)->List[DatasetFile]: return [f for f in self.files if f.variable==variable]
    def require_complete(self)->None:
        bad=[f.path for f in self.files if not f.complete]
        if bad: raise ValueError("incomplete dataset files: "+", ".join(bad))
    def to_json(self)->str: return self.model_dump_json(indent=2)
    @classmethod
    def from_json(cls,data:str)->"DatasetManifest": return cls.model_validate_json(data)
