"""Durable simulation state machine used by all execution backends."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field
SCHEMA_VERSION=2
def _utcnow(): return datetime.now(timezone.utc)
class JobStatus(str,Enum):
    CREATED="created"; PENDING="pending"; PREPARED="prepared"; VALIDATED="validated"
    BUILDING="building"; COMPILING="compiling"; BUILT="built"; COMPILED="compiled"
    QUEUED="queued"; SUBMITTED="submitted"; RUNNING="running"; CHECKPOINTING="checkpointing"
    COMPLETED="completed"; FAILED="failed"; CANCELLED="cancelled"; FETCHING="fetching"; FETCHED="fetched"
    POSTPROCESSING="postprocessing"; POSTPROCESSED="postprocessed"; ANALYSING="analysing"
    ANALYSED="analysed"; ANALYSIS_READY="analysis_ready"; INSUFFICIENT_DATA="insufficient_data"; VISUALISING="visualising"; VISUALIZED="visualized"
    REVIEWED="reviewed"; ACCEPTED="accepted"; REJECTED="rejected"
    @property
    def is_terminal(self): return self in {self.COMPLETED,self.FAILED,self.CANCELLED,self.ACCEPTED,self.REJECTED,self.ANALYSIS_READY,self.INSUFFICIENT_DATA}
class JobPaths(BaseModel):
    model_config=ConfigDict(extra="forbid")
    case_dir:Optional[str]=None; build_dir:Optional[str]=None; output_dir:Optional[str]=None
    raw_dir:Optional[str]=None; processed_dir:Optional[str]=None; figures_dir:Optional[str]=None
    report_dir:Optional[str]=None; log_path:Optional[str]=None; checkpoint_dir:Optional[str]=None
class SimulationJob(BaseModel):
    model_config=ConfigDict(extra="allow")
    schema_version:int=SCHEMA_VERSION; job_id:str; backend:str; case_name:Optional[str]=None
    status:JobStatus=JobStatus.PENDING; paths:JobPaths=Field(default_factory=JobPaths)
    external_id:Optional[str]=None; parent_job_id:Optional[str]=None; return_code:Optional[int]=None
    message:str=""; progress:Optional[float]=None
    created_at:datetime=Field(default_factory=_utcnow); submitted_at:Optional[datetime]=None
    started_at:Optional[datetime]=None; finished_at:Optional[datetime]=None
    resources:Dict[str,Any]=Field(default_factory=dict); solver_version:Optional[str]=None
    solver_commit:Optional[str]=None; container_image:Optional[str]=None
    requested_config:Dict[str,Any]=Field(default_factory=dict); effective_config:Dict[str,Any]=Field(default_factory=dict)
    measured:Dict[str,Any]=Field(default_factory=dict); metadata:Dict[str,Any]=Field(default_factory=dict)
    def mark(self,status:JobStatus,*,message:str="",return_code:Optional[int]=None):
        self.status=status
        if message:self.message=message
        if return_code is not None:self.return_code=return_code
        now=_utcnow()
        if status in {JobStatus.QUEUED,JobStatus.SUBMITTED} and self.submitted_at is None:self.submitted_at=now
        if status==JobStatus.RUNNING and self.started_at is None:self.started_at=now
        if status.is_terminal and self.finished_at is None:self.finished_at=now
        return self
    def to_json(self):return self.model_dump_json(indent=2)
    @classmethod
    def from_json(cls,data):return cls.model_validate_json(data)
