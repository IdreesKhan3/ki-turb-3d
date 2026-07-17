"""Typed planning and response models for the single KI-TURB workflow engine."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

RoleName = Literal[
    "orchestrator", "steward", "simulation", "analyst", "visualizer", "reviewer", "engineer"
]
WorkflowKind = Literal["hit_workflow", "agent_workflow", "engineering_workflow"]


class WorkflowStep(BaseModel):
    # KI_TURB_DIRECT_TOOL_STEPS_V1
    model_config = ConfigDict(extra="forbid")
    role: RoleName
    instruction: str = Field(min_length=1)
    tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: WorkflowKind = "agent_workflow"
    steps: list[WorkflowStep] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_steps(self):
        if self.kind == "agent_workflow" and not self.steps:
            self.steps = [WorkflowStep(role="orchestrator", instruction="Answer the user request directly.")]
        if self.kind == "engineering_workflow" and not self.steps:
            self.steps = [WorkflowStep(role="engineer", instruction="Discover, plan, and execute the engineering request.")]
        return self


class EngineeringDiscovery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str = Field(min_length=1)
    role: str = ""


class EngineeringStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    create: list[str] = Field(default_factory=list)
    modify: list[str] = Field(default_factory=list)
    verify: list[str] = Field(default_factory=list)


class EngineeringPlan(BaseModel):
    """File-level engineering plan produced before any mutating edits."""

    model_config = ConfigDict(extra="forbid")
    goal: str = Field(min_length=1)
    capability: str = ""
    capabilities: list[str] = Field(default_factory=list)
    discoveries: list[EngineeringDiscovery] = Field(default_factory=list)
    create: list[str] = Field(default_factory=list)
    modify: list[str] = Field(default_factory=list)
    do_not_touch: list[str] = Field(default_factory=list)
    verify: list[str] = Field(default_factory=list)
    steps: list[EngineeringStep] = Field(default_factory=list)
    plan_only: bool = False
    rationale: str = ""


class WorkflowSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "RoleName",
    "WorkflowKind",
    "WorkflowStep",
    "WorkflowPlan",
    "EngineeringDiscovery",
    "EngineeringStep",
    "EngineeringPlan",
    "WorkflowSummary",
]
