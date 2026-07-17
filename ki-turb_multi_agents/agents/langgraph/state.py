"""One serializable state contract for every KI-TURB workflow."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

WorkflowStatus = Literal[
    "created", "planned", "awaiting_approval", "running", "completed",
    "parsed", "validated", "approved", "prepared", "built", "fetched",
    "analysed", "visualized", "reviewed", "accepted", "rejected",
    "insufficient_data", "failed", "cancelled",
]


class KITurbState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    workflow_version: int
    thread_id: str
    user_request: str
    chat_history: list[dict[str, Any]]
    session_summary: dict[str, Any]
    intent_override_text: str
    prevent_tools: list[str]
    plan: dict[str, Any]
    task_index: int
    active_role: str
    active_tool: str
    active_tool_args: Dict[str, Any]
    message_cursor: int
    task_results: Annotated[List[Dict[str, Any]], operator.add]
    status: WorkflowStatus
    final_text: str
    artifacts: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    events: Annotated[List[Dict[str, Any]], operator.add]
    metadata: Dict[str, Any]

    # HIT workflow fields
    requested_config: Dict[str, Any]
    derived_config: Dict[str, Any]
    effective_config: Dict[str, Any]
    measured: Dict[str, Any]
    physics_report: Dict[str, Any]
    capability_report: Dict[str, Any]
    approved: bool
    require_approval: bool
    run_root: str
    openlb_app_dir: str
    session_path: str
    run_id: str
    manifest_path: str
    analysis_products_path: str
    validation_path: str
    dashboard_path: str
    report_path: str

    # Engineering workflow fields
    engineering_plan: Dict[str, Any]
    engineering_step_index: int
    engineering_capability: str
    engineering_context: str
    engineering_discoveries: List[Dict[str, Any]]
    engineering_last_step_result: str
    engineering_verify_ok: bool
    engineering_repair_attempts: int


# Compatibility alias for existing HIT-specific imports.
HITWorkflowState = KITurbState

__all__ = ["KITurbState", "HITWorkflowState", "WorkflowStatus"]
