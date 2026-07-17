"""
Autonomous Lab module — UI for the seven-role agent chat (Orchestrator, Steward,
Simulation, Analyst, Visualizer, Reviewer, Engineer).
"""

from .styles import inject_chat_styles
from .session_context import build_session_context
from .confirmation import (
    auto_choice_for_duplicate_pending,
    init_confirmation_state,
    render_tool_confirmation_ui,
    handle_tool_confirmation_resume,
    render_retrieve_ui,
    render_revert_ui,
)
from .chat_toolbar import render_chat_toolbar
from .simulation_workflow import render_simulation_workflow

__all__ = [
    "inject_chat_styles",
    "build_session_context",
    "auto_choice_for_duplicate_pending",
    "init_confirmation_state",
    "render_tool_confirmation_ui",
    "handle_tool_confirmation_resume",
    "render_retrieve_ui",
    "render_revert_ui",
    "render_chat_toolbar",
    "render_simulation_workflow",
]
