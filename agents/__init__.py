"""
Agents package — 5 LLM-driven agents with tools.
"""

from .team_manager import UnifiedTeam
from .routing import route, is_analysis_intent, is_explain_intent
from .writer import WriterAgent, is_writing_intent
from .intent_detection import get_analysis_intent, get_plot_routing

__all__ = [
    "UnifiedTeam",
    "route",
    "is_analysis_intent",
    "is_explain_intent",
    "WriterAgent",
    "is_writing_intent",
    "get_analysis_intent",
    "get_plot_routing",
]
