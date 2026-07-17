"""KI-TURB agents: a single LangChain/LangGraph runtime plus deterministic tools."""
__all__ = ["UnifiedTeam", "KITurbGraphEngine", "LangGraphWorkflowEngine", "get_analysis_intent", "get_plot_routing"]


def __getattr__(name):
    if name == "UnifiedTeam":
        from .team_manager import UnifiedTeam
        return UnifiedTeam
    if name in {"KITurbGraphEngine", "LangGraphWorkflowEngine"}:
        from .langgraph import KITurbGraphEngine, LangGraphWorkflowEngine
        return {"KITurbGraphEngine": KITurbGraphEngine, "LangGraphWorkflowEngine": LangGraphWorkflowEngine}[name]
    if name in {"get_analysis_intent", "get_plot_routing"}:
        from .intent_detection import get_analysis_intent, get_plot_routing
        return {"get_analysis_intent": get_analysis_intent, "get_plot_routing": get_plot_routing}[name]
    raise AttributeError(name)
