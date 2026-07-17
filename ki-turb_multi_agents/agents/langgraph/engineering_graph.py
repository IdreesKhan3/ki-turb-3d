"""Plan-gated engineering subgraph for platform self-improvement."""
from __future__ import annotations

from .engineering_services import EngineeringGraphServices
from .state import KITurbState


def _failed(state: KITurbState) -> str:
    return "stop" if state.get("errors") or state.get("status") in {"failed", "rejected", "cancelled", "completed"} else "continue"


def _after_approve(state: KITurbState) -> str:
    if state.get("errors") or state.get("status") in {"failed", "rejected", "cancelled", "completed"}:
        return "stop"
    if not state.get("approved") and not state.get("engineering_plan"):
        return "stop"
    # plan_only path sets status completed without approved
    if state.get("status") == "completed":
        return "stop"
    if state.get("approved"):
        return "continue"
    return "stop"


def _after_verify(state: KITurbState) -> str:
    if state.get("errors") or state.get("status") in {"failed", "rejected", "cancelled"}:
        return "stop"
    if state.get("engineering_verify_ok"):
        return "advance"
    return "repair"


def _after_advance(state: KITurbState) -> str:
    if state.get("errors") or state.get("status") in {"failed", "rejected", "cancelled", "completed"}:
        return "stop"
    return "continue"


def build_engineering_subgraph(services: EngineeringGraphServices):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(KITurbState)
    graph.add_node("eng_discover", services.discover)
    graph.add_node("eng_draft_plan", services.draft_plan)
    graph.add_node("eng_approve_plan", services.approve_plan)
    graph.add_node("eng_execute_step", services.execute_step)
    graph.add_node("eng_verify_step", services.verify_step)
    graph.add_node("eng_repair_step", services.repair_step)
    graph.add_node("eng_advance", services.advance_or_finish)
    graph.add_node("eng_finalize", services.finalize)

    graph.add_edge(START, "eng_discover")
    graph.add_conditional_edges(
        "eng_discover",
        _failed,
        {"continue": "eng_draft_plan", "stop": "eng_finalize"},
    )
    graph.add_conditional_edges(
        "eng_draft_plan",
        _failed,
        {"continue": "eng_approve_plan", "stop": "eng_finalize"},
    )
    graph.add_conditional_edges(
        "eng_approve_plan",
        _after_approve,
        {"continue": "eng_execute_step", "stop": "eng_finalize"},
    )
    graph.add_conditional_edges(
        "eng_execute_step",
        _failed,
        {"continue": "eng_verify_step", "stop": "eng_finalize"},
    )
    graph.add_conditional_edges(
        "eng_verify_step",
        _after_verify,
        {"advance": "eng_advance", "repair": "eng_repair_step", "stop": "eng_finalize"},
    )
    graph.add_conditional_edges(
        "eng_repair_step",
        _failed,
        {"continue": "eng_verify_step", "stop": "eng_finalize"},
    )
    graph.add_conditional_edges(
        "eng_advance",
        _after_advance,
        {"continue": "eng_execute_step", "stop": "eng_finalize"},
    )
    graph.add_edge("eng_finalize", END)
    return graph.compile(name="kiturb_engineering")


__all__ = ["build_engineering_subgraph"]
