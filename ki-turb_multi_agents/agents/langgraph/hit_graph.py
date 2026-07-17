"""Durable OpenLB HIT subgraph."""
from __future__ import annotations

from .hit_services import HITGraphServices
from .state import KITurbState


def _failed(state: KITurbState) -> str:
    return "stop" if state.get("errors") or state.get("status") in {"failed", "rejected", "cancelled"} else "continue"


def build_hit_subgraph(services: HITGraphServices):
    from langgraph.graph import END, START, StateGraph
    graph = StateGraph(KITurbState)
    graph.add_node("parse_hit_request", services.normalize_request)
    graph.add_node("validate_hit_physics", services.validate_physics)
    graph.add_node("approve_hit_execution", services.approval)
    graph.add_node("prepare_hit_case", services.prepare)
    graph.add_node("compile_openlb", services.compile)
    graph.add_node("run_collect_hit", services.run_collect)
    graph.add_node("analyse_hit", services.analyse)
    graph.add_node("visualize_review_hit", services.finalize)
    graph.add_node("summarize_hit", services.summarize)
    graph.add_edge(START, "parse_hit_request")
    for source, target in [
        ("parse_hit_request", "validate_hit_physics"),
        ("validate_hit_physics", "approve_hit_execution"),
        ("approve_hit_execution", "prepare_hit_case"),
        ("prepare_hit_case", "compile_openlb"),
        ("compile_openlb", "run_collect_hit"),
        ("run_collect_hit", "analyse_hit"),
        ("analyse_hit", "visualize_review_hit"),
    ]:
        graph.add_conditional_edges(source, _failed, {"continue": target, "stop": "summarize_hit"})
    graph.add_edge("visualize_review_hit", "summarize_hit")
    graph.add_edge("summarize_hit", END)
    return graph.compile(name="kiturb_openlb_hit")


__all__ = ["build_hit_subgraph"]
