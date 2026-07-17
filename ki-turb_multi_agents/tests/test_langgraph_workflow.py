from __future__ import annotations

import pytest
pytest.importorskip("langgraph")
pytest.importorskip("langchain")

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.langgraph.app_graph import ROLES, build_app_graph
from agents.langgraph.hit_graph import build_hit_subgraph
from agents.langgraph.models import WorkflowPlan, WorkflowStep
from agents.langgraph.router import RequestRouter


class FakeHitServices:
    def normalize_request(self, state): return {"requested_config": {"name": "fake"}, "status": "parsed"}
    def validate_physics(self, state): return {"derived_config": {"actual_mach": 0.05}, "status": "validated"}
    def approval(self, state):
        from langgraph.types import interrupt
        decision = interrupt({"message": "approve fake run"})
        return {"approved": bool(decision.get("approved") if isinstance(decision, dict) else decision), "status": "approved"}
    def prepare(self, state): return {"session_path": "/tmp/session.json", "run_id": "fake-1", "status": "prepared"}
    def compile(self, state): return {"effective_config": {"collision": "BGK"}, "status": "built"}
    def run_collect(self, state): return {"manifest_path": "/tmp/manifest.json", "status": "fetched"}
    def analyse(self, state): return {"analysis_products_path": "/tmp/products.json", "status": "analysed"}
    def finalize(self, state): return {"report_path": "/tmp/report.html", "status": "accepted", "artifacts": []}
    def summarize(self, state): return {"final_text": f"finished {state['status']}"}


def test_hit_subgraph_interrupts_and_resumes():
    graph = build_hit_subgraph(FakeHitServices()).with_config({"configurable": {"thread_id": "test-hit"}})
    # Embed it in a parent so the parent owns persistence, matching production.
    from langgraph.graph import START, END, StateGraph
    from agents.langgraph.state import KITurbState
    parent = StateGraph(KITurbState)
    parent.add_node("hit", graph)
    parent.add_edge(START, "hit")
    parent.add_edge("hit", END)
    compiled = parent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "test-hit"}}
    interrupted = compiled.invoke({"user_request": "run hit", "warnings": [], "errors": [], "events": [], "artifacts": []}, cfg)
    assert interrupted["__interrupt__"][0].value["message"] == "approve fake run"
    result = compiled.invoke(Command(resume={"approved": True}), cfg)
    assert result["status"] == "accepted"
    assert result["final_text"] == "finished accepted"


class FixedRouter(RequestRouter):
    def plan(self, request, session_summary):
        return WorkflowPlan(steps=[WorkflowStep(role="analyst", instruction="analyse")])


def _fake_agent(state):
    return {"messages": [AIMessage(content="analysis complete")]}


def test_root_graph_routes_all_requests_without_legacy_runner():
    agents = {role: _fake_agent for role in ROLES}
    graph = build_app_graph(
        router=FixedRouter(None),
        role_agents=agents,
        hit_services=FakeHitServices(),
        checkpointer=InMemorySaver(),
        project_root=".",
        session_context={},
    )
    result = graph.invoke({"user_request": "analyse data", "messages": [{"role": "user", "content": "analyse data"}], "session_summary": {}, "warnings": [], "errors": [], "events": [], "artifacts": [], "task_results": []}, {"configurable": {"thread_id": "root"}})
    assert result["status"] == "completed"
    assert result["final_text"] == "analysis complete"
    assert result["task_results"][0]["role"] == "analyst"
