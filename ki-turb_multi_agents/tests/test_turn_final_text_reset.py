"""Same-thread follow-ups must not replay the previous turn's final_text."""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain")

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Overwrite

from agents.langgraph.app_graph import ROLES, build_app_graph
from agents.langgraph.models import WorkflowPlan, WorkflowStep
from agents.langgraph.router import RequestRouter


class _HitStub:
    settings = None
    role_factory = None

    def normalize_request(self, state):
        return {}

    def validate_physics(self, state):
        return {}

    def approval(self, state):
        return {}

    def prepare(self, state):
        return {}

    def compile(self, state):
        return {}

    def run_collect(self, state):
        return {}

    def analyse(self, state):
        return {}

    def finalize(self, state):
        return {}

    def summarize(self, state):
        return {"final_text": "hit"}


class _WriteThenDeleteRouter(RequestRouter):
    def __init__(self):
        super().__init__(planner_agent=None)
        self._n = 0

    def plan(self, request, session_summary=None):
        self._n += 1
        if self._n == 1:
            return WorkflowPlan(
                steps=[WorkflowStep(role="steward", instruction="pretend write")],
                rationale="write turn",
            )
        return WorkflowPlan(
            steps=[
                WorkflowStep(
                    role="steward",
                    instruction="delete the file",
                    tool="delete_file",
                    tool_args={"filepath": "examples/does_not_need_to_exist_for_this_test.py"},
                )
            ],
            rationale="delete turn",
        )


def _turn_input(user_request: str) -> dict:
    return {
        "user_request": user_request,
        "messages": [{"role": "user", "content": user_request}],
        "session_summary": {},
        "final_text": "",
        "task_results": Overwrite([]),
        "artifacts": Overwrite([]),
        "warnings": Overwrite([]),
        "errors": Overwrite([]),
        "events": Overwrite([]),
        "task_index": 0,
        "plan": {},
    }


def test_second_turn_final_text_is_not_previous_answer(tmp_path):
    """Reproduce: write answer sticks after a later delete on the same thread_id."""

    def _writer(state):
        req = str(state.get("user_request") or "").lower()
        # Recovery may hand a failed delete to free-form steward — answer THIS turn.
        if "remove" in req or "delete" in req:
            return {"messages": [AIMessage(content="delete_file: path not found or already removed")]}
        return {"messages": [AIMessage(content="## All validations passed\nold write summary")]}

    def _other(_state):
        return {"messages": [AIMessage(content="ok")]}

    agents = {role: (_writer if role == "steward" else _other) for role in ROLES}
    graph = build_app_graph(
        router=_WriteThenDeleteRouter(),
        role_agents=agents,
        hit_services=_HitStub(),
        checkpointer=InMemorySaver(),
        project_root=tmp_path,
        session_context={},
    )
    cfg = {"configurable": {"thread_id": "stale-final-text"}}

    first = graph.invoke(_turn_input("write a script"), cfg)
    assert "All validations passed" in (first.get("final_text") or "")

    second = graph.invoke(_turn_input("now remove that file"), cfg)
    text = second.get("final_text") or ""
    assert "All validations passed" not in text
    assert "old write summary" not in text
    assert text.strip()
    assert "delete" in text.lower() or "not found" in text.lower() or "Error" in text
