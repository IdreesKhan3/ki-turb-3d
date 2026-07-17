"""General file-inspection routing: no README hardcoding; locate then read."""
from __future__ import annotations

from agents.langgraph.models import WorkflowPlan, WorkflowStep
from agents.langgraph.router import RequestRouter, _sanitize_plan


def test_docs_requests_go_to_freeform_steward_not_hardcoded_readme_tool():
    router = RequestRouter(planner_agent=None)
    plan = router.plan(
        "check the ki turb 3d README.md is correct and updated and fully inline "
        "with the actual tools functionalities? check carefully so i can later publish my repo to git",
        {},
    )
    assert plan.steps
    first = plan.steps[0]
    assert first.role == "steward"
    # Must not hardcode a direct README.md tool call; free-form steward (LLM or fallback).
    assert first.tool is None
    assert plan.rationale.startswith("Free-form steward")


def test_sanitize_fills_filepath_from_any_path_in_text():
    bad = WorkflowPlan(
        steps=[
            WorkflowStep(
                role="steward",
                instruction="Read agents/runtime/tool_registry.py carefully",
                tool="read_document",
                tool_args={},
            )
        ]
    )
    fixed = _sanitize_plan(bad, "inspect tool_registry.py")
    assert fixed.steps[0].tool == "read_file"
    assert fixed.steps[0].tool_args.get("filepath") == "agents/runtime/tool_registry.py"


def test_sanitize_missing_path_becomes_locate_then_read():
    bad = WorkflowPlan(
        steps=[
            WorkflowStep(
                role="steward",
                instruction="Read the project documentation",
                tool="read_document",
                tool_args={},
            )
        ]
    )
    fixed = _sanitize_plan(bad, "read the project documentation")
    assert fixed.steps[0].tool is None
    assert "find_file" in fixed.steps[0].instruction.lower()


def test_sanitize_demotes_simulation_file_tools_to_steward():
    bad = WorkflowPlan(
        steps=[
            WorkflowStep(
                role="simulation",
                instruction="Read LICENSE",
                tool="read_file",
                tool_args={"filepath": "LICENSE"},
            )
        ]
    )
    fixed = _sanitize_plan(bad, "read LICENSE")
    assert fixed.steps[0].role == "steward"


def test_find_any_named_path_routes_to_steward():
    router = RequestRouter(planner_agent=None)
    plan = router.plan("find LICENSE and summarize it", {})
    assert plan.steps[0].role == "steward"
    assert plan.steps[0].tool is None
    assert router.deterministic_plan("find LICENSE and summarize it") is None


def test_list_path_routes_to_steward():
    router = RequestRouter(planner_agent=None)
    plan = router.plan("list agents/tools and tell me what packages exist", {})
    assert plan.steps[0].role == "steward"


def test_open_source_path_not_stolen_by_lifecycle_hard_gate():
    router = RequestRouter(planner_agent=None)
    plan = router.plan("open agents/page_schema.py and explain PAGE_SCHEMA", {})
    assert plan.steps[0].role == "steward"
    assert plan.rationale.startswith("Free-form steward")
    assert router.deterministic_plan("open agents/page_schema.py and explain PAGE_SCHEMA") is None
