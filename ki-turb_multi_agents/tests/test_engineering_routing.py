"""Engineering workflow routing and capability/plan smoke tests."""
from __future__ import annotations

from pathlib import Path

from agents.knowledge.capability_loader import load_capability_context, match_capabilities
from agents.knowledge.lesson_store import LessonStore, format_lessons, record_lesson, retrieve_lessons
from agents.langgraph.engineering_intent import (
    engineering_workflow_plan,
    is_engineering_request,
    is_simple_file_edit_request,
    parse_engineering_intent,
)
from agents.langgraph.engineering_services import build_deterministic_plan
from agents.langgraph.models import EngineeringDiscovery, WorkflowPlan, WorkflowStep
from agents.langgraph.router import RequestRouter
from agents.tools.verify import execute_tool as verify_execute

ROOT = Path(__file__).resolve().parents[1]


def test_engineering_request_detection_for_pages_and_vtk():
    assert is_engineering_request("Inspect how pages work and make a plan to add a new analysis page")
    assert is_engineering_request("Wire KI-TURB to VTK export")
    assert is_engineering_request("Connect the remote runner to HPC GPUs")
    assert not is_engineering_request("plot energy spectra")
    assert not is_engineering_request("compile and run openlb hit 64^3")
    assert not is_engineering_request("modify examples/test.py add pope spectra and plot")
    assert is_simple_file_edit_request("do modifications in this file examples/test.py")


def test_router_returns_engineering_workflow_for_plan_only():
    text = "Inspect the plotting tools and make a plan to add a new plot page"
    intent = parse_engineering_intent(text, {})
    assert intent is not None
    assert intent.plan_only is True

    class _FakeEngPlanner:
        def invoke(self, payload):
            return {"structured_response": engineering_workflow_plan(text, intent)}

    router = RequestRouter(planner_agent=_FakeEngPlanner(), project_root=ROOT)
    # Engineering is planner-chosen, not a keyword hard gate.
    assert router.deterministic_plan(text, {}) is None
    plan = router.plan(text, {})
    assert plan.kind == "engineering_workflow"
    assert plan.steps[0].role == "engineer"


def test_continue_uses_existing_engineering_plan():
    summary = {"engineering_plan": {"goal": "x", "steps": []}}
    intent = parse_engineering_intent("continue", summary)
    assert intent is not None
    assert intent.continue_execution is True
    plan = engineering_workflow_plan("do step 2", intent)
    assert plan.kind == "engineering_workflow"


def test_capability_packs_match_and_load():
    caps = match_capabilities("add a streamlit analysis page", ROOT)
    assert "app_pages" in caps
    ctx = load_capability_context("add a streamlit analysis page", ROOT)
    assert ctx["primary_capability"] == "app_pages"
    assert "page_schema" in ctx["context"]


def test_deterministic_plan_mentions_page_schema():
    discoveries = [EngineeringDiscovery(file="agents/page_schema.py", role="schema")]
    plan = build_deterministic_plan(
        "Plan adding a new analysis page for custom metrics",
        capabilities=["app_pages"],
        discoveries=discoveries,
        plan_only=True,
    )
    assert plan.plan_only is True
    assert "agents/page_schema.py" in plan.modify
    assert plan.steps
    assert any("page_schema" in p for p in plan.modify)


def test_deterministic_plan_named_script_does_not_become_page():
    plan = build_deterministic_plan(
        "now modify test.py add a patch in which compute spectra with pope model "
        "spectra and then plotting save fig code lines in examples/test.py",
        capabilities=["plotting"],
        discoveries=[],
        plan_only=False,
    )
    assert plan.capability == "file_edit"
    assert "examples/test.py" in plan.modify
    assert "agents/page_schema.py" not in plan.modify
    assert "pages/NN_New_Page.py" not in plan.create


def test_router_demotes_engineering_for_examples_script():
    text = "do modifications in this file examples/test.py add pope spectra plot"

    class _FakeEngPlanner:
        def invoke(self, payload):
            return {
                "structured_response": WorkflowPlan(
                    kind="engineering_workflow",
                    steps=[WorkflowStep(role="engineer", instruction=text)],
                    rationale="biased",
                )
            }

    router = RequestRouter(planner_agent=_FakeEngPlanner(), project_root=ROOT)
    plan = router.plan(text, {})
    assert plan.kind == "agent_workflow"
    assert plan.steps[0].role == "steward"


def test_verify_translates_cat_to_python(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("x = 1\n", encoding="utf-8")
    out = verify_execute(
        "run_verify_command",
        {"command": f"cat {target.name}"},
        tmp_path,
    )
    assert "status: ok" in out
    assert "x = 1" in out


def test_verify_python_c_allows_semicolons(tmp_path):
    out = verify_execute(
        "run_verify_command",
        {"command": 'python -c "a=1; b=a+1; print(b)"'},
        tmp_path,
    )
    assert "status: ok" in out
    assert "2" in out


def test_run_verify_command_not_confirmable():
    from agents.runtime import tool_registry
    assert tool_registry.requires_confirmation("run_verify_command") is False
    assert tool_registry.requires_confirmation("write_file") is True


def test_lesson_store_roundtrip(tmp_path):
    record_lesson(
        tmp_path,
        task="wire plot tool",
        capability="plotting",
        symptoms="missing registry entry",
        fix="add tool to visualizer set",
        files=["agents/runtime/tool_registry.py"],
        verify=["pytest tests/agents/test_tool_registry_permissions.py -q"],
        reuse_when="plot tool permission missing",
        outcome="success",
    )
    lessons_path = tmp_path / "knowledge" / "lessons" / "lessons.jsonl"
    assert lessons_path.is_file()
    store = LessonStore(lessons_path)
    assert store.read_all()
    found = retrieve_lessons(tmp_path, "plot tool registry", capability="plotting", k=3)
    assert found
    assert "registry" in format_lessons(found).lower()
