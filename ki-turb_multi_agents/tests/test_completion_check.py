"""Pre-finalize completion self-check."""
from __future__ import annotations

from agents.langgraph.completion_check import (
    MAX_COMPLETION_ATTEMPTS,
    evaluate_completion,
    exhaustion_message,
    finish_work_plan,
)
from agents.langgraph.app_graph import AppGraphNodes
from agents.langgraph.models import WorkflowPlan, WorkflowStep
from agents.langgraph.router import RequestRouter


def test_evaluate_multi_case_incomplete_without_two_jobs():
    request = (
        "Run two OpenLB FHIT cases Case A: MRT Case B: BGK then plot_spectrum "
        "and report both job_ids"
    )
    result = evaluate_completion(
        user_request=request,
        task_results=[{"text": "Loaded manifest for job_aaaaaaaaaaaa", "tool_outputs": []}],
        session_context={"comparison_job_ids": ["job_aaaaaaaaaaaa"]},
        artifacts=[],
    )
    assert result["complete"] is False
    assert "multiple_cases" in result["missing"]


def test_evaluate_multi_case_complete_with_two_jobs_and_figure():
    request = (
        "Run two OpenLB FHIT cases Case A: MRT Case B: BGK then compute_spectra "
        "plot_spectrum and report both job_ids"
    )
    result = evaluate_completion(
        user_request=request,
        task_results=[
            {
                "text": (
                    "Completed job_aaaaaaaaaaaa and job_bbbbbbbbbbbb; "
                    "compute_spectra + plot_spectrum overlay done."
                ),
                "tool_outputs": ["plot_spectrum: ok"],
            }
        ],
        session_context={
            "comparison_job_ids": ["job_aaaaaaaaaaaa", "job_bbbbbbbbbbbb"],
            "last_figure": {"path": "tmp/fig.png"},
        },
        artifacts=[{"artifact_type": "figure"}],
    )
    assert result["complete"] is True
    assert result["missing"] == []


def test_evaluate_allows_skip_when_unsupported_explained():
    request = "Run two cases Case A: MRT Case B: Cumulative and plot spectra"
    result = evaluate_completion(
        user_request=request,
        task_results=[
            {
                "text": (
                    "Unsupported collision Cumulative; supported: MRT, BGK. "
                    "Ran job_aaaaaaaaaaaa with MRT; compute_spectra plot_spectrum done."
                ),
                "tool_outputs": [],
            }
        ],
        session_context={
            "comparison_job_ids": ["job_aaaaaaaaaaaa"],
            "last_figure": True,
        },
        artifacts=[{"artifact_type": "figure"}],
    )
    assert result["complete"] is True


def test_finish_work_plan_injects_simulation_step():
    evaluation = {
        "complete": False,
        "missing": ["multiple_cases", "figure"],
        "gaps": ["Need two jobs"],
        "job_ids": ["job_aaaaaaaaaaaa"],
    }
    plan = finish_work_plan(
        user_request="run two cases",
        evaluation=evaluation,
        attempt=0,
    )
    assert plan.steps[0].role == "simulation"
    assert "COMPLETION SELF-CHECK FAILED" in plan.steps[0].instruction
    assert plan.rationale.startswith("completion_check:finish")


def test_completion_check_node_injects_and_exhausts():
    router = RequestRouter(planner_agent=None)
    nodes = AppGraphNodes(router, project_root=".", session_context={})
    plan = WorkflowPlan(
        steps=[WorkflowStep(role="steward", instruction="done", tool=None)],
        rationale="test",
    )
    state = {
        "user_request": (
            "Compile and run two OpenLB FHIT cases Case A: MRT Case B: BGK "
            "then plot_spectrum and report both job_ids"
        ),
        "plan": plan.model_dump(mode="json"),
        "task_index": 1,
        "task_results": [{"text": "Loaded KI-TURB manifest", "tool_outputs": []}],
        "artifacts": [],
        "final_text": "Loaded KI-TURB manifest",
        "metadata": {},
        "status": "running",
        "errors": [],
    }
    out = nodes.completion_check(state)
    assert out["status"] == "running"
    assert int(out["metadata"]["completion_attempts"]) == 1
    new_plan = WorkflowPlan.model_validate(out["plan"])
    assert len(new_plan.steps) == 2
    assert "COMPLETION SELF-CHECK" in new_plan.steps[1].instruction

    # Exhaust budget
    state2 = {
        **state,
        "plan": out["plan"],
        "task_index": len(new_plan.steps),
        "metadata": out["metadata"],
        "task_results": state["task_results"]
        + [{"text": "still only one job", "tool_outputs": []}],
    }
    # Force attempts to max-1 then one more to exhaust after inject... 
    # After first inject attempts=1. Run check again with still incomplete → attempts=2.
    out2 = nodes.completion_check(state2)
    assert int(out2["metadata"]["completion_attempts"]) == 2
    assert out2["status"] == "running"

    state3 = {
        **state2,
        "plan": out2["plan"],
        "task_index": len(WorkflowPlan.model_validate(out2["plan"]).steps),
        "metadata": out2["metadata"],
    }
    out3 = nodes.completion_check(state3)
    assert out3["status"] == "insufficient_data"
    assert "completion self-check budget" in out3["final_text"].lower()
    assert MAX_COMPLETION_ATTEMPTS == 2
    assert "Still missing" in exhaustion_message(state["user_request"], evaluate_completion(
        user_request=state["user_request"],
        task_results=state["task_results"],
        session_context={},
    ))
