"""Document/source file edits must not be rewritten into analysis pipelines."""
from __future__ import annotations

from pathlib import Path

from agents.langgraph.engineering_intent import (
    is_document_or_latex_edit_request,
    is_simple_file_edit_request,
)
from agents.langgraph.models import WorkflowPlan, WorkflowStep
from agents.langgraph.router import RequestRouter, _sanitize_plan


ROOT = Path(__file__).resolve().parents[1]


def test_tex_figure_phrasing_is_document_edit():
    text = (
        "change the figure in the .tex paper file to higher quality; "
        "keep modifying the current one, do not create a new .tex"
    )
    assert is_document_or_latex_edit_request(text)
    assert is_simple_file_edit_request(text)


def test_deterministic_plan_routes_document_edit_to_steward():
    text = "edit the figure in my latex manuscript; modify the existing file only"
    plan = RequestRouter(planner_agent=None, project_root=ROOT).deterministic_plan(text)
    assert plan is not None
    assert plan.steps[0].role == "steward"


def test_sanitize_demotes_manifest_plan_for_document_edit():
    text = "update figures inside the current latex paper"
    bad = WorkflowPlan(
        kind="agent_workflow",
        steps=[
            WorkflowStep(
                role="analyst",
                instruction="Load job manifest",
                tool="load_dataset_manifest",
                tool_args={},
            )
        ],
        rationale="misrouted",
    )
    fixed = _sanitize_plan(bad, text)
    assert fixed.steps[0].role == "steward"
    assert fixed.steps[0].tool in (None, "")


def test_named_tex_path_is_file_edit():
    text = "raise figure DPI in exports/my_manuscript.tex"
    assert is_document_or_latex_edit_request(text)
    plan = RequestRouter(planner_agent=None, project_root=ROOT).deterministic_plan(text)
    assert plan is not None
    assert "exports/my_manuscript.tex" in plan.steps[0].instruction


def test_plain_analysis_request_still_not_forced_to_file_edit():
    text = "compute spectra for the latest HIT job and plot the energy spectrum"
    assert not is_document_or_latex_edit_request(text)
    assert not is_simple_file_edit_request(text)
