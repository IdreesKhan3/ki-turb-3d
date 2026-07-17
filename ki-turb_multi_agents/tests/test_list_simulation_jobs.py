"""list_simulation_jobs counts saved OpenLB jobs without loading a manifest."""
from __future__ import annotations

from pathlib import Path

from agents.tools.simulation.manifest import execute_tool


def test_list_simulation_jobs_counts_repo_jobs():
    root = Path(__file__).resolve().parents[1]
    text = execute_tool("list_simulation_jobs", {}, root)
    assert "Saved simulation jobs under simulations/:" in text
    # Repo currently has multiple job_* dirs
    assert "job_" in text


def test_overwrite_logging_does_not_crash():
    from langgraph.types import Overwrite
    from agents.langgraph.workflow_logging import format_workflow_events

    events = format_workflow_events(
        "recover_step",
        {"errors": Overwrite([]), "events": [{"stage": "recover", "status": "ok", "message": "handoff"}]},
    )
    assert isinstance(events, list)
