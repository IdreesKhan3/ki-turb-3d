"""Simulation progress events for the Autonomous Lab activity UI."""

from __future__ import annotations

from types import SimpleNamespace

from agents.tools.simulation._activity import (
    emit_simulation_progress,
    progress_percent_from_job,
)
from pages.AutonomousLab.live_activity import LiveActivityRenderer


class _Placeholder:
    def __init__(self) -> None:
        self.html = ""

    def markdown(self, content: str, **_: object) -> None:
        self.html = content


def test_emit_simulation_progress_invokes_callback():
    events: list[dict] = []

    emit_simulation_progress(
        {"_activity_callback": events.append},
        phase="simulation",
        job_id="job_abc",
        progress_pct=42.5,
        message="OpenLB running",
        step=425,
        max_steps=1000,
    )

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "simulation_progress"
    assert event["progress"] == 42.5
    assert "425" in event["summary"]
    assert "1,000" in event["summary"]


class _NoSessionContext(Exception):
    pass


def test_emit_simulation_progress_survives_missing_streamlit_context():
    def _raise_no_context(_event: dict) -> None:
        raise _NoSessionContext()

    ctx: dict = {"_activity_callback": _raise_no_context}
    emit_simulation_progress(
        ctx,
        phase="compile",
        job_id="job_abc",
        progress_pct=5.0,
        message="Compiling",
    )
    assert ctx["_simulation_progress"]["progress"] == 5.0
    assert len(ctx["_simulation_progress_queue"]) == 1


def test_progress_percent_from_job_uses_diagnostics_fraction():
    job = SimpleNamespace(
        progress=0.12,
        measured={"step": 1200},
        requested_config={"runtime": {"max_steps": 10000}},
        resources={},
        job_id="job_x",
    )
    pct, step = progress_percent_from_job(job)
    assert pct == 12.0
    assert step == 1200


def test_live_activity_renders_progress_bar():
    placeholder = _Placeholder()
    renderer = LiveActivityRenderer(placeholder)
    renderer.log(
        {
            "type": "simulation_progress",
            "agent": "simulation",
            "status": "running",
            "title": "Simulation",
            "summary": "step 500 / 1,000 (50.0%)",
            "progress": 50.0,
            "job_id": "job_demo",
        }
    )
    assert "ki-pro-progress" in placeholder.html
    assert "50.0%" in placeholder.html
    assert "step 500 / 1,000" in placeholder.html


def test_live_activity_panel_is_height_capped():
    placeholder = _Placeholder()
    renderer = LiveActivityRenderer(placeholder, max_visible_events=3)
    for i in range(8):
        renderer.log(
            {
                "type": "activity",
                "agent": "simulation",
                "status": "success",
                "title": f"Step {i}",
                "summary": f"update {i}",
            }
        )
    assert "max-height: min(38vh, 22rem)" in placeholder.html
    assert "overflow-y: auto" in placeholder.html
    assert "earlier update(s) not shown" in placeholder.html
    # Newest first inside the capped list
    assert placeholder.html.index("Step 7") < placeholder.html.index("Step 5")
