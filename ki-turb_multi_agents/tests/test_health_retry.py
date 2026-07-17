"""Tests for simulation health rejection → retune → retry."""
from __future__ import annotations

from pathlib import Path

from agents.langgraph.health_retry import (
    MAX_HEALTH_RETRIES,
    build_params_from_job,
    is_recoverable_health_rejection,
    lifecycle_retry_steps,
    retune_build_params,
    splice_health_retry_plan,
)
from agents.langgraph.models import WorkflowPlan, WorkflowStep


def test_detects_divergence_health_rejection():
    msg = (
        "Simulation status. job_id: job_e2698a2e9304 backend: openlb status: rejected "
        "progress: 1.0% step: 100 message: simulation health rejection: "
        "divergence RMS 0.147517 exceeds 0.08"
    )
    assert is_recoverable_health_rejection(msg) is True
    assert is_recoverable_health_rejection("status: completed") is False


def test_retune_cuts_unsafe_lattice_velocity():
    params = {
        "backend": "openlb",
        "flow": "hit",
        "name": "DHIT_64_validation",
        "hit_mode": "decaying",
        "char_velocity": 1.0,
        "target_urms": 1.0,
        "mach_number": 0.05,
        "reynolds_number": 1000.0,
        "scheme": "SmagorinskyBGK",
        "turbulence_regime": "les",
        "forcing_type": "none",
    }
    retuned = retune_build_params(
        params,
        "simulation health rejection: divergence RMS 0.147517 exceeds 0.08",
        attempt=0,
        measured={"mach_max": 0.219645, "divergence_rms": 0.147517},
    )
    assert retuned["char_velocity"] <= 0.1
    assert retuned["target_urms"] == retuned["char_velocity"]
    assert retuned["mach_number"] <= 0.05
    assert retuned["scheme"] in {"BGK", "MRT"}
    assert retuned["turbulence_regime"] == "dns"
    assert retuned["reynolds_number"] == 100.0
    assert "retry1" in retuned["name"]


def test_build_params_from_rejected_job():
    root = Path(__file__).resolve().parents[1]
    job_id = "job_e2698a2e9304"
    if not (root / "simulations" / job_id / "requested_case.json").is_file():
        return
    params = build_params_from_job(root, job_id)
    assert params["backend"] == "openlb"
    assert params["flow"] == "hit"
    assert float(params["char_velocity"]) == 1.0
    assert params["hit_mode"] == "decaying"


def test_splice_inserts_rebuild_compile_start_supervise():
    plan = WorkflowPlan(
        steps=[
            WorkflowStep(role="analyst", instruction="research"),
            WorkflowStep(role="simulation", instruction="build", tool="build_simulation_case"),
            WorkflowStep(role="simulation", instruction="compile", tool="compile_simulation"),
            WorkflowStep(role="simulation", instruction="start", tool="start_simulation"),
            WorkflowStep(role="simulation", instruction="supervise", tool="supervise_simulation"),
            WorkflowStep(role="simulation", instruction="fetch", tool="fetch_simulation_outputs"),
        ],
        rationale="schema:run",
    )
    retuned = {
        "backend": "openlb",
        "flow": "hit",
        "name": "DHIT_retry1",
        "char_velocity": 0.1,
        "target_urms": 0.1,
        "scheme": "BGK",
        "hit_mode": "decaying",
        "health_retry_attempt": 1,
    }
    new_plan, idx = splice_health_retry_plan(plan, supervise_index=4, build_args=retuned)
    assert idx == 4
    tools = [s.tool for s in new_plan.steps]
    assert tools[4:8] == [
        "build_simulation_case",
        "compile_simulation",
        "start_simulation",
        "supervise_simulation",
    ]
    assert tools[-1] == "fetch_simulation_outputs"
    assert new_plan.steps[4].tool_args["char_velocity"] == 0.1
    assert MAX_HEALTH_RETRIES >= 3
    assert len(lifecycle_retry_steps(retuned)) == 4
