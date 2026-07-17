"""Failed steps should hand off / reassess — not die immediately."""
from __future__ import annotations

from agents.langgraph.recovery import recovery_plan


def test_unauthorized_tool_hands_off_to_steward():
    plan = recovery_plan(
        user_request="read the README and summarize tools",
        failure="simulation.build_simulation_case: Error: unknown tool 'read_file' — not authorized",
    )
    assert plan.steps[0].role == "steward"
    assert "HANDOFF" in plan.steps[0].instruction
    assert "Recover: unauthorized" in plan.rationale


def test_physics_failure_explains_instead_of_blind_rebuild():
    plan = recovery_plan(
        user_request="compile FHIT 16^3 MRT and stop",
        failure="Error: physics validation failed. step0_divergence_budget exceeds limit",
    )
    assert plan.steps[0].role == "orchestrator"
    assert "Do NOT blindly rebuild" in plan.steps[0].instruction


def test_missing_csv_hands_off_to_steward_locate():
    plan = recovery_plan(
        user_request="please plot lumley triangle",
        failure="eps_real_validation*.csv or turbulence_validation*.csv not found. Use data_dir",
    )
    assert plan.steps[0].role == "steward"
    assert "list_simulation_jobs" in plan.steps[0].instruction
