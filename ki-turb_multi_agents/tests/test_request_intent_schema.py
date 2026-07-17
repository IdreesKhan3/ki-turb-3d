"""Schema-first request classification and plan mapping."""

from __future__ import annotations

from pathlib import Path

from agents.langgraph.intent_plans import plan_from_intent
from agents.langgraph.request_intent import classify_request
from agents.langgraph.router import RequestRouter


def test_classify_load_vs_run():
    load = classify_request("load already saved agents data from openlb simulations")
    assert load is not None
    assert load.action == "load"
    assert load.backend == "openlb"

    run = classify_request("run openlb FHIT 16^3 MRT for 1000 iterations")
    assert run is not None
    assert run.action == "run"
    assert run.case_params.get("resolution") == [16, 16, 16]


def test_classify_compile_only():
    intent = classify_request("compile FHIT 32^3 MRT on openlb and stop")
    assert intent is not None
    assert intent.action == "compile"
    plan = plan_from_intent(intent)
    assert [s.tool for s in plan.steps] == ["build_simulation_case", "compile_simulation"]


def test_compile_do_not_start_is_compile_only_not_run():
    intent = classify_request(
        "compile FHIT 16^3 MRT on openlb and stop — do not start the solver"
    )
    assert intent is not None
    assert intent.action == "compile"
    plan = plan_from_intent(intent)
    tools = [s.tool for s in plan.steps]
    assert tools == ["build_simulation_case", "compile_simulation"]
    assert "start_simulation" not in tools


def test_plan_from_intent_action_selects_graph():
    load = classify_request("load existing openlb results and plot energy spectra")
    plan = plan_from_intent(load)
    tools = [s.tool for s in plan.steps]
    assert tools[0] == "load_dataset_manifest"
    assert "start_simulation" not in tools
    assert "plot_spectrum" in tools


def test_keyword_spectra_openlb_is_not_auto_analyze():
    intent = classify_request(
        "does our spectra code use the same formulae as this reference for openlb data?",
        session_summary={"manifest_path": "simulations/job_x/manifest.json"},
    )
    assert intent is None


def test_router_hard_gates_load_lifecycle():
    root = Path(__file__).resolve().parents[1]
    plan = RequestRouter(planner_agent=None, project_root=root).deterministic_plan(
        "load already saved agents data from openlb simulations"
    )
    assert plan is not None
    assert plan.rationale.startswith("schema:load")
    assert plan.steps[0].tool == "load_dataset_manifest"


def test_classify_named_solver_backends():
    """Lifecycle hard gate is backend-agnostic; named solvers are not forced to openlb."""
    palabos = classify_request("compile a palabos case and stop")
    assert palabos is not None
    assert palabos.action == "compile"
    assert palabos.backend == "palabos"
    assert palabos.case_params.get("backend") == "palabos"

    ansys = classify_request("run ansys channel flow")
    assert ansys is not None
    assert ansys.action == "run"
    assert ansys.backend == "ansys"
