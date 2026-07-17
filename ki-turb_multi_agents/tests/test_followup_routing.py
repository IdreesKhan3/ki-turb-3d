"""LLM-first router: hard gate for lifecycle only; everything else uses planner/free-form."""
from __future__ import annotations

from agents.langgraph.models import WorkflowPlan, WorkflowStep
from agents.langgraph.router import RequestRouter
from agents.langgraph.turn_memory import update_turn_memory


class _FixedPlanner:
    """Returns a fixed plan — proves the router uses the LLM path, not keyword scripts."""

    def __init__(self, plan: WorkflowPlan):
        self.plan = plan
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return {"structured_response": self.plan}


def test_non_lifecycle_requests_skip_hard_gate():
    """Domain nouns / questions must not be stolen by the OpenLB lifecycle gate."""
    router = RequestRouter(planner_agent=None)
    session = {"manifest_path": "simulations/job_x/manifest.json"}
    samples = [
        "check this reference code vs our spectra formulae for openlb:\n```python\nx=1\n```",
        "can the agents plot a lumley triangle from saved simulation data?",
        "please plot the lumley triangle from the openlb simulation data",
        "write a python script that prints hello",
    ]
    for q in samples:
        assert router.deterministic_plan(q, session) is None, q


def test_planner_path_used_when_present():
    fixed = WorkflowPlan(
        steps=[WorkflowStep(role="orchestrator", instruction="Answer from context.")],
        rationale="from-planner",
    )
    planner = _FixedPlanner(fixed)
    router = RequestRouter(planner_agent=planner)
    plan = router.plan(
        "any follow-up about openlb spectra or isotropy tools",
        {"manifest_path": "simulations/job_x/manifest.json"},
    )
    assert planner.calls == 1
    assert plan.rationale == "from-planner"
    assert plan.steps[0].role == "orchestrator"
    assert plan.steps[0].tool is None


def test_no_planner_fallback_is_freeform_not_keyword_pipeline():
    router = RequestRouter(planner_agent=None)
    plan = router.plan(
        "check this reference code vs our spectra formulae for openlb",
        {"manifest_path": "simulations/job_x/manifest.json"},
    )
    assert plan.rationale.startswith("Free-form steward")
    assert plan.steps[0].role == "steward"
    assert plan.steps[0].tool is None


def test_lifecycle_hard_gate_still_compiles():
    router = RequestRouter(planner_agent=None)
    plan = router.deterministic_plan("compile FHIT 32^3 MRT on openlb and stop")
    assert plan is not None
    assert [s.tool for s in plan.steps] == ["build_simulation_case", "compile_simulation"]


def test_lifecycle_hard_gate_compile_do_not_start():
    router = RequestRouter(planner_agent=None)
    plan = router.deterministic_plan(
        "compile FHIT 16^3 MRT on openlb and stop — do not start the solver"
    )
    assert plan is not None
    assert [s.tool for s in plan.steps] == ["build_simulation_case", "compile_simulation"]
    assert "start_simulation" not in [s.tool for s in plan.steps]


def test_turn_memory_records_last_paths():
    mem = update_turn_memory(
        None,
        user_request="write examples/smoke_hello.py",
        plan={"steps": [{"role": "steward", "tool": "write_file", "tool_args": {"filepath": "examples/smoke_hello.py"}}]},
        task_results=[{
            "role": "steward",
            "text": "File written: examples/smoke_hello.py",
            "tool_outputs": ["File written: examples/smoke_hello.py"],
        }],
        final_text="created examples/smoke_hello.py",
        status="completed",
    )
    assert "examples/smoke_hello.py" in mem["last_paths"]


def test_turn_memory_records_compile_tools():
    mem = update_turn_memory(
        None,
        user_request="compile and run openlb",
        plan={"steps": [{"role": "simulation", "tool": "compile_simulation"}]},
        task_results=[{"role": "simulation", "text": "compiled", "tool_outputs": []}],
        session_context={"simulation_job_id": "job_1"},
        final_text="compiled ok",
        status="completed",
    )
    assert mem["job_id"] == "job_1"
    assert mem["compile_mentioned"] is True
    assert "compile_simulation" in mem["last_tools"]
