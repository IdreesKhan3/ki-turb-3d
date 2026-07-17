from agents.langgraph.app_graph import _extract_job_id, _remember_simulation_job


def test_extract_job_id_from_tool_output():
    text = "Prepared case.\njob_id: sim-abc123\nstatus: created"
    assert _extract_job_id(text) == "sim-abc123"


def test_remember_simulation_job_updates_sidebar_keys():
    ctx: dict = {}
    _remember_simulation_job(ctx, "sim-xyz")
    assert ctx["simulation_job_id"] == "sim-xyz"
    assert ctx["sim_workflow_job"] == "sim-xyz"
