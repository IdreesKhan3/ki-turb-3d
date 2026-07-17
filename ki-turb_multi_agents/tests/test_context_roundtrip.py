from agents.langgraph.engine import KITurbGraphEngine


def test_sync_context_out_merges_engine_writes_back_to_caller():
    engine = KITurbGraphEngine.__new__(KITurbGraphEngine)
    engine.session_context = {
        "data_directory": "/tmp/examples/DNS/512",
        "spectra_start_idx": 10,
        "spectra_end_idx": 200,
    }
    caller = {"data_directory": "/old", "plot_styles": {"Raw Energy Spectrum": {"line_width": 2}}}
    engine._sync_context_out(caller, "thread-1", {"status": "completed"})
    assert caller["data_directory"] == "/tmp/examples/DNS/512"
    assert caller["spectra_start_idx"] == 10
    assert caller["spectra_end_idx"] == 200
    assert caller["langgraph_thread_id"] == "thread-1"
    assert caller["kiturb_workflow_state"]["status"] == "completed"
