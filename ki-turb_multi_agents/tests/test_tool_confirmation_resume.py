from pages.AutonomousLab.confirmation import (
    _action_requests_from_pending,
    _build_langgraph_resume_value,
    auto_choice_for_duplicate_pending,
    pending_fingerprint,
)


def test_build_langgraph_resume_value_matches_all_action_requests():
    pending = {
        "kind": "langgraph_interrupt",
        "langgraph_interrupt_type": "tool_review",
        "langgraph_interrupt": {
            "action_requests": [
                {"name": "build_simulation_case", "args": {"case": "a"}},
                {"name": "compile_simulation", "args": {"case": "a"}},
                {"name": "start_simulation", "args": {"case": "a"}},
            ]
        },
    }

    approved = _build_langgraph_resume_value(pending, True)
    rejected = _build_langgraph_resume_value(pending, False)

    assert len(approved["decisions"]) == 3
    assert all(item == {"type": "approve"} for item in approved["decisions"])
    assert len(rejected["decisions"]) == 3
    assert all(item["type"] == "reject" for item in rejected["decisions"])
    assert "Do not retry" in rejected["decisions"][0]["message"]


def test_action_requests_from_pending_prefers_top_level_list():
    pending = {
        "action_requests": [{"name": "write_file", "args": {"filepath": "a.py"}}],
        "langgraph_interrupt": {"action_requests": [{"name": "ignored"}]},
    }
    assert _action_requests_from_pending(pending)[0]["name"] == "write_file"


def test_pending_fingerprint_stable_for_same_payload():
    pending = {
        "kind": "langgraph_interrupt",
        "langgraph_interrupt_type": "tool_review",
        "langgraph_thread_id": "thread-1",
        "tool": "write_file",
        "args": {"filepath": "paper.tex", "content": "hello"},
        "action_requests": [
            {"name": "write_file", "args": {"filepath": "paper.tex", "content": "hello"}},
        ],
    }
    a = pending_fingerprint(pending)
    b = pending_fingerprint(dict(pending))
    assert a == b
    other = dict(pending)
    other["args"] = {"filepath": "paper.tex", "content": "hello world"}
    other["action_requests"] = [
        {"name": "write_file", "args": {"filepath": "paper.tex", "content": "hello world"}},
    ]
    assert pending_fingerprint(other) != a


def test_auto_choice_for_duplicate_pending(monkeypatch):
    class _State(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    state = _State(lab_confirm_decisions={})
    monkeypatch.setattr(
        "pages.AutonomousLab.confirmation.st",
        type("S", (), {"session_state": state})(),
    )

    pending = {
        "kind": "langgraph_interrupt",
        "langgraph_interrupt_type": "tool_review",
        "langgraph_thread_id": "t1",
        "tool": "write_file",
        "action_requests": [
            {"name": "write_file", "args": {"filepath": "qlbm.tex", "content": "x"}},
        ],
    }
    assert auto_choice_for_duplicate_pending(pending) is None
    fp = pending_fingerprint(pending)
    state.lab_confirm_decisions[fp] = {"decision": "approved", "count": 1}
    assert auto_choice_for_duplicate_pending(pending) == "approved"
    state.lab_confirm_decisions[fp] = {"decision": "rejected", "count": 1}
    assert auto_choice_for_duplicate_pending(pending) == "rejected"
