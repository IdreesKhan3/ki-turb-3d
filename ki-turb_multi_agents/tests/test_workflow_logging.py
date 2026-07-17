from agents.langgraph.workflow_logging import format_workflow_update


def test_format_plan_update():
    lines = format_workflow_update(
        "plan",
        {
            "plan": {
                "kind": "agent_workflow",
                "rationale": "simple chat",
                "steps": [{"role": "orchestrator", "instruction": "Say hello"}],
            },
            "events": [{"stage": "plan", "status": "ok", "message": "simple chat"}],
        },
    )
    assert any("[PLAN]" in line for line in lines)
    assert any("orchestrator" in line.lower() for line in lines)


def test_format_agent_messages():
    from langchain_core.messages import AIMessage, ToolMessage

    lines = format_workflow_update(
        "orchestrator_agent",
        {
            "messages": [
                AIMessage(
                    content="Checking files.",
                    tool_calls=[{"id": "1", "name": "read_file", "args": {"path": "a.txt"}}],
                ),
                ToolMessage(content='{"ok": true}', name="read_file", tool_call_id="1"),
            ]
        },
    )
    assert any("orchestrator" in line.lower() for line in lines)
    assert any("read_file" in line for line in lines)
