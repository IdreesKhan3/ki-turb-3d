import plotly.graph_objects as go

from agents.shared.session_context_sanitize import sanitize_session_context_for_persistence


def test_sanitize_session_context_strips_plotly_figure():
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 4])])
    ctx = {
        "data_directory": "/tmp/data",
        "last_figure": fig,
        "figure_queue": [fig],
    }
    safe = sanitize_session_context_for_persistence(ctx)
    assert "last_figure" not in safe
    assert "figure_queue" not in safe
    assert safe["data_directory"] == "/tmp/data"
    assert "last_figure_json" in safe
    assert "figure_queue_json" in safe
