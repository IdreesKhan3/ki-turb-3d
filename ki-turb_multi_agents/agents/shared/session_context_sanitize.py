"""Strip non-persistent objects from agent session_context before Streamlit roundtrip."""
from __future__ import annotations

from typing import Any, Dict


def sanitize_session_context_for_persistence(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy safe for Streamlit session_state (msgpack-serializable)."""
    if not ctx:
        return {}

    try:
        import plotly.graph_objects as go
    except Exception:  # pragma: no cover
        go = None  # type: ignore

    out = dict(ctx)

    try:
        from agents.tools.simulation._activity import ACTIVITY_CALLBACK_KEY

        out.pop(ACTIVITY_CALLBACK_KEY, None)
    except Exception:
        pass

    fig = out.pop("last_figure", None)
    if fig is not None and go is not None and isinstance(fig, go.Figure):
        out.setdefault("last_figure_json", fig.to_json())

    queue = out.pop("figure_queue", None)
    if queue and go is not None:
        json_queue = [item.to_json() for item in queue if isinstance(item, go.Figure)]
        if json_queue:
            out["figure_queue_json"] = json_queue

    for key in list(out.keys()):
        if callable(out[key]):
            out.pop(key, None)

    return out
