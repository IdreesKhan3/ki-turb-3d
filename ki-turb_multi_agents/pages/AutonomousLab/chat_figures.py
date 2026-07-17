"""Serialize Plotly figures in chat history for Streamlit session_state."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import plotly.io as pio


def store_plotly_artifact_in_message(msg_entry: Dict[str, Any], artifact_content: Any) -> str:
    """Persist plot JSON on the chat message; return parsed JSON string."""
    content = artifact_content
    if isinstance(content, dict):
        content = json.dumps(content)
    content = str(content)
    figures_json = list(msg_entry.get("figures_json") or [])
    figures_json.append(content)
    msg_entry["figures_json"] = figures_json
    if len(figures_json) == 1:
        msg_entry["figure_json"] = content
    msg_entry.pop("figure", None)
    msg_entry.pop("figures", None)
    return content


def figures_from_message(msg: Dict[str, Any]) -> List[Any]:
    """Reconstruct Plotly figures from a chat history entry."""
    jsons: List[str] = []
    if msg.get("figures_json"):
        jsons.extend(msg["figures_json"])
    elif msg.get("figure_json"):
        jsons.append(msg["figure_json"])

    figures = []
    for content in jsons:
        try:
            if isinstance(content, dict):
                content = json.dumps(content)
            figures.append(pio.from_json(content))
        except Exception:
            continue

    if figures:
        return figures

    legacy = msg.get("figures") or ([msg["figure"]] if msg.get("figure") else [])
    return list(legacy)
