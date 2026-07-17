"""Stable UI facade backed exclusively by LangChain and LangGraph."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..langgraph import KITurbGraphEngine


class UnifiedTeam:
    """Compatibility facade; there is only one agent engine: LangGraph."""

    def __init__(
        self,
        log_callback=None,
        stream_callback=None,
        activity_render_callback=None,
        project_root: Optional[Path] = None,
        provider_name: Optional[str] = None,
        **_: Any,
    ):
        self.project_root = (project_root or Path.cwd()).resolve()
        self.log_callback = log_callback or (lambda message: None)
        self.stream_callback = stream_callback
        self.activity_render_callback = activity_render_callback
        self.provider_name = provider_name or "ollama"
        self.engine = KITurbGraphEngine(
            self.project_root,
            provider_name=self.provider_name,
            log_callback=self.log_callback,
            stream_callback=self.stream_callback,
            activity_render_callback=self.activity_render_callback,
        )

    def run_chat_loop(self, user_message: str, chat_history: Optional[List[Dict[str, Any]]] = None, session_context: Optional[Dict[str, Any]] = None, resume_state=None, **kwargs):
        return self.engine.run_chat_loop(
            user_message,
            chat_history=chat_history,
            session_context=session_context,
            resume_state=resume_state,
            **kwargs,
        )

    def close(self) -> None:
        self.engine.close()


__all__ = ["UnifiedTeam"]
