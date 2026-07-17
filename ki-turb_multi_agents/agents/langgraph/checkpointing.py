"""Checkpointer factory with a durable SQLite default and memory fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .compat import require_lang_dependencies


class CheckpointerHandle:
    def __init__(self, saver: Any, manager: Any = None):
        self.saver = saver
        self._manager = manager

    def close(self) -> None:
        if self._manager is not None:
            self._manager.__exit__(None, None, None)
            self._manager = None


def create_checkpointer(path: str | Path | None = None, *, memory: bool = False) -> CheckpointerHandle:
    require_lang_dependencies(sqlite=not memory)
    if memory:
        from langgraph.checkpoint.memory import InMemorySaver
        return CheckpointerHandle(InMemorySaver())
    from langgraph.checkpoint.sqlite import SqliteSaver
    destination = Path(path or ".kiturb/langgraph/checkpoints.sqlite").expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manager = SqliteSaver.from_conn_string(str(destination))
    saver = manager.__enter__()
    saver.setup()
    return CheckpointerHandle(saver, manager)


__all__ = ["CheckpointerHandle", "create_checkpointer"]
