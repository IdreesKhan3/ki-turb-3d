"""Required dependency boundary for the single LangChain/LangGraph runtime."""
from __future__ import annotations

from importlib.util import find_spec


def dependency_status() -> dict[str, bool]:
    return {
        "langchain": find_spec("langchain") is not None and find_spec("langchain_core") is not None,
        "langgraph": find_spec("langgraph") is not None,
        "sqlite": find_spec("langgraph.checkpoint.sqlite") is not None if find_spec("langgraph") else False,
    }


def require_lang_dependencies(*, sqlite: bool = True) -> None:
    status = dependency_status()
    missing = [name for name, available in status.items() if not available and (name != "sqlite" or sqlite)]
    if missing:
        raise RuntimeError(
            "KI-TURB now uses LangChain/LangGraph as its only agent engine. Missing: "
            + ", ".join(missing)
            + ". Install with: python -m pip install -r requirements-langgraph.txt"
        )


__all__ = ["dependency_status", "require_lang_dependencies"]
