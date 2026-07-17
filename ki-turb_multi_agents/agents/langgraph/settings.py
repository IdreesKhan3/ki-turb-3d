"""Configuration for KI-TURB's single LangGraph workflow engine."""
from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict


class LangGraphSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = ".kiturb/langgraph/checkpoints.sqlite"
    require_execution_approval: bool = True
    model: str = "ollama:qwen2.5-coder:32b"
    run_root: str = "tmp/langgraph_runs"
    max_poll_seconds: float = 5.0
    run_timeout_seconds: float | None = None
    use_llm_planner: bool = True
    use_llm_hit_parser: bool = True
    use_llm_summary: bool = True
    max_plan_steps: int = 8

    @classmethod
    def from_environment(cls, project_root: str | Path, provider_name: str = "ollama") -> "LangGraphSettings":
        root = Path(project_root).resolve()
        provider = (provider_name or "ollama").lower()
        if provider == "gemini":
            default_model = f"google_genai:{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}"
        elif provider == "deepseek":
            default_model = f"deepseek:{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-pro')}"
        else:
            default_model = f"ollama:{os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:32b')}"
        timeout = os.getenv("KITURB_LANGGRAPH_RUN_TIMEOUT")
        return cls(
            checkpoint_path=os.getenv("KITURB_LANGGRAPH_CHECKPOINT_DB", str(root / ".kiturb/langgraph/checkpoints.sqlite")),
            require_execution_approval=os.getenv("KITURB_REQUIRE_RUN_APPROVAL", "1").lower() not in {"0", "false", "no"},
            model=os.getenv("KITURB_LANGCHAIN_MODEL", default_model),
            run_root=os.getenv("KITURB_LANGGRAPH_RUN_ROOT", str(root / "tmp/langgraph_runs")),
            max_poll_seconds=float(os.getenv("KITURB_LANGGRAPH_POLL_SECONDS", "5")),
            run_timeout_seconds=float(timeout) if timeout else None,
            use_llm_planner=os.getenv("KITURB_LANGCHAIN_PLANNER", "1").lower() not in {"0", "false", "no"},
            use_llm_hit_parser=os.getenv("KITURB_LANGCHAIN_PARSE_REQUESTS", "1").lower() not in {"0", "false", "no"},
            use_llm_summary=os.getenv("KITURB_LANGCHAIN_SUMMARIZE", "1").lower() not in {"0", "false", "no"},
            max_plan_steps=int(os.getenv("KITURB_LANGGRAPH_MAX_PLAN_STEPS", "8")),
        )


__all__ = ["LangGraphSettings"]
