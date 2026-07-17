"""LangChain agents for KI-TURB's designed specialist roles."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .compat import require_lang_dependencies
from .models import WorkflowPlan, WorkflowSummary
from .prompts import PLANNER_PROMPT, ROLE_PROMPTS
from .structured_output import StructuredAgentRunner, invoke_structured
from .tool_adapter import build_langchain_tools, confirmation_middleware


class RoleAgentFactory:
    def __init__(self, model_name: str, project_root: str | Path, session_context: Dict[str, Any]):
        require_lang_dependencies(sqlite=False)
        from langchain.chat_models import init_chat_model

        self.model_name = model_name
        self.project_root = Path(project_root).resolve()
        self.session_context = session_context
        self.model = init_chat_model(model_name, temperature=0)

    def create_role_agent(self, role: str):
        from langchain.agents import create_agent
        tools = build_langchain_tools(role, self.project_root, self.session_context)
        middleware = confirmation_middleware(role)
        return create_agent(
            model=self.model,
            tools=tools,
            system_prompt=ROLE_PROMPTS[role],
            middleware=[middleware] if middleware else [],
            name=f"kiturb_{role}",
        )

    def create_planner(self):
        return StructuredAgentRunner(
            self.model,
            self.model_name,
            WorkflowPlan,
            PLANNER_PROMPT,
            "kiturb_planner",
        )

    def summarize(self, payload: str) -> WorkflowSummary:
        return invoke_structured(
            self.model,
            self.model_name,
            WorkflowSummary,
            (
                "Summarize the completed KI-TURB workflow from supplied evidence only. "
                "Do not call failures or insufficient data successful."
            ),
            payload,
            agent_name="kiturb_summarizer",
        )


__all__ = ["RoleAgentFactory"]
