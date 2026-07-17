"""Structured LLM output helpers with provider-specific fallbacks."""
from __future__ import annotations

from typing import Any, Type, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def provider_name(model_name: str) -> str:
    if ":" in model_name:
        return model_name.split(":", 1)[0].lower()
    return (model_name or "").lower()


def supports_agent_response_format(model_name: str) -> bool:
    """DeepSeek rejects LangChain json_schema response_format on the chat API."""
    return provider_name(model_name) not in {"deepseek"}


def invoke_structured(
    model,
    model_name: str,
    schema: Type[SchemaT],
    system_prompt: str,
    user_content: str,
    *,
    agent_name: str,
) -> SchemaT:
    if supports_agent_response_format(model_name):
        from langchain.agents import create_agent

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=system_prompt,
            response_format=schema,
            name=agent_name,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
        return schema.model_validate(result["structured_response"])

    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}\n\nReturn JSON only.\n{format_instructions}"),
            ("human", "{input}"),
        ]
    ).partial(system_prompt=system_prompt, format_instructions=parser.get_format_instructions())
    return (prompt | model | parser).invoke({"input": user_content})


class StructuredAgentRunner:
    """Agent-like invoke() wrapper for planner-style structured calls."""

    def __init__(self, model, model_name: str, schema: Type[SchemaT], system_prompt: str, agent_name: str):
        self.model = model
        self.model_name = model_name
        self.schema = schema
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self._agent = None
        if supports_agent_response_format(model_name):
            from langchain.agents import create_agent

            self._agent = create_agent(
                model=model,
                tools=[],
                system_prompt=system_prompt,
                response_format=schema,
                name=agent_name,
            )

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        messages = inputs.get("messages") or []
        user_content = messages[-1]["content"] if messages else ""
        if self._agent is not None:
            return self._agent.invoke(inputs)
        parsed = invoke_structured(
            self.model,
            self.model_name,
            self.schema,
            self.system_prompt,
            user_content,
            agent_name=self.agent_name,
        )
        return {"structured_response": parsed.model_dump()}


__all__ = [
    "StructuredAgentRunner",
    "invoke_structured",
    "provider_name",
    "supports_agent_response_format",
]
