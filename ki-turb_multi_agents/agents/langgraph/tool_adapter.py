"""Convert KI-TURB's existing role-scoped tools into LangChain tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel, Field, create_model

from agents.runtime import tool_registry
from agents.tools import execute_tool, get_tools_for_agent
from .compat import require_lang_dependencies

_JSON_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool, "object": dict, "array": list}

# KI_TURB_BINARY_ARTIFACT_TRANSPORT_V1
_ARTIFACT_TOOL_NAMES = frozenset({"read_document"})


def _artifact_summary(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(
            raw.get("message")
            or raw.get("error")
            or raw.get("artifact_title")
            or "Document artifact created."
        )
    return str(raw)


def _python_type(schema: Dict[str, Any]):
    kind = schema.get("type")
    if isinstance(kind, list):
        non_null = [item for item in kind if item != "null"]
        base = _JSON_TYPES.get(non_null[0], Any) if non_null else Any
        return Optional[base]
    if kind == "array":
        return List[_python_type(schema.get("items") or {})]
    return _JSON_TYPES.get(kind, Any)


def _args_model(definition: Dict[str, Any]) -> Type[BaseModel]:
    parameters = definition.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required: Set[str] = set(parameters.get("required") or [])
    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, schema in properties.items():
        annotation = _python_type(schema)
        default = ... if name in required else schema.get("default", None)
        if name not in required:
            annotation = Optional[annotation]
        fields[name] = (annotation, Field(default=default, description=schema.get("description") or ""))
    return create_model("".join(part.title() for part in definition["name"].split("_")) + "Args", **fields)


def build_langchain_tools(role: str, project_root: str | Path, session_context: Dict[str, Any]):
    require_lang_dependencies(sqlite=False)
    from langchain_core.tools import StructuredTool

    project = Path(project_root).resolve()
    definitions = get_tools_for_agent(role)
    allowed = frozenset(item["name"] for item in definitions)
    result = []
    for definition in definitions:
        name = definition["name"]
        args_schema = _args_model(definition)

        returns_artifact = name in _ARTIFACT_TOOL_NAMES

        def invoke(_tool_name=name, _returns_artifact=returns_artifact, **kwargs):
            cleaned = {key: value for key, value in kwargs.items() if value is not None}
            # LangChain's HumanInTheLoopMiddleware has already approved confirmable tools.
            old = session_context.get("tool_confirmation_approved")
            if tool_registry.requires_confirmation(_tool_name):
                session_context["tool_confirmation_approved"] = True
            try:
                raw = execute_tool(
                    _tool_name,
                    cleaned,
                    project,
                    session_context=session_context,
                    allowed_tool_names=allowed,
                )
            finally:
                if old is None:
                    session_context.pop("tool_confirmation_approved", None)
                else:
                    session_context["tool_confirmation_approved"] = old

            if _returns_artifact:
                artifact = raw if isinstance(raw, dict) and raw.get("artifact_type") else None
                return _artifact_summary(raw), artifact
            if isinstance(raw, (dict, list)):
                return json.dumps(raw, default=str)
            return str(raw)

        tool_kwargs = {
            "func": invoke,
            "name": name,
            "description": definition.get("description") or name,
            "args_schema": args_schema,
        }
        if returns_artifact:
            tool_kwargs["response_format"] = "content_and_artifact"
        result.append(StructuredTool.from_function(**tool_kwargs))
    return result


def confirmation_middleware(role: str):
    require_lang_dependencies(sqlite=False)
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    role_tools = tool_registry.tools_for_agent(role)
    interrupt_on = {
        name: {"allowed_decisions": ["approve", "reject"]}
        for name in role_tools
        if tool_registry.requires_confirmation(name)
    }
    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on) if interrupt_on else None


__all__ = ["build_langchain_tools", "confirmation_middleware"]
