"""
Base LLM Agent — LLM + System Prompt + Tools.
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .shared.llm_provider import LLMProvider, get_llm_provider
from .tools import get_tools_definition, execute_tool
from .tools.status_messages import get_tool_status_before, get_tool_status_after


class LLMAgent:
    """LLM agent with system prompt and tools. Tools can be restricted per-agent to prevent scope creep."""

    def __init__(
        self,
        name: str,
        role_prompt: str,
        llm_provider: Optional[LLMProvider] = None,
        project_root: Optional[Path] = None,
        log_func=None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.system_prompt = role_prompt
        self.llm = llm_provider or get_llm_provider()
        self.project_root = project_root or Path.cwd()
        self.log = log_func or (lambda agent, msg: None)
        self._tools = tools if tools is not None else get_tools_definition()
        self.allowed_tool_names: Optional[set] = (
            frozenset(t.get("name") for t in self._tools if t.get("name")) if tools is not None else None
        )
        self.tools_text = self._format_tools_for_prompt()

    def _format_tools_for_prompt(self) -> str:
        """Format tool definitions for the LLM prompt."""
        tools = self._tools
        lines = ["Available tools (respond with JSON to call):"]
        for t in tools:
            lines.append(f"- {t['name']}: {t['description']}")
            if "parameters" in t and "properties" in t["parameters"]:
                for p, v in t["parameters"]["properties"].items():
                    lines.append(f"  - {p}: {v.get('description', '')}")
        lines.append('\nTo call a tool, respond with ONLY valid JSON: {"tool": "tool_name", "args": {...}}')
        lines.append("To reply to the user, respond with plain text (no JSON).")
        return "\n".join(lines)

    def think_and_act(
        self,
        user_input: str,
        context: str = "",
        max_tool_rounds: int = 15,
        session_context: Optional[Dict[str, Any]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        stream_reset_callback: Optional[Callable[[], None]] = None,
        tool_result_callback: Optional[Callable[[str], None]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        LLM thinks, optionally calls a tool, returns result.
        Returns dict with "text" and "artifact" when a tool produces a figure.
        When execute_tool returns pending_confirmation, returns that dict for UI to show Accept/Reject.

        1. Sends prompt + tools to LLM.
        2. LLM responds with text or JSON tool call.
        3. If tool call: execute, feed result back, repeat.
        4. Return final response (and artifact if any).
        """
        if resume_state:
            messages = resume_state.get("messages", [])
            last_assistant_response = resume_state.get("last_assistant_response", "")
            tool_result = resume_state.get("tool_result", "")
        else:
            messages = [
                {"role": "system", "content": f"{self.system_prompt}\n\n{self.tools_text}"},
                {"role": "user", "content": f"{context}\n\nUser: {user_input}" if context else f"User: {user_input}"},
            ]
            last_assistant_response = ""
            tool_result = ""

        if resume_state:
            messages.append({"role": "assistant", "content": last_assistant_response})
            messages.append({"role": "user", "content": f"[Tool result]\n{tool_result}\n\nContinue or provide final answer to user."})

        last_artifact: Optional[Dict[str, Any]] = None

        llm_kwargs = {"temperature": 0.2}
        if images:
            llm_kwargs["images"] = images

        for _ in range(max_tool_rounds):
            response = ""
            if stream_callback and hasattr(self.llm, "generate_stream_with_messages"):
                try:
                    collected = []
                    if stream_reset_callback:
                        stream_reset_callback()
                    for chunk in self.llm.generate_stream_with_messages(messages, **llm_kwargs):
                        collected.append(chunk)
                        stream_callback(chunk)
                    response = "".join(collected)
                except Exception:
                    response = self.llm.generate_with_messages(messages, **llm_kwargs)
            else:
                response = self.llm.generate_with_messages(messages, **llm_kwargs)
            if not response or not response.strip():
                return self._wrap_response("No response.", last_artifact)

            # Parse tool call from response
            tool_call = self._parse_tool_call(response)
            if tool_call:
                tool_name = tool_call.get("tool")
                args = tool_call.get("args", {})
                self.log(self.name, get_tool_status_before(tool_name, args))
                result = execute_tool(
                    tool_name, args, self.project_root,
                    session_context=session_context or {},
                    allowed_tool_names=self.allowed_tool_names,
                )

                # Pending confirmation: return for UI to show Accept/Reject
                if isinstance(result, dict) and result.get("status") == "pending_confirmation":
                    result["messages"] = messages
                    result["last_assistant_response"] = response
                    return result

                self.log(self.name, get_tool_status_after(tool_name, args, result))
                if tool_result_callback:
                    display = self._format_tool_result_for_display(tool_name, args, result)
                    if display:
                        tool_result_callback(display)

                # Check for artifact (e.g. plotly figure, markdown table, markdown content, downloadable file)
                if isinstance(result, dict) and result.get("artifact_type") in ("plotly_figure", "markdown_table", "markdown", "downloadable_file", "report_html"):
                    last_artifact = result
                    return self._wrap_response(result.get("message", "Done."), last_artifact)
                tool_output_for_llm = str(result)

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"[Tool result]\n{tool_output_for_llm}\n\nContinue or provide final answer to user."})
            else:
                return self._wrap_response(response.strip(), last_artifact)

        return self._wrap_response("Max tool rounds reached.", last_artifact)

    def _format_tool_result_for_display(self, tool_name: str, args: Dict[str, Any], result: Any) -> str:
        """Format tool result for chat display (brief, human-readable)."""
        if isinstance(result, dict):
            if result.get("status") == "pending_confirmation":
                return ""
            msg = result.get("message", "")
            if msg and len(msg) < 500:
                return msg
            if result.get("success") or result.get("ok"):
                path = args.get("filepath") or args.get("path", "")
                if tool_name == "read_file" and path:
                    return f"✓ Read `{path}`"
                if tool_name == "find_file":
                    return "✓ Found matching files."
                if tool_name == "list_directory":
                    return "✓ Listed directory."
                if tool_name in ("write_file", "modify_file") and path:
                    return f"✓ Updated `{path}`"
                if tool_name == "search_research_papers":
                    return "✓ Papers found."
                if tool_name == "web_search":
                    return "✓ Search complete."
                return "✓ Done."
        res_str = str(result)
        if "Error" in res_str:
            return res_str[:200]
        return res_str[:150] if len(res_str) > 150 else res_str

    def _wrap_response(self, text: str, artifact: Optional[Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """Return text only, or dict with text and artifact."""
        if artifact is not None:
            return {"text": text, "artifact": artifact}
        return text

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from LLM response. Handles JSON in various formats."""
        text = response.strip()
        if not text:
            return None

        # 1. Strip markdown code blocks (```json ... ``` or ``` ... ```)
        for pattern in [
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]:
            for m in re.finditer(pattern, text):
                parsed = self._extract_tool_from_json(m.group(1).strip())
                if parsed:
                    return parsed

        # 2. Try parsing the whole response as JSON
        parsed = self._extract_tool_from_json(text)
        if parsed:
            return parsed

        # 3. Find first complete JSON object (handles nested braces)
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start : i + 1])
                            parsed = self._normalize_tool_call(obj)
                            if parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

        return None

    def _extract_tool_from_json(self, s: str) -> Optional[Dict[str, Any]]:
        """Parse JSON string and extract tool call if present."""
        # Fix common LLM JSON issues: trailing commas, single quotes
        s = s.strip()
        if not s or not s.startswith("{"):
            return None
        # Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            obj = json.loads(s)
            return self._normalize_tool_call(obj)
        except json.JSONDecodeError:
            return None

    def _normalize_tool_call(self, obj: dict) -> Optional[Dict[str, Any]]:
        """Extract tool name and args from parsed object. Handles multiple key names."""
        if not isinstance(obj, dict):
            return None
        # Tool name: tool, tool_name, name, function
        tool = (
            obj.get("tool")
            or obj.get("tool_name")
            or (obj.get("name") if "args" in obj or "arguments" in obj else None)
            or (obj.get("function", {}).get("name") if isinstance(obj.get("function"), dict) else None)
        )
        # Args: args, arguments, parameters
        args = (
            obj.get("args")
            or obj.get("arguments")
            or obj.get("parameters")
            or {}
        )
        if isinstance(obj.get("function"), dict) and "arguments" in obj["function"]:
            try:
                args = json.loads(obj["function"]["arguments"]) if isinstance(obj["function"]["arguments"], str) else obj["function"]["arguments"]
            except (json.JSONDecodeError, TypeError):
                pass
        if tool and isinstance(args, dict):
            return {"tool": str(tool).strip(), "args": args}
        return None
