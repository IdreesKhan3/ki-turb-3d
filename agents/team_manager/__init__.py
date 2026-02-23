"""
Team Manager — 5 LLM-driven agents (Orchestrator, Steward, Analyst, Visualizer, Reviewer).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base_agent import LLMAgent
from ..orchestrator import OrchestratorAgent
from ..shared.llm_provider import get_llm_provider

from .prompts import (
    ORCHESTRATOR_PROMPT,
    STEWARD_PROMPT,
    ANALYST_PROMPT,
    VISUALIZER_PROMPT,
    REVIEWER_PROMPT,
)
from ..intent_detection import get_plot_routing
from agents.shared.image_processor import (
    convert_to_provider_format,
    plotly_figure_to_image_dict,
    extract_figure_data_for_agent,
    figure_fingerprint,
)
from ..tools import get_tools_for_agent
from ..tools.status_messages import friendly_delegation


def _parse_plan_steps(plan: str) -> List[str]:
    """Parse numbered plan into list of step descriptions. E.g. '1. Find files\\n2. Plot' -> ['Find files', 'Plot']."""
    if not plan or not plan.strip():
        return []
    steps = []
    for line in plan.strip().split("\n"):
        line = line.strip()
        # Match "1. Step text" or "1) Step text"
        m = re.match(r"^\d+[.)]\s*(.+)$", line)
        if m:
            steps.append(m.group(1).strip())
    return steps if steps else [plan.strip()]


def _count_add_report_steps_in_plan(plan_steps: List[str]) -> int:
    """Count plan steps that add to report (add_report_section). Includes plot, table, theory, text."""
    if not plan_steps:
        return 1
    count = 0
    for step in plan_steps:
        s = step.lower()
        # Explicit add_report_section or "add X to report" (figure, table, theory, text)
        if "add_report_section" in s or ("add" in s and "report" in s):
            count += 1
    return count if count > 0 else 1


def _count_add_report_successes(ctx_joined: str) -> int:
    """Count add_report_section successes in context."""
    return ctx_joined.count("[Step done: add_report_section succeeded]")


def _is_report_mode(plan_steps: List[str]) -> bool:
    """True when plan builds a report (add to report + preview). In report mode, figures/tables go only inside the report, not as standalone artifacts."""
    if not plan_steps:
        return False
    plan_joined = " ".join(s.lower() for s in plan_steps)
    has_add = "add" in plan_joined and "report" in plan_joined
    has_preview = "preview_report" in plan_joined or "preview report" in plan_joined or "show report" in plan_joined
    return has_add and has_preview


def _is_step_failure(result_text: str) -> bool:
    """Detect if an agent step failed (error message, no files found, etc.)."""
    if not result_text or not isinstance(result_text, str):
        return False
    text = result_text.strip().lower()
    failure_indicators = [
        "error:",
        "error ",
        "no files found",
        "no matching files",
        "not found",
        "could not",
        "failed",
        "failed to",
        "unable to",
        "does not exist",
        "tool error",
        "blocked dangerous",
        "timed out",
        "timeout",
    ]
    return any(ind in text for ind in failure_indicators)


class UnifiedTeam:
    """Five LLM-driven agents: each thinks and uses tools. Strong agentic design: Plan → Act → Reflect."""

    def __init__(
        self,
        log_callback=None,
        project_root: Optional[Path] = None,
        provider_name: Optional[str] = None,
    ):
        self.project_root = project_root or Path.cwd()
        self.log_callback = log_callback or (lambda x: None)

        def log(agent: str, msg: str):
            self.log_callback(f"**[{agent.upper()}]** {msg}")

        from ..shared.llm_provider import get_llm_provider
        self.provider_name = provider_name or "ollama"
        llm = get_llm_provider(self.provider_name)
        self.planner = OrchestratorAgent(log, llm_provider=llm)
        self.orchestrator = LLMAgent(
            "Orchestrator", ORCHESTRATOR_PROMPT, llm, self.project_root, log,
            tools=get_tools_for_agent("orchestrator"),
        )
        self.steward = LLMAgent(
            "Data Steward", STEWARD_PROMPT, llm, self.project_root, log,
            tools=get_tools_for_agent("steward"),
        )
        self.analyst = LLMAgent(
            "Analyst", ANALYST_PROMPT, llm, self.project_root, log,
            tools=get_tools_for_agent("analyst"),
        )
        self.visualizer = LLMAgent(
            "Visualizer", VISUALIZER_PROMPT, llm, self.project_root, log,
            tools=get_tools_for_agent("visualizer"),
        )
        self.reviewer = LLMAgent(
            "Reviewer", REVIEWER_PROMPT, llm, self.project_root, log,
            tools=get_tools_for_agent("reviewer"),
        )

    def _parse_delegation(self, response: str) -> Optional[Tuple[str, str]]:
        """Parse orchestrator response for delegation. Returns (agent_name, task) or None."""
        text = response.strip()
        if not text:
            return None
        for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
            text = m.group(1).strip()
            break
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict) and "delegate" in obj and "task" in obj:
                            agent = str(obj["delegate"]).strip().lower()
                            task = str(obj["task"]).strip()
                            if agent in ("steward", "analyst", "visualizer", "reviewer"):
                                return (agent, task)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break
        return None

    def _get_agent(self, name: str):
        """Return agent by name."""
        return {
            "steward": self.steward,
            "analyst": self.analyst,
            "visualizer": self.visualizer,
            "reviewer": self.reviewer,
        }.get(name)

    def run_chat_loop(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        session_context: Optional[Dict[str, Any]] = None,
        max_delegate_rounds: int = 25,
        use_plan: bool = True,
        resume_state: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Any] = None,
        stream_reset_callback: Optional[Any] = None,
        tool_result_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Strong agentic loop: Plan → Act → Reflect.
        Orchestrator plans first, then delegates with step context. Reviewer validates artifacts.
        chat_history: list of {"role": "user"|"assistant", "content": str}.
        session_context: optional dict with data_directory, all_loaded_files (current loaded data).
        use_plan: If True, generate execution plan before delegation (strong agentic mode).
        """
        context_parts: List[str] = []
        last_artifact: Optional[Dict[str, Any]] = None
        collected_artifacts: List[Dict[str, Any]] = []  # Multi-task: accumulate all figures/tables in one response
        final_text: Optional[str] = None  # When last step is "explain", keep that text for the response
        max_artifacts_per_turn = 10  # Cap to avoid runaway
        plan = ""
        plan_steps: List[str] = []
        current_step_index = 0
        rejection_count = 0  # Prevent infinite loop when Reviewer keeps rejecting
        last_rejection_reason = ""  # Reflection: inject into next orchestrator turn so it changes course

        def _format_history(history: List[Dict[str, Any]]) -> str:
            lines = []
            for m in history:
                role = "User" if m.get("role") == "user" else "Assistant"
                content = m.get("content", "")
                if content:
                    lines.append(f"{role}: {content}")
            return "\n".join(lines[-20:]) if lines else ""

        def _format_artifact_history(artifact_history: List[Dict[str, Any]]) -> str:
            """Format artifact history so agents can refer to any previous figure/table/image."""
            if not artifact_history:
                return ""
            lines = [
                "RECENT ARTIFACTS (most recent = 1; user can ask to explain any of these):",
                "When the user says 'explain the first figure', 'what about the table above', 'interpret that plot', use the artifact with that number.",
                "When the user asks to modify/add functionality to a figure, use source_file and tool_name to find the code.",
                "",
            ]
            for idx, art in enumerate(reversed(artifact_history), start=1):
                kind = art.get("type", "figure")
                caption = (art.get("caption") or "")[:120]
                if kind == "figure":
                    src = art.get("source_file")
                    tool = art.get("tool_name")
                    src_info = f" [source: {src}, tool: {tool}]" if (src or tool) else ""
                    lines.append(f"--- Artifact {idx} (figure){src_info}" + (f": {caption}" if caption else "") + " ---")
                    if art.get("figure_data"):
                        lines.append(art["figure_data"])
                elif kind == "table":
                    lines.append(f"--- Artifact {idx} (table)" + (f": {caption}" if caption else "") + " ---")
                    if art.get("table_md"):
                        lines.append(art["table_md"])
                elif kind == "image":
                    lines.append(f"--- Artifact {idx} (user-uploaded image)" + (f": {caption}" if caption else "") + " ---")
                lines.append("")
            return "\n".join(lines).strip()

        def _format_session(sess: Optional[Dict[str, Any]]) -> str:
            if not sess:
                return ""
            parts = []
            if sess.get("data_directory"):
                parts.append(
                    f"SESSION DATA PATH (use for ALL requests when user doesn't specify a different path): {sess['data_directory']}\n"
                    "When user says 'now plot X', 'also show Y', or switches topic without a path, use this path."
                )
            if sess.get("data_directories"):
                parts.append(f"Loaded directories: {', '.join(str(d) for d in sess['data_directories'][:5])}")
            if sess.get("all_loaded_files"):
                fl = sess["all_loaded_files"]
                summary = []
                for ft, lst in fl.items():
                    if lst and isinstance(lst, list):
                        n = len(lst)
                        summary.append(f"{ft}: {n} file(s)")
                if summary:
                    parts.append("Available file types: " + "; ".join(summary[:10]))
            if sess.get("style_config") or sess.get("spectra_style"):
                parts.append("User's plot style available for plot_spectrum—only pass when user explicitly requests custom style.")
            if sess.get("axis_labels_raw") or sess.get("axis_labels_norm"):
                parts.append("Axis labels available for plot_spectrum—only pass when user explicitly requests custom labels.")
            if sess.get("axis_labels_pdfs") or sess.get("legend_titles_pdfs"):
                parts.append("Legend & Axis Labels available for plot_pdf—only pass when user explicitly requests custom labels.")
            if sess.get("pdfs_plot_styles") or sess.get("pdfs_style_config"):
                parts.append("Plot style available for plot_pdf (fonts, grid, per-sim overrides)—only pass when user explicitly requests custom style.")
            artifact_hist = sess.get("artifact_history") or []
            if artifact_hist:
                parts.append(_format_artifact_history(artifact_hist))
            return "\n".join(parts) if parts else ""

        history_str = _format_history(chat_history or [])
        session_str = _format_session(session_context or {})

        # Build images for vision: all recent figures and user-uploaded images (so agents can explain any)
        images = []
        artifact_history = (session_context or {}).get("artifact_history") or []
        img_list = []
        for art in reversed(artifact_history):
            if art.get("figure_image") and art.get("figure_image").get("data"):
                img_list.append(art["figure_image"])
        # Limit to last 5 images to avoid token/API limits
        img_list = img_list[:5]
        if img_list:
            images = convert_to_provider_format(img_list, self.provider_name)

        # Intent override: use turbulence intent parser (like extra/chatbot_agent/intent_detection)
        routing = get_plot_routing(user_message)
        intent_override = routing.get("intent_override_text") or ""

        # Step 1: Planning phase (Strong Agentic — deliberate before acting)
        if use_plan:
            try:
                planning_context = {
                    "session_str": session_str,
                    "chat_history": chat_history or [],
                }
                plan = self.planner.plan(user_message, planning_context)
                plan_steps = _parse_plan_steps(plan)
                if plan:
                    self.log_callback("**[ORCHESTRATOR]** Ready to proceed.")
                    context_parts.append(f"[EXECUTION PLAN]\n{plan}")
            except Exception as e:
                self.log_callback(f"**[ORCHESTRATOR]** Skipping plan, proceeding directly: {e}")
                plan = "1. Process the user's request"
                plan_steps = [plan]

        # Resume from pending tool confirmation (user clicked Accept/Reject)
        if resume_state:
            response = self.orchestrator.think_and_act(
                "", session_context=session_context, resume_state=resume_state,
                stream_callback=stream_callback, stream_reset_callback=stream_reset_callback,
                tool_result_callback=tool_result_callback,
                images=images,
            )
            if isinstance(response, dict) and response.get("status") == "pending_confirmation":
                return response
            response_text = response.get("text", response) if isinstance(response, dict) else response
            if isinstance(response, dict) and response.get("artifact"):
                return response
            delegation = self._parse_delegation(response_text)
            if delegation is None:
                return {"text": response_text, "artifact": response.get("artifact") if isinstance(response, dict) else None}
            agent_name, task = delegation
            agent = self._get_agent(agent_name)
            if agent is None:
                return {"text": response_text, "artifact": None}
            self.log_callback(f"**[ORCHESTRATOR]** {friendly_delegation(agent_name, task)}")
            agent_context = []
            if intent_override:
                agent_context.append(intent_override.strip())
            if session_str:
                agent_context.append("Session (current loaded data):\n" + session_str)
            if history_str:
                agent_context.append("Previous conversation:\n" + history_str)
            if context_parts:
                agent_context.append("Context from previous steps:\n" + "\n".join(context_parts))
            result = agent.think_and_act(
                task,
                context="\n\n".join(agent_context) if agent_context else "",
                session_context=session_context,
                stream_callback=stream_callback, stream_reset_callback=stream_reset_callback,
                tool_result_callback=tool_result_callback,
                images=images,
            )
            if isinstance(result, dict) and result.get("status") == "pending_confirmation":
                return result
            result_text = result.get("text", result) if isinstance(result, dict) else result
            return {"text": result_text, "artifact": result.get("artifact") if isinstance(result, dict) else None}

        for _ in range(max_delegate_rounds):
            full_input = intent_override + user_message
            if last_rejection_reason:
                full_input += f"\n\n[WARNING] Previous attempt failed: {last_rejection_reason}\nYOU MUST CHANGE YOUR PLAN. Delegate to a different agent or fix the approach."
            if session_str:
                full_input = "Session (current loaded data):\n" + session_str + "\n\n" + full_input
            if history_str:
                full_input = "Previous conversation:\n" + history_str + "\n\nCurrent request: " + full_input
            if plan and plan_steps:
                step_hint = ""
                if current_step_index < len(plan_steps):
                    step_hint = f"\nCurrent step: {current_step_index + 1} of {len(plan_steps)} — {plan_steps[current_step_index]}\n"
                full_input = f"EXECUTION PLAN:\n{plan}{step_hint}\nWork through the plan step by step.\n\n" + full_input
            if collected_artifacts:
                n = len(collected_artifacts)
                hint = (
                    f"\n[Collected {n} artifact(s) so far. "
                    "If the plan is complete (every requested item produced), respond with plain text and STOP. "
                    "Otherwise delegate ONLY the NEXT unfulfilled step. Never re-delegate for a step that already produced an artifact.]\n\n"
                )
                full_input = hint + full_input
            # Programmatic duplicate prevention: allow N add_report_section when plan has N add steps (multi-figure reports)
            ctx_joined = "\n".join(context_parts)
            add_steps_in_plan = _count_add_report_steps_in_plan(plan_steps)
            add_successes = _count_add_report_successes(ctx_joined)
            if "Added" in ctx_joined and "to report" in ctx_joined and "[visualizer result]" in ctx_joined:
                if add_successes >= add_steps_in_plan:
                    full_input = (
                        "\n[CRITICAL: All add_report_section steps are DONE (plan had "
                        f"{add_steps_in_plan}, {add_successes} succeeded). "
                        "Do NOT delegate add_report_section again. Proceed to preview_report or respond with plain text.]\n\n"
                    ) + full_input
                else:
                    full_input = (
                        f"\n[add_report_section: {add_successes}/{add_steps_in_plan} done. "
                        "Continue with next add_report_section (plot/text) per plan, then preview_report.]\n\n"
                    ) + full_input
            if any(a.get("artifact_type") == "report_html" for a in collected_artifacts):
                full_input = (
                    "\n[CRITICAL: preview_report (report HTML) has ALREADY been collected. "
                    "Plan COMPLETE. Respond with plain text. Do NOT delegate again.]\n\n"
                ) + full_input
            if context_parts:
                full_input = full_input + "\n\nContext from previous steps:\n" + "\n\n".join(context_parts)

            response = self.orchestrator.think_and_act(
                full_input, session_context=session_context,
                stream_callback=stream_callback, stream_reset_callback=stream_reset_callback,
                tool_result_callback=tool_result_callback,
                images=images,
            )
            if isinstance(response, dict) and response.get("status") == "pending_confirmation":
                return response
            response_text = response.get("text", response) if isinstance(response, dict) else response
            if isinstance(response, dict) and response.get("artifact"):
                last_artifact = response["artifact"]

            delegation = self._parse_delegation(response_text)
            if delegation is None:
                # Reflection: verify plan completion before returning (end-to-end persistence)
                # Run whenever we have a plan—even with partial artifacts—to avoid stopping early
                if use_plan and plan_steps:
                    n_artifacts = len(collected_artifacts)
                    reflect_prompt = (
                        f"Your response: {response_text[:500]}...\n\n"
                        f"User requested: {user_message}\n\n"
                        f"Plan has {len(plan_steps)} step(s). We have collected {n_artifacts} artifact(s) so far.\n\n"
                        "CRITICAL: Did we FULLY achieve the user's goal? Complete EVERY item in the plan. "
                        "If ANY step is missing or failed, delegate to the right agent NOW. "
                        "Only respond with plain text (no JSON) when ALL plan items are truly done."
                    )
                    if any(a.get("artifact_type") == "report_html" for a in collected_artifacts):
                        reflect_prompt += (
                            "\n\n[Report preview has been collected. Plan is COMPLETE. "
                            "Respond with plain text ONLY. Do NOT delegate again.]"
                        )
                    try:
                        reflect_response = self.orchestrator.think_and_act(
                            reflect_prompt,
                            context="\n\n".join(context_parts),
                            session_context=session_context,
                            stream_callback=stream_callback, stream_reset_callback=stream_reset_callback,
                            tool_result_callback=tool_result_callback,
                            images=images,
                        )
                        reflect_text = reflect_response.get("text", reflect_response) if isinstance(reflect_response, dict) else str(reflect_response)
                        reflect_delegation = self._parse_delegation(reflect_text)
                        if reflect_delegation is not None:
                            delegation = reflect_delegation
                            response_text = reflect_text
                            self.log_callback("**[ORCHESTRATOR]** Continuing with another step.")
                    except Exception:
                        pass
                if delegation is None:
                    # Truly done: return collected artifacts or final response
                    if collected_artifacts:
                        text = final_text if final_text else f"Produced {len(collected_artifacts)} artifact(s) as requested."
                        return {"text": text, "artifacts": collected_artifacts, "artifact": collected_artifacts[-1]}
                    return {"text": response_text, "artifact": last_artifact}

            agent_name, task = delegation
            agent = self._get_agent(agent_name)
            if agent is None:
                context_parts.append(f"Error: unknown agent '{agent_name}'. Reply to the user.")
                continue

            self.log_callback(f"**[ORCHESTRATOR]** {friendly_delegation(agent_name, task)}")
            agent_context = []
            if intent_override:
                agent_context.append(intent_override.strip())
            if session_str:
                agent_context.append("Session (current loaded data):\n" + session_str)
            if history_str:
                agent_context.append("Previous conversation:\n" + history_str)
            if context_parts:
                agent_context.append("Context from previous steps:\n" + "\n".join(context_parts))

            # When analyst needs to explain or answer: inject collected_artifacts from this turn (not yet in session)
            agent_images = images
            if agent_name == "analyst" and collected_artifacts:
                try:
                    import plotly.io as pio
                    lines = [
                        "ARTIFACTS PRODUCED THIS TURN (explain these—they are not yet in session history):",
                        "Artifacts are numbered 1 (first) to N (last). Use these when the user asks to explain.",
                        "",
                    ]
                    img_list = []
                    for idx, art in enumerate(collected_artifacts, start=1):
                        atype = art.get("artifact_type", "")
                        if atype == "markdown_table":
                            lines.append(f"--- Artifact {idx} (table) ---")
                            lines.append(art.get("artifact_content") or "")
                            lines.append("")
                        elif atype == "plotly_figure":
                            lines.append(f"--- Artifact {idx} (figure) ---")
                            content = art.get("artifact_content")
                            if content:
                                try:
                                    fig = pio.from_json(content if isinstance(content, str) else json.dumps(content))
                                    lines.append(extract_figure_data_for_agent(fig))
                                    img_dict = plotly_figure_to_image_dict(fig)
                                    if img_dict and img_dict.get("data"):
                                        img_list.append(img_dict)
                                except Exception:
                                    lines.append("(Figure data available as image)")
                            lines.append("")
                    if lines:
                        agent_context.append("\n".join(lines))
                    if img_list:
                        agent_images = convert_to_provider_format(img_list[:5], self.provider_name)
                except Exception as e:
                    self.log_callback(f"**[TEAM]** Could not inject artifacts for analyst: {e}")

            result = agent.think_and_act(
                task,
                context="\n\n".join(agent_context) if agent_context else "",
                session_context=session_context,
                stream_callback=stream_callback, stream_reset_callback=stream_reset_callback,
                tool_result_callback=tool_result_callback,
                images=agent_images,
            )
            if isinstance(result, dict) and result.get("status") == "pending_confirmation":
                return result
            result_text = result.get("text", result) if isinstance(result, dict) else result
            if isinstance(result, dict) and result.get("artifact"):
                last_artifact = result["artifact"]
            else:
                # No artifact (e.g. analyst explaining): keep as final text if we have collected artifacts
                if result_text and collected_artifacts:
                    final_text = result_text
            context_parts.append(f"[{agent_name} result]\n{result_text}")

            # Report workflow: add_report_section success (no artifact) — track for multi-figure reports
            # Also treat "already added" / "skipped duplicate" as success (section was in report from prior run)
            _add_success = (
                ("Added" in str(result_text) and "to report" in str(result_text))
                or ("already added" in str(result_text).lower() and "to report" in str(result_text))
                or "skipped duplicate" in str(result_text).lower()
            )
            if agent_name == "visualizer" and not last_artifact and _add_success:
                add_steps = _count_add_report_steps_in_plan(plan_steps)
                add_done = _count_add_report_successes("\n".join(context_parts)) + 1
                if add_done >= add_steps:
                    context_parts.append(
                        "[Step done: add_report_section succeeded]. All report add steps done. "
                        "Delegate to visualizer: preview_report to show the report in chat (if user asked to see it)."
                    )
                else:
                    context_parts.append(
                        f"[Step done: add_report_section succeeded]. {add_done}/{add_steps} add steps done. "
                        "Continue with next add_report_section per plan, then preview_report."
                    )
                if plan_steps and current_step_index < len(plan_steps) - 1:
                    current_step_index += 1

            # Detect step failure: do NOT advance step index; inject retry signal for orchestrator
            step_failed = _is_step_failure(result_text)
            if step_failed:
                context_parts.append(
                    "[STEP FAILED] The previous step returned an error. Do NOT advance. "
                    "Fix the cause (e.g. steward: find correct path, analyst: use different data_dir) and retry, "
                    "or skip this item and continue with the next plan step."
                )
                self.log_callback("**[TEAM]** Step failed, will retry or skip.")
            elif not last_artifact and plan_steps and current_step_index < len(plan_steps) - 1:
                # Intermediate success (e.g. steward found files, analyst computed)—no artifact but step done
                current_step_index += 1

            # Report HTML (preview_report): collect and mark plan complete — prevents re-delegation
            if last_artifact and last_artifact.get("artifact_type") == "report_html":
                if not step_failed and plan_steps and current_step_index < len(plan_steps):
                    current_step_index = len(plan_steps)  # Mark all steps done
                collected_artifacts.append(last_artifact)
                last_artifact = None
                context_parts.append(
                    "[Collected artifact: report preview]. Plan COMPLETE. "
                    "add_report_section and preview_report are DONE. Respond with plain text. Do NOT delegate again."
                )
                continue

            # Table: in report mode, don't collect (table goes only inside report). Otherwise collect.
            if last_artifact and last_artifact.get("artifact_type") == "markdown_table":
                if not step_failed and plan_steps and current_step_index < len(plan_steps) - 1:
                    current_step_index += 1
                if not _is_report_mode(plan_steps):
                    collected_artifacts.append(last_artifact)
                    rejection_count = 0
                    context_parts.append(f"[Collected artifact {len(collected_artifacts)}: table]. Continue with next step in plan.")
                    if len(collected_artifacts) >= max_artifacts_per_turn:
                        return {"text": f"Produced {len(collected_artifacts)} artifact(s).", "artifacts": collected_artifacts, "artifact": collected_artifacts[-1]}
                else:
                    context_parts.append("[Table for report—not shown standalone]. Continue with next step in plan.")
                last_artifact = None
                continue

            # Figure/file: Reviewer validates, then collect and continue (multi-task) or return single
            if last_artifact and last_artifact.get("artifact_type") in ("plotly_figure", "downloadable_file"):
                if rejection_count >= 3:
                    self.log_callback("**[REVIEWER]** Max rejections reached, returning artifact(s).")
                    if collected_artifacts:
                        collected_artifacts.append(last_artifact)
                        return {"text": f"Produced {len(collected_artifacts)} artifact(s).", "artifacts": collected_artifacts, "artifact": collected_artifacts[-1]}
                    return {"text": result_text, "artifact": last_artifact}
                skip_reviewer = last_artifact.get("artifact_type") == "downloadable_file"
                if not skip_reviewer:
                    validation_prompt = (
                        f"User requested: {user_message}\n\n"
                        f"The {agent_name} produced: {result_text[:400]}...\n"
                        f"Artifact type: {last_artifact.get('artifact_type', 'unknown')}.\n\n"
                        "Does this artifact match what the user asked for? "
                        "Reply: APPROVED or REJECTED: [reason]."
                    )
                    try:
                        validation = self.reviewer.think_and_act(
                            validation_prompt,
                            context="You are validating whether the produced output matches the user request. No tools.",
                            session_context=session_context,
                        )
                        val_text = validation.get("text", validation) if isinstance(validation, dict) else str(validation)
                        val_upper = val_text.upper().strip()
                        if val_upper.startswith("REJECTED"):
                            rejection_count += 1
                            last_rejection_reason = val_text[:500]
                            hint = (
                                "Path is ALREADY known from steward. Delegate directly to VISUALIZER—do NOT re-delegate to steward. "
                                "Retry the SAME plot tool with correct parameters. Do not switch to a different page or plot type."
                            )
                            context_parts.append(f"[REVIEWER REJECTED]\n{val_text}\n{hint}\nFix and try again.")
                            self.log_callback("**[REVIEWER]** Rejected artifact, continuing...")
                            last_artifact = None
                            continue
                    except Exception as e:
                        self.log_callback(f"**[REVIEWER]** Validation skipped: {e}")
                # Deduplicate: skip if we already have same figure or same export file
                is_duplicate = False
                if last_artifact.get("artifact_type") == "plotly_figure":
                    new_fp = figure_fingerprint(last_artifact.get("artifact_content"))
                    for prev in collected_artifacts:
                        if prev.get("artifact_type") != "plotly_figure":
                            continue
                        prev_fp = figure_fingerprint(prev.get("artifact_content"))
                        if new_fp and prev_fp and new_fp == prev_fp:
                            is_duplicate = True
                            self.log_callback("**[REVIEWER]** Skipping duplicate figure (same content already collected).")
                            break
                elif last_artifact.get("artifact_type") == "downloadable_file":
                    new_fname = last_artifact.get("filename") or ""
                    for prev in collected_artifacts:
                        if prev.get("artifact_type") == "downloadable_file" and (prev.get("filename") == new_fname or not new_fname):
                            is_duplicate = True
                            self.log_callback("**[REVIEWER]** Skipping duplicate export (same file already collected).")
                            break
                if is_duplicate:
                    # We already have this figure—user's request is satisfied. Return immediately.
                    last_artifact = None
                    text = final_text if final_text else f"Produced {len(collected_artifacts)} artifact(s)."
                    return {"text": text, "artifacts": collected_artifacts, "artifact": collected_artifacts[-1]}
                # Approved: in report mode, don't collect (figure goes only inside report). Otherwise collect.
                if not step_failed and plan_steps and current_step_index < len(plan_steps) - 1:
                    current_step_index += 1
                if not _is_report_mode(plan_steps):
                    collected_artifacts.append(last_artifact)
                    rejection_count = 0
                    art_kind = "exported file" if last_artifact.get("artifact_type") == "downloadable_file" else "figure"
                    if len(collected_artifacts) >= max_artifacts_per_turn:
                        return {"text": f"Produced {len(collected_artifacts)} artifact(s).", "artifacts": collected_artifacts, "artifact": collected_artifacts[-1]}
                    context_parts.append(f"[Collected artifact {len(collected_artifacts)}: {art_kind}]. Continue with next step in plan or respond to user.")
                else:
                    context_parts.append("[Figure for report—not shown standalone]. Continue with next step in plan.")
                last_artifact = None
                continue

        last_part = context_parts[-1] if context_parts else response_text
        if collected_artifacts:
            text = final_text if final_text else f"Produced {len(collected_artifacts)} artifact(s)."
            return {"text": text, "artifacts": collected_artifacts, "artifact": collected_artifacts[-1]}
        if last_artifact:
            return {"text": result_text if last_artifact else "Here is the plot.", "artifact": last_artifact}
        fallback_text = "Max delegation rounds reached. " + (str(last_part)[:500] if last_part else str(response_text))
        return {"text": fallback_text, "artifact": last_artifact}
