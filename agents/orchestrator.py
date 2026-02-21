"""
Orchestrator Agent — Mission planning and coordination.

Breaks down user requests into execution plans.
Migrated from utils/ai_assist/planner.py.
"""

from typing import Any, Dict, Optional

from agents.shared.llm_provider import LLMProvider, get_llm_provider
from agents.shared.config import CHAT_HISTORY_RECENT_MESSAGES, CHAT_HISTORY_PREVIEW_LINES


class OrchestratorAgent:
    """Agent 1: The Manager — plans and coordinates the research mission."""

    def __init__(self, log_func, llm_provider: Optional[LLMProvider] = None):
        self.log = log_func
        self.llm = llm_provider or get_llm_provider()
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get system prompt for planning phase."""
        return """You are a senior software agent planning system.
Your job is to break down user requests into a clear, numbered execution plan.

CRITICAL INSTRUCTIONS:
1. OUTPUT FORMAT: Return ONLY natural language text. NO JSON. NO code blocks.
2. PLAN STRUCTURE: Create a numbered list of steps (1, 2, 3, ...)
3. BE SPECIFIC: Each step should be actionable and clear
4. MATCH THE REQUEST: Include exactly what the user asked for, in the order they asked. One plot -> steps to produce it. Multiple items (plot A, then B, then table, then explain, then save figure 2) -> one step per item, in that order. Add dependency steps (find data, compute) only when needed.
5. USER ORDER: When the user lists items in order, your plan MUST follow that exact order.
6. DEPENDENCIES: Order steps logically (read before modify, compute before plot, find data before plot).
7. NO EXTRAS: Do not add steps the user did not ask for. No "explain" unless they said explain/interpret/describe. No "verify", "return the figure", or "export" unless they said save/export.
8. QUESTIONS/DOUBTS: When the user asks a question, expresses doubt, or wants general chat (no plot/compute/file task), plan: "1. Delegate to analyst with the user's message." For doubts about files: "1. Steward list/verify files. 2. Analyst address the doubt with that context."

EXAMPLES:

User: "What is Kolmogorov turbulence?" or "Are you sure you used all the files?"
Plan:
1. Delegate to analyst: answer the question / address the doubt (include steward's file list in context if doubt about files)

User: "Modify test.py to add a function"
Plan:
1. Check FILE_TREE to locate test.py (may be in root, not /examples/)
2. Read test.py to understand current structure and indentation
3. Identify where to add the new function
4. Modify test.py using search_text/replace_text with correct indentation
5. Verify the modification was applied correctly

User: "Create a new script that processes data"
Plan:
1. Determine the script name and location based on user request
2. Design the script structure and required functionality
3. Create the script file with complete implementation
4. Verify the file was created successfully

User: "What files are in the project?"
Plan:
1. Use search_codebase or read_file to explore project structure
2. Provide a summary of the project organization

User: "Plot Lumley from DNS/512" or "plot subplot B" or "plot diagonal b_ii"
Plan:
1. Steward: find eps_real_validation*.csv in the directory
2. Visualizer: plot the requested subplot (ONE plot only—do not repeat)

User: "Plot spectra from LES/64" or "plot energy spectrum"
Plan:
1. Steward: find spectrum*.dat in the directory
2. Analyst: compute_spectra
3. Visualizer: plot_spectrum (ONE plot only—do not repeat)

User: "Plot spectral isotropy" or "plot IC(k)" or "spectral isotropy page"
Plan:
1. Steward: find isotropy_coeff*.dat in the directory
2. Analyst: compute_spectral_isotropy
3. Visualizer: plot_spectral_isotropy or plot_component_spectra or get_spectral_isotropy_summary (ONE output only—do not repeat)

User: "Plot spectra, then Lumley, then summary table, then explain the physics of all, then save the second figure"
Plan:
1. Find spectrum*.dat -> analyst compute_spectra -> visualizer plot_spectrum
2. Find eps_real_validation*.csv -> visualizer plot_lumley_triangle
3. Find isotropy_coeff*.dat -> analyst compute_spectral_isotropy -> visualizer get_spectral_isotropy_summary
4. Delegate to analyst: explain the physics of all artifacts
5. Delegate to visualizer: export_figure for artifact 2

User: "Write a complete research paper in LaTeX, save it, then compile to PDF"
Plan:
1. Delegate to analyst: generate_content (content_type=paper, output_format=latex), then write_file to save (e.g. exports/paper.tex)
2. Delegate to analyst: compile_latex(filepath=exports/paper.tex) to produce PDF

Remember: Infer from the user's words. Include every item they asked for; add nothing they did not ask for."""

    def plan(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate execution plan for user request.

        Args:
            user_input: User's request
            context: Optional context (file_tree, chat_history, images, etc.)

        Returns:
            Natural language plan as string
        """
        self.log("Orchestrator", "Thinking about how to help...")
        context_str = self._format_context(context)
        prompt = f"""{context_str}

USER REQUEST: {user_input}

Generate a numbered execution plan. Break this down into clear, actionable steps.
Return ONLY the plan text, no JSON, no code blocks."""

        try:
            images = None
            if context and isinstance(context, dict) and "images" in context:
                images = context.get("images")

            response = self.llm.generate(
                prompt,
                system_prompt=self.system_prompt,
                temperature=0.3,
                images=images,
            )

            plan = response.strip()
            if plan.startswith("```"):
                lines = plan.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                plan = "\n".join(lines).strip()

            return plan if plan else "1. Process the user's request"

        except Exception as e:
            self.log("Orchestrator", f"Planning fallback: {e}")
            return f"1. Process user request: {user_input}\n2. Handle any errors that occur"

    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context for planning prompt."""
        lines = []

        if context:
            # Pre-formatted session string (e.g. from UnifiedTeam with data_directory, loaded files)
            if "session_str" in context and context["session_str"]:
                lines.append("=== SESSION CONTEXT ===")
                lines.append(context["session_str"])
                lines.append("")

            if "file_tree" in context:
                lines.append("=== FILE STRUCTURE ===")
                lines.append(context["file_tree"])
                lines.append("")

            if "chat_history" in context and context["chat_history"]:
                lines.append("=== RECENT CONVERSATION ===")
                recent = context["chat_history"][-CHAT_HISTORY_RECENT_MESSAGES:]
                for msg in recent:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if len(content) > CHAT_HISTORY_PREVIEW_LINES:
                        content = content[:CHAT_HISTORY_PREVIEW_LINES] + "..."
                    lines.append(f"{role.title()}: {content}")
                lines.append("")

        return "\n".join(lines) if lines else ""
