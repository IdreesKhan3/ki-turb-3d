"""
Planner - Generates execution plans before action execution
Separates reasoning from actions (Cursor-style architecture)
"""

from typing import Dict, Optional
from utils.llm_provider import LLMProvider, get_llm_provider


class Planner:
    """Generates natural language execution plans"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for planning phase"""
        return """You are a senior software agent planning system.
Your job is to break down user requests into a clear, numbered execution plan.

CRITICAL INSTRUCTIONS:
1. OUTPUT FORMAT: Return ONLY natural language text. NO JSON. NO code blocks.
2. PLAN STRUCTURE: Create a numbered list of steps (1, 2, 3, ...)
3. BE SPECIFIC: Each step should be actionable and clear
4. CONSIDER DEPENDENCIES: Order steps logically (read before modify, etc.)
5. CHECK FILE PATHS: If user mentions files, note that paths must be verified against FILE_TREE
6. BE THOROUGH: Break complex tasks into smaller steps

EXAMPLES:

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

Remember: This is PLANNING only. You are NOT executing anything yet.
Be strategic and think through the entire task before acting."""

    def plan(self, user_input: str, context: Optional[Dict] = None) -> str:
        """
        Generate execution plan for user request
        
        Args:
            user_input: User's request
            context: Optional context (file tree, chat history, etc.)
        
        Returns:
            Natural language plan as string
        """
        context_str = self._format_context(context)
        prompt = f"""{context_str}

USER REQUEST: {user_input}

Generate a numbered execution plan. Break this down into clear, actionable steps.
Return ONLY the plan text, no JSON, no code blocks."""

        try:
            # Extract images from context if available (for vision-capable models)
            images = None
            if context and isinstance(context, dict) and "images" in context:
                images = context.get("images")
            
            # NO JSON constraint - free text for reasoning
            response = self.llm.generate(
                prompt, 
                system_prompt=self.system_prompt, 
                temperature=0.3,  # Lower temperature for more focused planning
                images=images  # Pass images if available
            )
            
            # Clean up response (remove any accidental JSON markers)
            plan = response.strip()
            if plan.startswith("```"):
                # Remove code block markers if LLM added them
                lines = plan.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                plan = "\n".join(lines).strip()
            
            return plan if plan else "1. Process the user's request"
            
        except Exception as e:
            # Fallback plan on error
            return f"1. Process user request: {user_input}\n2. Handle any errors that occur"
    
    def _format_context(self, context: Optional[Dict]) -> str:
        """Format context for planning prompt"""
        lines = []
        
        if context:
            # Include file tree if available (from ActionParser context format)
            if "file_tree" in context:
                lines.append("=== FILE STRUCTURE ===")
                lines.append(context["file_tree"])
                lines.append("")
            
            # Include recent chat history for context
            if "chat_history" in context and context["chat_history"]:
                lines.append("=== RECENT CONVERSATION ===")
                recent = context["chat_history"][-3:]  # Last 3 messages for planning
                for msg in recent:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    lines.append(f"{role.title()}: {content}")
                lines.append("")
        
        return "\n".join(lines) if lines else ""

