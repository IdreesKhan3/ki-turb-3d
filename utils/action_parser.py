"""
Action Parser - Converts natural language to structured actions
Uses LLM to understand user requests and generate executable actions
"""

import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils.llm_provider import get_llm_provider, LLMProvider


# No hardcoded mappings - LLM will discover and understand everything dynamically


class ActionParser:
    """Parses natural language requests into structured actions"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM - fully LLM-driven like Cursor AI"""
        return """You are an intelligent AI coding assistant running inside a Streamlit multi-page application called "KI-TURB 3D".

APPLICATION INFORMATION:
- App Name: KI-TURB 3D
- You are the AI assistant for the KI-TURB 3D platform
- When users ask "who are you", "what's your name", "what is this app/platform", "what is KI-TURB 3D", or similar questions about the application or yourself, introduce yourself as the AI assistant for KI-TURB 3D and explain that KI-TURB 3D is a scientific data analysis and visualization platform with an AI chatbot that can answer any questions

STREAMLIT APP CONTEXT:
- This is a Streamlit application with multiple pages in the pages/ directory
- Pages are Python files named like "01_Page_Name.py", "18_Chatbot.py", etc.
- The app uses Streamlit's session state (st.session_state) to maintain state across pages
- Navigation between pages is done via the "navigate" action with target_page parameter
- The app has a theme system with "Light Scientific" and "Dark Scientific" themes
- Plotting is done using Plotly (plotly.graph_objects)
- Data is loaded into session state and can be accessed across pages
- The app structure includes: pages/, utils/, data_readers/, visualizations/, scripts/

AVAILABLE ACTIONS:
- conversational_response: Answer questions, provide information (use "message")
- read_file: Read file (use "filepath")
- modify_file: Modify file (use "filepath", "new_content" with complete modified file). System shows diff for approval.
- create_file: Create file (use "filepath", "content"). System asks for confirmation before creating.
- rename_file: Rename/move file (use "filepath", "new_filepath"). System asks for confirmation.
- delete_file: Delete file (use "filepath"). System asks for confirmation.
- run_shell_command: Execute shell command (use "command")
- execute_code: Execute Python code (use "code"). Can access Streamlit via 'import streamlit as st'
- search_codebase: Search code (use "query")
- navigate: Navigate to Streamlit page (use "target_page" like "18_Chatbot" or "01_Energy_Spectra")
- load_data: Load data directory (use "path") - updates Streamlit session state. Be intelligent: 
  * If user gives a hint (e.g., "DNS", "250", "examples/DNS"), the system will search and show suggestions
  * If exact path doesn't exist, the system will find similar directories and show suggestions
  * You should help the user by explaining what was found and offering to load the closest match
  * For example, if user says "DNS/250" but only "DNS/256" exists, the system will find it and you should suggest loading "DNS/256"
  * Always be helpful and proactive in suggesting alternatives when paths don't match exactly
- modify_plot_style: Modify plot style settings (use "plot_name", "settings" dict)
- set_theme: Change app theme (use "theme" or "theme_name" parameter. Accepts: "dark", "light", "Dark Scientific", "Light Scientific", or any variation)
- select_plot_type: Select plot type (use "plot_name")
- export_plot: Export plot as image/PDF (use "format", "filename")
- generate_report: Generate HTML/PDF report with plots
- git_operation: Git commands (use "operation", "files", "message")
- web_search: Search the web using Google/DuckDuckGo (use "query", optional "num_results")
- search_papers: Search research papers from arXiv/Google Scholar/PubMed (use "query", "source" like "arxiv", optional "max_results")
- download_file: Download file from URL (use "url", optional "save_path")
- browse_web: Browse and extract content from a web page (use "url")

STREAMLIT-SPECIFIC NOTES:
- When executing code, you can import and use Streamlit: import streamlit as st
- Session state is persistent across page navigations
- Pages are automatically discovered from the pages/ directory
- The current page name is provided in context
- Available plots and data are stored in session state and provided in context

File system: Full access to entire system. Absolute paths work anywhere. No restrictions - fully LLM-driven and generalized.

IMPORTANT - CONFIRMATION REQUIREMENTS:
- NEVER set "confirmed": true in your action JSON for create_file, modify_file, rename_file, or delete_file
- These actions ALWAYS require user confirmation - the system will handle confirmation automatically
- Only the user can confirm these actions through the UI - you cannot bypass this security measure

Analyze the user's prompt and choose the appropriate action(s). Use your intelligence to understand intent. When users ask about the app, pages, or navigation, you understand this is a Streamlit multi-page application.

MULTIPLE ACTIONS SUPPORT:
- You can return MULTIPLE actions in a single response as a JSON array
- For example, if user says "move file1.txt to folder/, delete file2.txt, and create new_file.py", return:
  [
    {"action": "rename_file", "filepath": "file1.txt", "new_filepath": "folder/file1.txt"},
    {"action": "delete_file", "filepath": "file2.txt"},
    {"action": "create_file", "filepath": "new_file.py", "content": "..."}
  ]
- Actions are executed SEQUENTIALLY in the order you provide them
- Each action will show its own result (success/failure) to the user
- Actions requiring confirmation (create_file, modify_file, rename_file, delete_file) will be shown for user approval before execution

Return JSON: [{"action": "action_name", ...}] - can be a single action or multiple actions in an array."""

    def parse(self, user_input: str, context: Optional[Dict] = None) -> List[Dict]:
        """
        Parse user input into structured actions
        
        Args:
            user_input: Natural language request
            context: Current app context (page, plots, chat history, etc.)
        
        Returns:
            List of action dictionaries
        """
        context_str = self._format_context(context)
        
        prompt = f"""{context_str}

User: {user_input}

Return JSON: [{{"action": "action_name", ...}}]"""

        try:
            # Use higher temperature for better reasoning and code generation
            response = self.llm.generate(prompt, system_prompt=self.system_prompt, temperature=0.7)
            
            # Extract JSON from response (handle markdown code blocks)
            json_str = self._extract_json(response)
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Handle if LLM returned a conversational response directly
            if isinstance(parsed, dict) and parsed.get("action") == "conversational_response":
                return [parsed]
            
            # Ensure it's a list
            if isinstance(parsed, dict):
                actions = [parsed]
            elif isinstance(parsed, list):
                actions = parsed
            else:
                # If parsing returned something unexpected, treat as conversational
                return [{"action": "conversational_response", "message": str(parsed)}]
            
            # Normalize actions (only if they're dicts)
            normalized = []
            for action in actions:
                if isinstance(action, dict):
                    normalized.append(action)
                elif isinstance(action, str):
                    # If action is a string, treat as conversational
                    normalized.append({"action": "conversational_response", "message": action})
            
            if normalized:
                return self._normalize_actions(normalized)
            else:
                # No actions found - treat as conversational query
                return self._handle_conversational_fallback(user_input, context, "No actions generated")
        
        except json.JSONDecodeError as e:
            # If JSON parsing fails, use LLM to answer conversationally
            return self._handle_conversational_fallback(user_input, context, f"JSON parsing error: {str(e)}")
        except Exception as e:
            # On any error, use LLM to provide helpful conversational response
            return self._handle_conversational_fallback(user_input, context, f"Error: {str(e)}")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response (handle code blocks)"""
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find JSON array or object
        json_match = re.search(r'(\[.*?\]|\{.*?\})', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Return as-is if no JSON found
        return text.strip()
    
    def _format_context(self, context: Optional[Dict]) -> str:
        """Format context for LLM prompt - discover project structure dynamically"""
        lines = []
        
        # Add Streamlit app header
        lines.append("=== STREAMLIT MULTI-PAGE APPLICATION ===")
        
        if context:
            # Include chat history first for conversation continuity
            if "chat_history" in context and context["chat_history"]:
                lines.append("\n=== Previous Conversation ===")
                for msg in context["chat_history"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # Preserve code blocks fully - they're critical for understanding "the above code"
                    # Check if message contains code blocks
                    if "```" in content:
                        # For messages with code, preserve the full code blocks
                        # but truncate non-code text if message is very long
                        if len(content) > 3000:
                            # Try to preserve code blocks while truncating
                            parts = content.split("```")
                            preserved = []
                            for i, part in enumerate(parts):
                                if i % 2 == 1:  # Code block content
                                    preserved.append(f"```{part}```")
                                else:  # Regular text
                                    if len(part) > 200:
                                        preserved.append(part[:200] + "...")
                                    else:
                                        preserved.append(part)
                            content = "".join(preserved)
                    elif len(content) > 1000:
                        # For non-code messages, truncate more aggressively
                        content = content[:1000] + "..."
                    
                    lines.append(f"{role.title()}: {content}")
                lines.append("=== End Previous Conversation ===\n")
            
            # Streamlit app state
            lines.append("=== STREAMLIT APP STATE ===")
            if "current_page" in context:
                current_page = context['current_page']
                lines.append(f"Current Streamlit page: {current_page}")
                # Extract page number and name for better context
                if "_" in current_page:
                    page_parts = current_page.split("_", 1)
                    if len(page_parts) == 2:
                        lines.append(f"  Page number: {page_parts[0]}")
                        lines.append(f"  Page name: {page_parts[1].replace('.py', '')}")
            
            if "data_loaded" in context:
                lines.append(f"Data loaded in session state: {context['data_loaded']}")
                if context.get("data_loaded") and "data_directory" in context:
                    lines.append(f"  Active data directory: {context.get('data_directory', 'N/A')}")
            
            if "available_plots" in context and context["available_plots"]:
                plots = context["available_plots"]
                lines.append(f"Available plots in session state: {', '.join(plots)}")
                if "available_plot_settings" in context:
                    settings = context.get("available_plot_settings", [])
                    if settings:
                        lines.append(f"  Plot settings available: {', '.join(settings)}")
            
            lines.append("")  # Empty line for readability
        
        # Dynamically discover project structure - provide rich context like Cursor AI
        try:
            # Use current working directory as project root (where user runs app from)
            import os
            project_root = Path(os.getcwd())
            
            lines.append("=== STREAMLIT APP STRUCTURE ===")
            
            # Discover pages directory (Streamlit multi-page app)
            pages_dir = project_root / "pages"
            if pages_dir.exists():
                pages = [f.name for f in pages_dir.glob("*.py") if f.name.startswith(("0", "1"))]
                if pages:
                    sorted_pages = sorted(pages)
                    lines.append(f"Streamlit pages available: {len(sorted_pages)} pages")
                    # Show first 15 pages with better formatting
                    for page in sorted_pages[:15]:
                        page_name = page.replace(".py", "").replace("_", " ", 1)
                        lines.append(f"  - {page_name}")
                    if len(sorted_pages) > 15:
                        lines.append(f"  ... and {len(sorted_pages) - 15} more pages")
                    lines.append("  (Use 'navigate' action with target_page like '18_Chatbot' to switch pages)")
            
            # Discover common directories
            common_dirs = []
            for dir_name in ["pages", "utils", "data_readers", "visualizations", "scripts", "SRC", "hdf5_lib"]:
                if (project_root / dir_name).exists():
                    common_dirs.append(dir_name)
            if common_dirs:
                lines.append(f"\nProject directories: {', '.join(common_dirs)}")
                if "utils" in common_dirs:
                    lines.append("  - utils/: Contains app utilities (theme_config, export, report_builder, etc.)")
                if "pages" in common_dirs:
                    lines.append("  - pages/: Streamlit multi-page app pages")
            
            # Discover file patterns in root
            root_files = [f.name for f in project_root.glob("*.py") if f.is_file()][:10]
            if root_files:
                lines.append(f"\nRoot Python files: {', '.join(root_files)}")
            
            # Check for main app file
            main_files = ["app.py", "main.py", "streamlit_app.py", "Home.py"]
            for main_file in main_files:
                if (project_root / main_file).exists():
                    lines.append(f"  Main app entry point: {main_file}")
                    break
                
        except Exception as e:
            lines.append(f"Note: Could not fully discover project structure ({str(e)})")
        
        lines.append("\n=== END CONTEXT ===")
        
        return "\n".join(lines) if lines else "Streamlit app context available - use your intelligence to discover structure"
    
    def _normalize_actions(self, actions: List[Dict]) -> List[Dict]:
        """Normalize actions - fully LLM-driven, minimal processing, trust LLM intelligence"""
        normalized = []
        
        for action in actions:
            # Skip if not a dict
            if not isinstance(action, dict):
                continue
            
            # Fully LLM-driven: Preserve whatever action type LLM proposed
            # Only fix obvious mistakes, but trust LLM's creativity
            action_type = action.get("action", "").strip()
            
            # Only fix truly broken cases (empty or placeholder text)
            if not action_type or action_type.lower() in ["action_name", "none", "null", ""]:
                # If LLM forgot to set action but provided content, infer from content
                if "message" in action or "response" in action or "answer" in action:
                    action["action"] = "conversational_response"
                elif "code" in action:
                    action["action"] = "execute_code"
                elif "command" in action:
                    action["action"] = "run_shell_command"
                else:
                    # Default to conversational but preserve LLM's content
                    action["action"] = "conversational_response"
                    if "message" not in action:
                        # Try to extract any meaningful content
                        for key in ["response", "answer", "text", "content"]:
                            if key in action:
                                action["message"] = str(action[key])
                                break
                        if "message" not in action:
                            action["message"] = "I'm here to help! What would you like to do?"
            else:
                # Trust LLM - preserve the action type exactly as proposed
                # This allows LLM to create new action types
                action["action"] = action_type
            
            normalized.append(action)
        
        return normalized
    
    def _handle_conversational_fallback(self, user_input: str, context: Optional[Dict] = None, error_msg: str = "") -> List[Dict]:
        """Use LLM to answer conversationally when JSON parsing fails or for questions"""
        try:
            context_str = self._format_context(context) if context else ""
            fallback_prompt = f"""{context_str}

User: {user_input}

{error_msg if error_msg else ""}

Answer the user's question or request conversationally. Return JSON: [{{"action": "conversational_response", "message": "your answer here"}}]"""
            
            response = self.llm.generate(fallback_prompt, system_prompt=self.system_prompt, temperature=0.7)
            
            # If response is empty, provide default
            if not response or not response.strip():
                return [{"action": "conversational_response", "message": "I'm here to help! Could you please rephrase your question?"}]
            
            json_str = self._extract_json(response)
            
            # If no JSON found, use the raw response as message
            if not json_str or json_str.strip() == "":
                return [{"action": "conversational_response", "message": response.strip() if response.strip() else "I'm here to help! What would you like to do?"}]
            
            parsed = json.loads(json_str)
            
            if isinstance(parsed, dict):
                # Ensure it's conversational_response
                if parsed.get("action") != "conversational_response":
                    parsed["action"] = "conversational_response"
                # Ensure message exists
                if "message" not in parsed or not parsed.get("message"):
                    parsed["message"] = response.strip() if response.strip() else "I'm here to help!"
                return [parsed]
            elif isinstance(parsed, list):
                # Normalize all to conversational_response if needed
                normalized = []
                for item in parsed:
                    if isinstance(item, dict):
                        if item.get("action") != "conversational_response":
                            item["action"] = "conversational_response"
                        if "message" not in item or not item.get("message"):
                            item["message"] = response.strip() if response.strip() else "I'm here to help!"
                        normalized.append(item)
                    elif isinstance(item, str):
                        normalized.append({"action": "conversational_response", "message": item})
                return normalized if normalized else [{"action": "conversational_response", "message": response}]
            else:
                # Use raw response as message
                return [{"action": "conversational_response", "message": response if response else "I'm here to help! What would you like to do?"}]
        except Exception as e:
            # Final fallback - use raw LLM response or default message
            error_str = str(e)
            try:
                # Try to get LLM response directly
                direct_response = self.llm.generate(
                    f"User asked: {user_input}\n\nProvide a helpful answer.",
                    system_prompt="You are a helpful AI assistant. Answer the user's question directly and conversationally.",
                    temperature=0.7
                )
                return [{"action": "conversational_response", "message": direct_response}]
            except Exception as llm_error:
                # Show the actual error so user knows what went wrong
                llm_error_str = str(llm_error)
                error_message = f"❌ LLM Error: {llm_error_str}"
                if "Gemini" in llm_error_str or "GOOGLE_API_KEY" in llm_error_str:
                    error_message += "\n\n💡 **Gemini Setup:** Check that:\n"
                    error_message += "- GOOGLE_API_KEY is set correctly\n"
                    error_message += "- google-generativeai package is installed\n"
                    error_message += "- API key is valid and has quota remaining"
                elif "quota" in llm_error_str.lower() or "rate limit" in llm_error_str.lower():
                    error_message += "\n\n⚠️ **Quota/Rate Limit:** You may have exceeded your API quota or rate limit."
                return [{"action": "conversational_response", "message": error_message}]
    
    def _rule_based_parse(self, user_input: str) -> List[Dict]:
        """Fallback - use LLM to generate response if JSON parsing fails"""
        return self._handle_conversational_fallback(user_input, None)

