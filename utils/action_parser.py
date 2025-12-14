"""
Action Parser - Converts natural language to structured actions
Uses LLM to understand user requests and generate executable actions
"""

import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils.llm_provider import get_llm_provider, LLMProvider


# Action Parser: Converts natural language to structured actions with strict whitelist


class ActionParser:
    """Parses natural language requests into structured actions"""
    
    # Strict action whitelist - must match executor's ALLOWED_ACTIONS
    ALLOWED_ACTIONS = {
        "read_file",
        "extract_section",  # Extract any code section by query (structure-aware)
        "extract_code",  # Alias for extract_section (backward compatibility)
        "search_codebase",  # Content search (grep-style)
        "find_file",  # Filename search (locate files by name)
        "list_dir",
        "modify_file",
        "create_file",
        "delete_file",
        "rename_file",
        "move_file",
        "run_shell_command",
        "execute_code",
        "git_operation",
        "web_search",
        "search_papers",
        "load_data",
        "set_theme",
        "conversational_response",
    }
    
    # Cached file tree (class-level cache)
    _file_tree_cache: Optional[List[str]] = None
    _file_tree_cache_root: Optional[Path] = None
    
    # Required parameters for each action type
    REQUIRED_PARAMS = {
        "read_file": ["filepath"],
        "extract_section": ["filepath", "query"],
        "extract_code": ["filepath", "query"],  # Alias
        "create_file": ["filepath"],
        "modify_file": [],  # Handled specially (needs search_text+replace_text OR new_content/content)
        "rename_file": ["filepath", "new_filepath"],
        "delete_file": ["filepath"],
    }
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None, project_root: Optional[Path] = None):
        self.llm = llm_provider or get_llm_provider()
        # Store project_root for consistent path resolution
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Infer from __file__ location (fallback - but prefer explicit project_root)
            self.project_root = self._get_project_root()
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are an elite AI Developer Agent for the "KI-TURB 3D" application.
Your goal is to autonomously explore, understand, and modify the codebase to fulfill user requests.

CRITICAL INSTRUCTIONS:
1. RESPONSE FORMAT: You must return ONLY a JSON array of actions. No conversational text outside the JSON.
2. UNDERSTAND USER INTENT: Interpret user requests flexibly - users may phrase requests in many ways:
   - "remove", "delete", "drop", "get rid of" → deletion operations
   - "change", "modify", "update", "edit", "replace", "fix" → modification operations
   - "add", "create", "make", "write" → creation operations
   - "show", "open", "read", "view", "display" → read operations
   - "find", "locate", "where is" → file location queries
   - Understand the user's goal and choose appropriate actions flexibly.
3. FILE OPERATIONS - Flexible Approaches:
   - **Reading files:** Choose the best method based on context:
     * For huge files (>500 lines): Use 'extract_section' with a query, or 'read_file' with start_line/end_line
     * For smaller files: 'read_file' or 'extract_section' both work
     * If user mentions line numbers: Use 'read_file' with those line numbers
   - **Modifying files:** Understand what the user wants to change:
     * If user mentions a function/subroutine/class name: Use 'extract_section' to get it, then 'modify_file' to change it
     * If user mentions specific lines: Use 'read_file' with line range, then 'modify_file'
     * If user wants to delete something: Extract or read it first, then use 'modify_file' with empty 'replace_text'
     * If user wants to replace something: Extract or read it first, then use 'modify_file' with new 'replace_text'
   - **Creating files:** User says "create", "add", "make" → use 'create_file'
   - **Deleting files:** User says "delete", "remove" a file → use 'delete_file' (don't read first)
4. AVAILABLE TOOLS - Use flexibly based on user intent:
   - **extract_section**: Extract any code section by query (structure-aware, works with huge files)
     * Example: {"action": "extract_section", "filepath": "[any_file]", "query": "[any_search_term]"}
     * Automatically detects block boundaries - works for functions, subroutines, classes, imports, loops, etc.
   - **read_file**: Read file content (supports optional start_line/end_line for huge files)
     * Example: {"action": "read_file", "filepath": "[any_file]", "start_line": 100, "end_line": 200}
   - **modify_file**: Modify/replace/remove code sections
     * Surgical mode: {"action": "modify_file", "filepath": "[any_file]", "search_text": "[exact_code]", "replace_text": "[new_code]"}
     * Delete mode: Set 'replace_text' to empty string ""
     * Full rewrite: {"action": "modify_file", "filepath": "[any_file]", "new_content": "[full_file]"}
   - **Note:** 'extract_section' and 'read_file' are READ-ONLY. 'modify_file' performs write operations.
   - **IMPORTANT:** All filepaths should come from FILE_TREE. Examples use [any_file] as placeholders - replace with actual paths.
5. **Context Aware:** You have the full file tree in your context. Use it to navigate valid paths.
6. **PATH HANDLING - CRITICAL:**
   - **PROJECT ROOT:** The project root is the directory containing `pages/` and `utils/` subdirectories.
   - **FILE_TREE FORMAT:** All paths in FILE_TREE are prefixed with `APP/` (e.g., `APP/utils/action_executor.py`, `APP/pages/01_AI_Assistant.py`).
   - **WHEN REFERENCING FILES:** Use paths WITHOUT the `APP/` prefix, relative to the project root:
     * ✅ CORRECT: `utils/action_executor.py`, `pages/01_AI_Assistant.py`, `app.py`
     * ❌ WRONG: `APP/utils/action_executor.py`, `APP/pages/01_AI_Assistant.py`
   - **STRIP APP/ PREFIX:** If you see `APP/file.py` in FILE_TREE, use `file.py` in your actions.
   - **EXAMPLES:**
     * To read `APP/utils/action_executor.py` → use `filepath: "utils/action_executor.py"`
     * To read `APP/pages/01_AI_Assistant.py` → use `filepath: "pages/01_AI_Assistant.py"`
     * To read `APP/app.py` → use `filepath: "app.py"`
   - Trust the `FILE_TREE` provided in context - it shows the correct structure with APP/ prefix for clarity.

7. **FILE LOCATION QUERIES:**
   - When user asks "where is file X", "locate file Y", "find file Z", or similar:
     * Use `find_file` (filename search) to locate the file - it searches by filename on disk
     * `search_codebase` searches file contents (grep-style), not file locations
     * Don't guess paths - use `find_file` to get the actual location
   - Example: User asks "where is [filename]"
     * Use: [{"action": "find_file", "filename": "[filename]"}, {"action": "conversational_response", "message": "Based on find_file results: [actual_path]"}]

DISALLOWED ACTIONS:
- Do NOT generate these actions: navigate, modify_plot_style, modify_labels, select_plot_type, export_plot, generate_report.
- Never output these action names even as strings in JSON; use conversational_response instead.
- If the user asks for any of these, respond with conversational_response explaining they are not supported.

APPLICATION INFORMATION:
- App Name: KI-TURB 3D (KI=>khan idrees)
- You are the AI assistant for the KI-TURB 3D (KI=>khan idrees) platform
- When users ask "who are you", "what's your name", "what is this app/platform", "what is KI-TURB 3D (KI=>khan idrees)", or similar questions about the application or yourself, introduce yourself as the AI assistant for KI-TURB 3D (KI=>khan idrees) and explain that KI-TURB 3D (KI=>khan idrees) is a scientific data analysis and visualization platform with an AI chatbot that can answer any questions

STREAMLIT APP CONTEXT:
- This is a Streamlit application with multiple pages in the pages/ directory
- Pages are Python files named like "01_Page_Name.py", "01_AI_Assistant.py", etc.
- The app uses Streamlit's session state (st.session_state) to maintain state across pages
- The app has a theme system with "Light Scientific" and "Dark Scientific" themes
- Data is loaded into session state and can be accessed across pages
- The app structure includes: pages/, utils/, data_readers/, visualizations/, scripts/

AVAILABLE ACTIONS:
- conversational_response: Answer questions, provide information (use "message")
- read_file: Read file content (use "filepath", optional "start_line" and "end_line")
    - Works with ANY file - use actual filepaths from FILE_TREE.
    - For huge files: Use start_line/end_line or use 'extract_section' instead.
    - Example: {"action": "read_file", "filepath": "[any_file]", "start_line": 100, "end_line": 200}
- extract_section: INTELLIGENT EXTRACTOR (READ-ONLY) - Extract any code section by query (structure-aware).
    - Works with ANY file in the project - use actual filepaths from FILE_TREE.
    - Example: {"action": "extract_section", "filepath": "[any_file]", "query": "[any_search_term]"}
    - The query can be any unique string that identifies the section (function name, class name, import statement, keyword, etc.)
    - Automatically detects block boundaries (indentation, braces, keywords) based on language.
    - Returns the exact code block with line numbers.
    - Works for functions, subroutines, classes, imports, loops, configuration blocks, etc.
    - **NOTE:** This is READ-ONLY. To modify/remove, use 'modify_file' with the extracted code as 'search_text'.
- modify_file: Modify/replace/remove code sections (WRITE operation)
    - Works with ANY file - use actual filepaths from FILE_TREE.
    - Surgical edit: {"action": "modify_file", "filepath": "[any_file]", "search_text": "[exact_code]", "replace_text": "[new_code]"}
    - Delete section: Set 'replace_text' to empty string ""
    - Full rewrite: {"action": "modify_file", "filepath": "[any_file]", "new_content": "[full_file]"}
    - System shows diff for approval.
- create_file: Create file (use "filepath", "content"). System asks for confirmation before creating.
- rename_file: Rename/move file (use "filepath", "new_filepath"). System asks for confirmation.
- delete_file: Delete file (use "filepath"). System asks for confirmation.
- run_shell_command: Execute shell command (use "command")
- execute_code: Execute Python code (use "code"). Can access Streamlit via 'import streamlit as st'
- search_codebase: Search code (use "query") - Use this to find code before editing
- list_dir: List directory contents (use "dirpath" or "path") - Use this to explore directory structure
- load_data: Load data directory (use "path") - updates Streamlit session state. Be intelligent: 
  * If user gives a hint (e.g., "DNS", "250", "examples/DNS"), the system will search and show suggestions
  * If exact path doesn't exist, the system will find similar directories and show suggestions
  * You should help the user by explaining what was found and offering to load the closest match
  * For example, if user says "DNS/250" but only "DNS/256" exists, the system will find it and you should suggest loading "DNS/256"
  * Always be helpful and proactive in suggesting alternatives when paths don't match exactly
- set_theme: Change app theme (use "theme" or "theme_name" parameter. Accepts: "dark", "light", "Dark Scientific", "Light Scientific", or any variation)
- git_operation: Git commands (use "operation", "files", "message")
- web_search: Search the web using Google/DuckDuckGo (use "query", optional "num_results")
- search_papers: Search research papers from arXiv/Google Scholar/PubMed (use "query", "source" like "arxiv", optional "max_results")

STREAMLIT-SPECIFIC NOTES:
- When executing code, you can import and use Streamlit: import streamlit as st
- Session state is persistent across page navigations
- Pages are automatically discovered from the pages/ directory
- The current page name is provided in context
- Data is stored in session state and provided in context

File system: Full access to entire system. Absolute paths work anywhere. No restrictions - fully LLM-driven and generalized.

IMPORTANT - CONFIRMATION REQUIREMENTS:
- NEVER set "confirmed": true in your action JSON for create_file, modify_file, rename_file, or delete_file
- These actions ALWAYS require user confirmation - the system will handle confirmation automatically
- Only the user can confirm these actions through the UI - you cannot bypass this security measure

Refuse to hallucinate paths. If you don't see a file in the File Tree, search for it first.

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

USER REQUEST: {user_input}

Generate the next logical step as a JSON array."""

        try:
            # Force JSON mode and use low temperature for precision
            response = self.llm.generate(
                prompt, 
                system_prompt=self.system_prompt, 
                temperature=0.1, 
                format="json"
            )
            
            # Extract JSON from response (handle markdown code blocks)
            json_str = self._extract_json(response)
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Normalize parsed response to list of actions
            actions = self._normalize_parsed_response(parsed)
            
            if not actions:
                return self._handle_conversational_fallback(user_input, context, "No actions generated")
            
            # Validate and normalize actions
            validated = self._validate_actions(actions)
            if not validated["valid"]:
                return self._retry_with_validation_error(user_input, context, validated["errors"])
            
            return self._normalize_actions(validated["actions"])
        
        except json.JSONDecodeError as e:
            # If JSON parsing fails, use LLM to answer conversationally
            return self._handle_conversational_fallback(user_input, context, f"JSON parsing error: {str(e)}")
        except Exception as e:
            # On any error, use LLM to provide helpful conversational response
            return self._handle_conversational_fallback(user_input, context, f"Error: {str(e)}")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response - simplified approach"""
        # First, try to find JSON in code blocks (most common case)
        code_block_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            candidate = code_block_match.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        
        # Try direct JSON parsing (fast path)
        text_stripped = text.strip()
        try:
            json.loads(text_stripped)
            return text_stripped
        except json.JSONDecodeError:
            pass
        
        # Fallback: Use balanced bracket matching for edge cases
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue
            
            # Extract balanced JSON
            candidate = self._extract_balanced_json(text, start_idx, start_char, end_char)
            if candidate:
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
        
        # Final fallback: return as-is (will fail with clear error)
        return text.strip()
    
    def _extract_balanced_json(self, text: str, start_idx: int, start_char: str, end_char: str) -> Optional[str]:
        """Extract balanced JSON starting at start_idx"""
        depth = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start_idx:i+1]
        
        return None
    
    
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
            
            # Structured observations from last execution
            if "observations" in context:
                obs = context["observations"]
                lines.append("\n=== LAST EXECUTION OBSERVATIONS ===")
                
                # Last actions
                if "last_actions" in obs and obs["last_actions"]:
                    lines.append("Last Actions Executed:")
                    for act in obs["last_actions"][:5]:  # Limit to 5 most recent
                        lines.append(f"  - {act.get('action', 'unknown')} on {act.get('filepath', 'N/A')}")
                
                # Last results (success/failure)
                if "last_results" in obs and obs["last_results"]:
                    lines.append("\nLast Results:")
                    for res in obs["last_results"][:5]:  # Limit to 5 most recent
                        status = "✅" if res.get("success") else "❌"
                        lines.append(f"  {status} {res.get('action', 'unknown')}: {res.get('message', '')[:100]}")
                        if "output_snippet" in res:
                            lines.append(f"    Output: {res['output_snippet'][:200]}...")
                
                # Files read (with previews only - full content stored in session state, not in prompt)
                if "files_read" in obs and obs["files_read"]:
                    lines.append("\nFiles Read (available for reference):")
                    
                    for filepath, preview in list(obs["files_read"].items())[:3]:  # Limit to 3 files
                        lines.append(f"  📄 {filepath} ({preview.get('line_count', 0)} lines)")
                        
                        # Include only small excerpt to avoid token explosion
                        # Full content is stored in session state and available via read_file if needed
                        if preview.get("first_lines"):
                            first_lines = preview["first_lines"].split('\n')[:20]  # First 20 lines (increased from 10)
                            lines.append(f"    First lines:\n    " + "\n    ".join(first_lines))
                        if preview.get("last_lines") and preview.get("line_count", 0) > 50:
                            lines.append(f"    ... ({preview['line_count'] - 50} more lines) ...")
                            last_lines = preview["last_lines"].split('\n')[-10:]  # Last 10 lines (increased from 5)
                            lines.append(f"    Last lines:\n    " + "\n    ".join(last_lines))
                
                # Errors
                if "errors" in obs and obs["errors"]:
                    lines.append("\n⚠️ Errors from Last Execution:")
                    for err in obs["errors"][:3]:  # Limit to 3 errors
                        lines.append(f"  - {err.get('action', 'unknown')}: {err.get('message', '')[:150]}")
                
                lines.append("=== End Observations ===\n")
            
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
            
            lines.append("")  # Empty line for readability
        
        # Generate full file tree for context awareness (cached)
        # Use stored project_root (set in __init__) for consistency
        project_root = getattr(self, 'project_root', None)
        if not project_root:
            project_root = self._get_project_root()
        
        tree_lines = self._get_file_tree()
        if tree_lines:
            lines.append("=== CURRENT PROJECT FILE STRUCTURE (FILE_TREE) ===")
            lines.append(f"Project root: {project_root}")
            lines.append("User-visible root: APP/ (logical prefix for paths)")
            lines.append("Filesystem root: directory containing pages/ and utils/")
            lines.append("Note: All paths in FILE_TREE are prefixed with 'APP/' to indicate they're relative to the project root.")
            lines.append("When referencing files, use paths WITHOUT the 'APP/' prefix (e.g., 'utils/action_executor.py', 'pages/01_AI_Assistant.py').")
            lines.append("")
            lines.append("\n".join(tree_lines))
            lines.append("========================================\n")
        else:
            lines.append("=== CURRENT PROJECT FILE STRUCTURE (FILE_TREE) ===")
            lines.append(f"Project root: {project_root}")
            lines.append("(File tree unavailable)")
            lines.append("========================================\n")
        
        lines.append("\n=== END CONTEXT ===")
        
        return "\n".join(lines) if lines else "Streamlit app context available - use your intelligence to discover structure"
    
    def _get_project_root(self) -> Path:
        """Get project root - infer from __file__ location"""
        try:
            current_file = Path(__file__)
            if current_file.name == "action_parser.py":
                # action_parser.py is in utils/, so parent.parent is project root
                project_root = current_file.parent.parent
                # Verify it's the correct root (has pages/ and utils/)
                if (project_root / "pages").exists() and (project_root / "utils").exists():
                    return project_root
        except Exception:
            pass
        
        # Fallback: try to find project root from cwd
        import os
        cwd = Path(os.getcwd())
        if (cwd / "pages").exists() and (cwd / "utils").exists():
            return cwd
        # Try parent
        parent = cwd.parent
        if (parent / "pages").exists() and (parent / "utils").exists():
            return parent
        
        # Last resort: use cwd
        return cwd
    
    def _get_file_tree(self) -> List[str]:
        """Get file tree - cached for performance"""
        # Use stored project_root (set in __init__) for consistency
        # CRITICAL: Never fall back to os.getcwd() - must use same root as UI/executor
        project_root = getattr(self, 'project_root', None)
        if not project_root:
            project_root = self._get_project_root()
            self.project_root = project_root  # Store for future use
        
        # Check if cache is valid
        if (ActionParser._file_tree_cache is not None and 
            ActionParser._file_tree_cache_root == project_root):
            return ActionParser._file_tree_cache
        
        # Generate file tree with APP/ prefix for clarity
        tree_lines = []
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.streamlit', 'myenv', '.venv', 'local_tools'}
        
        try:
            for path in project_root.rglob("*"):
                if path.is_file() and not any(p in path.parts for p in skip_dirs) and not path.name.startswith('.'):
                    rel_path = str(path.relative_to(project_root))
                    # Prefix with APP/ to make it clear this is the project root
                    tree_lines.append(f"APP/{rel_path}")
            
            # Sort for consistency
            tree_lines = sorted(tree_lines)
            
            # Limit to reasonable size (prevent token overflow)
            if len(tree_lines) > 500:
                tree_lines = tree_lines[:500]
            
            # Cache the result
            ActionParser._file_tree_cache = tree_lines
            ActionParser._file_tree_cache_root = project_root
            
            return tree_lines
        except Exception:
            return []
    
    def _normalize_filepath(self, filepath: str) -> str:
        """
        Normalize filepath for consistent comparison - simplified version
        """
        if not filepath:
            return ""
        
        normalized = str(filepath).strip().replace("\\", "/")
        
        # Strip common prefixes
        for prefix in ["./", "APP/"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        # Collapse redundant separators
        normalized = re.sub(r'/+', '/', normalized)
        
        # Remove leading/trailing slashes (for relative paths)
        normalized = normalized.strip('/')
        
        return normalized
    
    def _validate_actions(self, actions: List[Dict]) -> Dict:
        """
        Validate actions against whitelist and enforce rules.
        Returns: {"valid": bool, "actions": List[Dict], "errors": List[str]}
        """
        errors = []
        validated_actions = []
        read_files = {}  # Track which files have been read (using normalized paths)
        
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                errors.append(f"Action {i+1}: Not a dictionary")
                continue
            
            action_type = action.get("action", "").strip().lower()
            
            # Check if action is in whitelist
            if action_type not in self.ALLOWED_ACTIONS:
                errors.append(f"Action {i+1}: '{action_type}' is not allowed. Allowed: {', '.join(sorted(self.ALLOWED_ACTIONS))}")
                continue
            
            # Validate required parameters using helper methods
            param_error = self._validate_action_parameters(action, action_type, i+1)
            if param_error:
                errors.append(param_error)
                continue
            
            # Enforce read-before-modify rule (with normalized filepaths)
            # Auto-insert read_file before modify_file if needed
            if action_type == "read_file":
                filepath = action.get("filepath", "")
                if filepath:
                    normalized_path = self._normalize_filepath(filepath)
                    read_files[normalized_path] = True
                validated_actions.append(action)
            elif action_type == "modify_file":
                filepath = action.get("filepath", "")
                if filepath:
                    normalized_path = self._normalize_filepath(filepath)
                    if normalized_path not in read_files:
                        # Check if it's a full rewrite (has new_content, not search_text)
                        if "new_content" in action or "content" in action:
                            # Full rewrite is allowed without read_file
                            validated_actions.append(action)
                        else:
                            # Surgical edit requires read_file - AUTO-INSERT it (planner repair)
                            read_file_action = {
                                "action": "read_file",
                                "filepath": filepath  # Use original filepath from modify_file
                            }
                            validated_actions.append(read_file_action)
                            read_files[normalized_path] = True  # Mark as read
                            # Now add the modify_file action
                            validated_actions.append(action)
                    else:
                        # File already read, just add modify_file
                        validated_actions.append(action)
                else:
                    # No filepath provided - should have been caught by parameter validation above
                    validated_actions.append(action)
            else:
                # Not a file operation, just add it
                validated_actions.append(action)
        
        return {
            "valid": len(errors) == 0,
            "actions": validated_actions,
            "errors": errors
        }
    
    def _retry_with_validation_error(self, user_input: str, context: Optional[Dict], errors: List[str]) -> List[Dict]:
        """Retry parsing once with validation error feedback - streamlined"""
        error_msg = "\n".join([f"- {e}" for e in errors])
        context_str = self._format_context(context)
        
        retry_prompt = f"""{context_str}

USER REQUEST: {user_input}

VALIDATION ERRORS (fix these):
{error_msg}

CRITICAL: Return corrected JSON array with valid actions only."""

        try:
            response = self.llm.generate(
                retry_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,
                format="json"
            )
            
            json_str = self._extract_json(response)
            parsed = json.loads(json_str)
            actions = self._normalize_parsed_response(parsed)
            
            # Validate again
            validated = self._validate_actions(actions)
            if validated["valid"]:
                return self._normalize_actions(validated["actions"])
            else:
                # Return error message if still invalid
                return self._make_conversational_response(
                    f"❌ Validation failed after retry:\n{chr(10).join(validated['errors'])}"
                )
        except Exception as e:
            return self._make_conversational_response(f"Validation error: {error_msg}\n\nRetry failed: {str(e)}")
    
    def _validate_action_parameters(self, action: Dict, action_type: str, action_num: int) -> Optional[str]:
        """Validate required parameters for an action - helper method to reduce duplication"""
        # Special handling for modify_file
        if action_type == "modify_file":
            if "search_text" not in action and "new_content" not in action and "content" not in action:
                return f"Action {action_num}: modify_file requires 'search_text'/'replace_text' OR 'new_content'/'content'"
            
            if "search_text" in action and "replace_text" not in action:
                return f"Action {action_num}: modify_file with 'search_text' requires 'replace_text'. Provide both for surgical edits."
            
            # Disallow empty content
            for content_key in ["new_content", "content"]:
                if content_key in action:
                    content = action.get(content_key, "")
                    if not isinstance(content, str) or not content.strip():
                        return f"Action {action_num}: modify_file '{content_key}' cannot be empty. Provide actual content or use delete_file to remove a file."
            return None
        
        # Check required parameters for other actions
        required = self.REQUIRED_PARAMS.get(action_type, [])
        for param in required:
            if not action.get(param):
                return f"Action {action_num}: {action_type} requires '{param}'"
        
        return None
    
    def _normalize_actions(self, actions: List[Dict]) -> List[Dict]:
        """Normalize actions - strict validation, enforce whitelist, normalize parameter names"""
        normalized = []
        
        # Parameter name mappings (canonical -> alternatives)
        param_mappings = {
            "list_dir": {
                "dirpath": "path",  # Executor accepts both, but normalize to "path"
                "filepath": "path"   # Also accept filepath as alias
            },
            "read_file": {
                "path": "filepath",
                "file": "filepath"
            },
            "rename_file": {
                "old_filepath": "filepath",
                "source": "filepath",
                "new_path": "new_filepath",
                "target": "new_filepath",
                "destination": "new_filepath"
            }
        }
        
        for action in actions:
            if not isinstance(action, dict):
                continue
            
            action_type = action.get("action", "").strip().lower()
            
            # Only fix truly broken cases (empty or placeholder text)
            if not action_type or action_type.lower() in ["action_name", "none", "null", ""]:
                # Infer from content
                if "message" in action or "response" in action or "answer" in action:
                    action["action"] = "conversational_response"
                elif "code" in action:
                    action["action"] = "execute_code"
                elif "command" in action:
                    action["action"] = "run_shell_command"
                else:
                    action["action"] = "conversational_response"
                    if "message" not in action:
                        for key in ["response", "answer", "text", "content"]:
                            if key in action:
                                action["message"] = str(action[key])
                                break
                        if "message" not in action:
                            action["message"] = "I'm here to help! What would you like to do?"
            else:
                # Enforce whitelist - reject unknown action types
                if action_type not in self.ALLOWED_ACTIONS:
                    # Convert to conversational response with error message
                    action["action"] = "conversational_response"
                    action["message"] = f"Action '{action_type}' is not supported. Please use one of the allowed actions."
                else:
                    action["action"] = action_type
                    
                    # Normalize parameter names for this action type
                    if action_type in param_mappings:
                        for alt_param, canonical_param in param_mappings[action_type].items():
                            if alt_param in action and canonical_param not in action:
                                action[canonical_param] = action.pop(alt_param)
            
            normalized.append(action)
        
        return normalized
    
    def _normalize_parsed_response(self, parsed: Any) -> List[Dict]:
        """Normalize parsed JSON response to list of action dicts"""
        # Handle conversational response directly
        if isinstance(parsed, dict) and parsed.get("action") == "conversational_response":
            return [parsed]
        
        # Convert to list
        if isinstance(parsed, dict):
            actions = [parsed]
        elif isinstance(parsed, list):
            actions = parsed
        else:
            # Unexpected type - treat as conversational
            return [{"action": "conversational_response", "message": str(parsed)}]
        
        # Normalize each action
        normalized = []
        for action in actions:
            if isinstance(action, dict):
                normalized.append(action)
            elif isinstance(action, str):
                normalized.append({"action": "conversational_response", "message": action})
        
        return normalized
    
    def _make_conversational_response(self, message: str) -> List[Dict]:
        """Helper to create conversational response action"""
        return [{"action": "conversational_response", "message": message}]
    
    def _handle_conversational_fallback(self, user_input: str, context: Optional[Dict] = None, error_msg: str = "") -> List[Dict]:
        """Use LLM to answer conversationally when JSON parsing fails or for questions - streamlined"""
        try:
            context_str = self._format_context(context) if context else ""
            fallback_prompt = f"""{context_str}

User: {user_input}

{error_msg if error_msg else ""}

Answer conversationally. Return JSON: [{{"action": "conversational_response", "message": "your answer"}}]"""
            
            response = self.llm.generate(fallback_prompt, system_prompt=self.system_prompt, temperature=0.7)
            
            if not response or not response.strip():
                return self._make_conversational_response("I'm here to help! Could you please rephrase your question?")
            
            # Try to extract and parse JSON
            try:
                json_str = self._extract_json(response)
                if json_str and json_str.strip():
                    parsed = json.loads(json_str)
                    actions = self._normalize_parsed_response(parsed)
                    # Ensure all are conversational
                    for action in actions:
                        if isinstance(action, dict):
                            action["action"] = "conversational_response"
                            if "message" not in action or not action.get("message"):
                                action["message"] = response.strip() or "I'm here to help!"
                    return actions if actions else self._make_conversational_response(response.strip())
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Fallback: use raw response as message
            return self._make_conversational_response(response.strip() or "I'm here to help! What would you like to do?")
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
                return self._make_conversational_response(direct_response)
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
                return self._make_conversational_response(error_message)
    
