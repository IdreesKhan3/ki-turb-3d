"""
AI Chatbot Assistant - LLM-Powered Agent
Natural language interface with full control over app functionalities
"""

import streamlit as st
from pathlib import Path
import sys
import json
import traceback
import os
import difflib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from utils.llm_provider import get_llm_provider, get_available_providers
from utils.action_parser import ActionParser
from utils.action_executor import ActionExecutor

st.set_page_config(page_icon="🤖", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot_llm_provider" not in st.session_state:
    st.session_state.chatbot_llm_provider = "ollama"
if "chatbot_parser" not in st.session_state:
    st.session_state.chatbot_parser = None
if "chatbot_executor" not in st.session_state:
    st.session_state.chatbot_executor = ActionExecutor()

def get_app_context() -> dict:
    """Get current app context for LLM"""
    context = {
        "data_loaded": st.session_state.get("data_loaded", False),
        "current_page": "18_Chatbot",
    }
    
    # Get available plots based on loaded data
    data_dirs = st.session_state.get("data_directories", [])
    if data_dirs:
        context["data_directory"] = str(Path(data_dirs[0]).name)
        context["data_directory_full"] = str(data_dirs[0])
        
        # Discover available plot types from session state dynamically
        plot_styles = st.session_state.get("plot_styles", {})
        if plot_styles:
            context["available_plots"] = list(plot_styles.keys())
        
        # Discover plot style settings if available
        if plot_styles:
            # Get settings from first plot to show what's available
            first_plot = list(plot_styles.keys())[0] if plot_styles else None
            if first_plot and plot_styles[first_plot]:
                context["available_plot_settings"] = list(plot_styles[first_plot].keys())
    
    # Include chat history - provider-aware limits
    chat_history = st.session_state.get("chat_history", [])
    if chat_history:
        provider = st.session_state.get("chatbot_llm_provider", "ollama")
        
        # Ollama is free and local - use more messages (no cost concern)
        # Paid APIs - use fewer messages to save costs
        if provider == "ollama":
            # Ollama: up to 50 messages (25 exchanges) - generous since it's free
            max_messages = 50
        else:
            # Paid APIs (OpenAI, Anthropic, Gemini): limit to 12 messages to save costs
            max_messages = 12
        
        context["chat_history"] = chat_history[-max_messages:]
    
    return context

def main():
    inject_theme_css()
    
    # Add custom CSS for chat interface
    st.markdown("""
    <style>
    /* Chat container styling */
    .stChatMessage {
        padding: 1rem;
    }
    
    /* Input area styling */
    [data-testid="stForm"] {
        position: sticky;
        bottom: 0;
        background-color: var(--background-color);
        padding: 1rem 0;
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 AI Chatbot Assistant")
    st.markdown("**LLM-Powered**")
    
    # Initialize LLM provider and parser
    provider_name = st.session_state.chatbot_llm_provider
    
    # Sidebar - LLM Provider Selection
    with st.sidebar:
        st.subheader("⚙️ LLM Provider")
        
        available_providers = get_available_providers()
        provider_options = []
        provider_status = {}
        
        for provider in ["ollama", "gemini", "openai", "anthropic"]:
            is_available = available_providers.get(provider, False)
            status_icon = "✅" if is_available else "❌"
            status_text = "Available" if is_available else "Not configured"
            provider_options.append(f"{status_icon} {provider.title()}")
            provider_status[provider] = is_available
        
        selected_provider_display = st.selectbox(
            "Select Provider",
            options=provider_options,
            index=["ollama", "gemini", "openai", "anthropic"].index(provider_name) if provider_name in ["ollama", "gemini", "openai", "anthropic"] else 0,
            help="Ollama is free and unlimited. Others require API keys."
        )
        
        # Extract provider name from selection
        selected_provider = selected_provider_display.split()[-1].lower()
        
        if selected_provider != provider_name:
            st.session_state.chatbot_llm_provider = selected_provider
            st.session_state.chatbot_parser = None  # Reset parser to force reinitialization
            st.info(f"🔄 Switched to {selected_provider.title()}. Parser will be reinitialized.")
        
        # Show provider status
        if provider_status.get(selected_provider, False):
            st.success(f"✅ {selected_provider.title()} is ready")
            # Debug: Show if API key is set (for Gemini)
            if selected_provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    # Show first/last few chars for verification (security: don't show full key)
                    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
                    st.caption(f"🔑 API Key detected: {masked_key}")
                else:
                    st.warning("⚠️ GOOGLE_API_KEY not found in environment")
        else:
            if selected_provider == "ollama":
                st.warning("⚠️ Ollama not running. Install: curl -fsSL https://ollama.com/install.sh | sh")
            elif selected_provider == "gemini":
                st.warning("⚠️ Gemini API key not set. Set GOOGLE_API_KEY environment variable:\n```bash\nexport GOOGLE_API_KEY='your-api-key-here'\n```")
            elif selected_provider == "openai":
                st.warning("⚠️ OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
            elif selected_provider == "anthropic":
                st.warning("⚠️ Anthropic API key not set. Set ANTHROPIC_API_KEY environment variable.")
            else:
                st.warning(f"⚠️ {selected_provider.title()} API key not set. Set environment variable.")
        
        st.markdown("---")
        st.subheader("📚 Available Actions")
        st.markdown("""
        **Navigation:**
        - "Go to energy spectra page"
        - "Navigate to structure functions"
        
        **Plotting:**
        - "Plot energy spectrum"
        - "Show time evolution"
        
        **Settings:**
        - "Change background to black"
        - "Set x label to 'Wavenumber k'"
        - "Change theme to dark"
        
        **Export:**
        - "Export plot as PDF"
        - "Download figure as PNG"
        
        **Reports:**
        - "Generate report with all plots"
        
        **Code & Scripts:**
        - "Create a new script to process data"
        - "Modify utils/export.py to add new format"
        - "Read pages/07_Energy_Spectra.py"
        - "Delete scripts/old_file.py"
        - "Execute Python code: print('Hello')"
        
        **Shell & Git:**
        - "Run: ls -la scripts/"
        - "Search for 'def plot_energy' in codebase"
        - "Git status"
        - "Git commit with message 'Update code'"
        """)
        
        st.markdown("---")
        st.subheader("💡 Example Queries")
        examples = [
            "Go to energy spectra page",
            "Change background to black",
            "Export plot as PDF",
            "Create a new data processing script",
            "Read utils/export.py",
            "Generate report"
        ]
        for example in examples:
            if st.button(example, key=f"example_{example}", width='stretch'):
                st.session_state.example_query = example
    
    # Initialize parser with selected provider (use session state, not local variable)
    try:
        current_provider = st.session_state.chatbot_llm_provider
        # Reinitialize parser if it doesn't exist or if provider changed
        # (parser is reset to None when provider changes in sidebar)
        if st.session_state.chatbot_parser is None:
            llm_provider = get_llm_provider(current_provider)
            st.session_state.chatbot_parser = ActionParser(llm_provider=llm_provider)
        parser = st.session_state.chatbot_parser
        
        # Debug: Verify parser is using correct provider
        if st.session_state.get("show_debug", False):
            parser_provider_type = type(parser.llm).__name__
            st.caption(f"🔍 Debug: Parser using {parser_provider_type} (expected: {current_provider})")
        executor = st.session_state.chatbot_executor
    except Exception as e:
        st.error(f"Failed to initialize LLM provider: {e}")
        st.info("💡 Make sure Ollama is running or API keys are set. Check sidebar for details.")
        return
    
    # Display chat history first
    st.markdown("---")
    st.subheader("💬 Conversation History")
    
    if not st.session_state.chat_history:
        st.info("👆 Start a conversation!")
    else:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    
                    # Show executed actions in expander (for debugging)
                    if "actions" in msg and st.session_state.get("show_actions", False):
                        with st.expander("🔍 View executed actions"):
                            st.json(msg["actions"])
    
    # Show pending file renames requiring confirmation
    if st.session_state.get("pending_renames"):
        st.markdown("---")
        st.subheader("📝 Pending File Renames - Review Required")
        pending = st.session_state.pending_renames.copy()
        for i, rename in enumerate(pending):
            filename = rename.get("filename", "Unknown")
            new_filename = rename.get("new_filename", "Unknown")
            filepath = rename.get("filepath", "")
            new_filepath = rename.get("new_filepath", "")
            
            st.warning(f"**File to rename:** `{filename}` → `{new_filename}`")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Accept Rename", key=f"accept_rename_{i}", type="primary", width='stretch'):
                    action = rename["action"].copy()
                    action["confirmed"] = True
                    result = executor.execute(action)
                    if result.get("success"):
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"✅ {result.get('message', 'File renamed')}"
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ {result.get('message', 'Rename failed')}"
                        })
                    st.session_state.pending_renames = [r for r in st.session_state.pending_renames if r != rename]
                    if not st.session_state.pending_renames:
                        del st.session_state.pending_renames
                    st.rerun()
            with col2:
                if st.button("❌ Reject Rename", key=f"reject_rename_{i}", width='stretch'):
                    st.session_state.pending_renames = [r for r in st.session_state.pending_renames if r != rename]
                    if not st.session_state.pending_renames:
                        del st.session_state.pending_renames
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Rename rejected: {filename}"
                    })
                    st.rerun()
            st.markdown("---")
    
    # Show pending file modifications requiring confirmation
    if st.session_state.get("pending_modifications"):
        st.markdown("---")
        st.subheader("📝 Pending File Modifications - Review Required")
        pending = st.session_state.pending_modifications.copy()
        for i, modification in enumerate(pending):
            filename = modification.get("filename", "Unknown")
            filepath = modification.get("filepath", "")
            current_content = modification.get("current_content", "")
            new_content = modification.get("new_content", "")
            
            # Handle any data type - convert to string safely
            if not isinstance(current_content, str):
                current_content = str(current_content) if current_content else ""
            if not isinstance(new_content, str):
                new_content = str(new_content) if new_content else ""
            
            st.warning(f"**File to modify:** `{filename}`")
            
            # Create diff view only if both are strings and non-empty
            if current_content and new_content:
                try:
                    current_lines = current_content.splitlines(keepends=True)
                    new_lines = new_content.splitlines(keepends=True)
                    diff = difflib.unified_diff(
                        current_lines,
                        new_lines,
                        fromfile=f"Original: {filename}",
                        tofile=f"Modified: {filename}",
                        lineterm=""
                    )
                    diff_text = "".join(diff)
                    
                    # Show diff in expandable section
                    with st.expander(f"📊 View Diff for {filename}", expanded=True):
                        if diff_text.strip():
                            st.code(diff_text, language="diff")
                        else:
                            st.info("No changes detected")
                except Exception as e:
                    st.warning(f"Could not generate diff: {str(e)}")
            
            # Show side-by-side comparison in tabs
            if current_content or new_content:
                tab1, tab2 = st.tabs(["📄 Current File", "✨ Proposed Changes"])
                with tab1:
                    if current_content:
                        lang = "python" if filename.endswith(".py") else ("fortran" if filename.endswith((".f90", ".f", ".f95")) else "text")
                        st.code(current_content, language=lang)
                    else:
                        st.info("No current content")
                with tab2:
                    if new_content:
                        lang = "python" if filename.endswith(".py") else ("fortran" if filename.endswith((".f90", ".f", ".f95")) else "text")
                        st.code(new_content, language=lang)
                    else:
                        st.info("No proposed content")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Accept Changes", key=f"accept_modify_{i}", type="primary", width='stretch'):
                    # Execute modification with confirmation
                    action = modification["action"].copy()
                    action["confirmed"] = True
                    result = executor.execute(action)
                    if result.get("success"):
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"✅ {result.get('message', 'File modified')}"
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ {result.get('message', 'Modification failed')}"
                        })
                    # Remove this modification from pending list
                    st.session_state.pending_modifications = [m for m in st.session_state.pending_modifications if m != modification]
                    if not st.session_state.pending_modifications:
                        del st.session_state.pending_modifications
                    st.rerun()
            with col2:
                if st.button("❌ Reject Changes", key=f"reject_modify_{i}", width='stretch'):
                    # Remove this modification from pending list
                    st.session_state.pending_modifications = [m for m in st.session_state.pending_modifications if m != modification]
                    if not st.session_state.pending_modifications:
                        del st.session_state.pending_modifications
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Modifications rejected: {filename}"
                    })
                    st.rerun()
            st.markdown("---")
    
    # Show pending deletions requiring confirmation
    if st.session_state.get("pending_deletions"):
        st.markdown("---")
        st.subheader("⚠️ Pending Deletions - Confirmation Required")
        pending = st.session_state.pending_deletions.copy()
        for i, deletion in enumerate(pending):
            filename = deletion.get("filename", "Unknown")
            filepath = deletion.get("filepath", "")
            st.warning(f"**File to delete:** `{filename}`")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Confirm Delete", key=f"confirm_delete_{i}", type="primary", width='stretch'):
                    # Execute deletion with confirmation
                    action = deletion["action"].copy()
                    action["confirmed"] = True
                    result = executor.execute(action)
                    if result.get("success"):
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"✅ {result.get('message', 'File deleted')}"
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ {result.get('message', 'Delete failed')}"
                        })
                    # Remove this deletion from pending list
                    st.session_state.pending_deletions = [d for d in st.session_state.pending_deletions if d != deletion]
                    if not st.session_state.pending_deletions:
                        del st.session_state.pending_deletions
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", key=f"cancel_delete_{i}", width='stretch'):
                    # Remove this deletion from pending list
                    st.session_state.pending_deletions = [d for d in st.session_state.pending_deletions if d != deletion]
                    if not st.session_state.pending_deletions:
                        del st.session_state.pending_deletions
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Deletion cancelled: {filename}"
                    })
                    st.rerun()
    
    # Show export/download if available
    if st.session_state.get("_chatbot_export_file"):
        st.markdown("---")
        st.subheader("📥 Export Ready")
        export_file = st.session_state._chatbot_export_file
        export_format = st.session_state.get("_chatbot_export_format", "pdf")
        
        if Path(export_file).exists():
            with open(export_file, "rb") as f:
                st.download_button(
                    f"📥 Download {Path(export_file).name}",
                    f.read(),
                    file_name=Path(export_file).name,
                    mime=f"application/{export_format}" if export_format == "pdf" else f"image/{export_format}"
                )
        
        if st.button("Clear export", key="clear_export"):
            del st.session_state._chatbot_export_file
            if "_chatbot_export_format" in st.session_state:
                del st.session_state._chatbot_export_format
            st.rerun()
    
    # Clear button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History", width='stretch'):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        show_actions = st.checkbox("Show executed actions", value=st.session_state.get("show_actions", False))
        st.session_state.show_actions = show_actions
    
    # Debug: Test LLM provider directly
    if st.session_state.get("show_debug", False):
        st.markdown("---")
        st.subheader("🔍 Debug: Test LLM Provider")
        if st.button("Test Current Provider", key="test_provider"):
            try:
                current_provider = st.session_state.chatbot_llm_provider
                test_llm = get_llm_provider(current_provider)
                test_response = test_llm.generate("Say 'Hello, I am working!' in one sentence.", temperature=0.7)
                st.success(f"✅ Provider test successful!")
                st.code(test_response)
                if hasattr(test_llm, 'model'):
                    st.info(f"Using model: {test_llm.model}")
            except Exception as e:
                st.error(f"❌ Provider test failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Debug: List available Gemini models
        if st.session_state.chatbot_llm_provider == "gemini":
            if st.button("List Available Gemini Models", key="list_gemini_models"):
                try:
                    import google.generativeai as genai
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if api_key:
                        genai.configure(api_key=api_key)
                        models = list(genai.list_models())
                        st.write("**Available Models:**")
                        for m in models:
                            name = getattr(m, 'name', str(m))
                            methods = getattr(m, 'supported_generation_methods', [])
                            st.write(f"- {name} (methods: {methods})")
                    else:
                        st.error("GOOGLE_API_KEY not set")
                except Exception as e:
                    st.error(f"Failed to list models: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Toggle debug mode
    if st.checkbox("Show debug info", value=st.session_state.get("show_debug", False), key="toggle_debug"):
        st.session_state.show_debug = True
    else:
        st.session_state.show_debug = False
    
    # Input area at the bottom (fixed position)
    st.markdown("---")
    
    # Input form at bottom
    with st.form(key="chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            example_query = st.session_state.get("example_query", "")
            user_input = st.text_input(
                "💬 Ask me anything:",
                value=example_query,
                key="user_input_chat",
                placeholder="e.g., 'Go to energy spectra, change background to black, export as PDF'",
                label_visibility="collapsed"
            )
            if example_query:
                del st.session_state.example_query
        with col2:
            send_button = st.form_submit_button("Send", type="primary", width='stretch')
    
    # Process input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get context
        context = get_app_context()
        
        # Parse user input to actions
        with st.spinner("🤔 Thinking..."):
            try:
                actions = parser.parse(user_input, context)
            except Exception as e:
                error_msg = str(e)
                # Show more detailed error for debugging
                detailed_error = f"❌ Error parsing request: {error_msg}"
                if "Gemini" in error_msg or "GOOGLE_API_KEY" in error_msg:
                    detailed_error += "\n\n💡 **Gemini Setup Help:**\n"
                    detailed_error += "- Make sure GOOGLE_API_KEY is set in run_app.sh\n"
                    detailed_error += "- Restart the app after setting the key\n"
                    detailed_error += "- Check that google-generativeai is installed: `pip install google-generativeai`"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": detailed_error
                })
                st.rerun()
        
        # Execute actions
        results = executor.execute_batch(actions)
        
        # Generate response message
        response_parts = []
        pending_deletions = []
        pending_modifications = []
        pending_renames = []
        for i, (action, result) in enumerate(zip(actions, results)):
            if result.get("success"):
                # For conversational_response, show message as-is (already contains markdown)
                if action.get("action") == "conversational_response":
                    msg = result.get('message', '')
                else:
                    msg = f"✅ {result.get('message', 'Done')}"
                
                # Add deleted file info for better context in history
                if action.get("action") == "delete_file" and "data" in result and "deleted_filepath" in result["data"]:
                    msg += f"\n**Deleted:** `{result['data']['deleted_filepath']}`"
                
                # Add renamed file info
                if action.get("action") in ["rename_file", "move_file"] and "data" in result:
                    old_path = result["data"].get("old_filepath", "")
                    new_path = result["data"].get("new_filepath", "")
                    if old_path and new_path:
                        msg += f"\n**Renamed:** `{Path(old_path).name}` → `{Path(new_path).name}`"
                
                # Add code output if available
                if "data" in result and "output" in result["data"]:
                    output = result["data"]["output"]
                    if output:
                        msg += f"\n```\n{output}\n```"
                # Add file content if available - show full content for read_file action
                if "data" in result and "content" in result["data"]:
                    # Use full_content if available, otherwise use content
                    content = result["data"].get("full_content") or result["data"]["content"]
                    filepath = action.get("filepath", "")
                    
                    # LLM can specify language, otherwise infer from file extension intelligently
                    lang = action.get("language") or result["data"].get("language")
                    if not lang and filepath:
                        # Intelligent language detection from extension (common patterns)
                        ext = Path(filepath).suffix.lower()
                        # Only map obvious extensions - let LLM handle edge cases
                        if ext in [".py"]: lang = "python"
                        elif ext in [".f90", ".f", ".f95", ".f03", ".f08"]: lang = "fortran"
                        elif ext in [".js", ".jsx"]: lang = "javascript"
                        elif ext in [".ts", ".tsx"]: lang = "typescript"
                        elif ext == ".html": lang = "html"
                        elif ext == ".css": lang = "css"
                        elif ext == ".json": lang = "json"
                        elif ext == ".md": lang = "markdown"
                        elif ext in [".sh", ".bash"]: lang = "bash"
                        elif ext in [".c", ".h"]: lang = "c"
                        elif ext in [".cpp", ".hpp", ".cc", ".cxx"]: lang = "cpp"
                        else: lang = "text"  # Default - let markdown handle it
                    else:
                        lang = lang or "text"
                    
                    # Show full content for read_file action
                    if action.get("action") == "read_file":
                        line_count = result["data"].get("line_count", 0)
                        msg += f"\n\n**File content ({line_count} lines):**\n```{lang}\n{content}\n```"
                    else:
                        # For other actions, show preview
                        if len(content) < 500:
                            msg += f"\n\n**File content:**\n```{lang}\n{content}\n```"
                        else:
                            msg += f"\n\n**File preview (first 500 chars):**\n```{lang}\n{content[:500]}...\n```"
                # Add search results if available
                if "data" in result and "formatted" in result["data"]:
                    msg += f"\n\n{result['data']['formatted']}"
                # Add git output if available
                if "data" in result and "output" in result["data"] and "git" in action.get("action", "").lower():
                    git_output = result["data"]["output"]
                    if git_output:
                        msg += f"\n```\n{git_output}\n```"
                response_parts.append(msg)
            else:
                # Check if this is a file modification requiring confirmation
                if result.get("requires_confirmation") and result.get("action") == "modify_file":
                    pending_modifications.append({
                        "filepath": result.get("filepath"),
                        "filename": result.get("data", {}).get("filename", Path(result.get("filepath", "")).name),
                        "current_content": result.get("data", {}).get("current_content", ""),
                        "new_content": result.get("data", {}).get("new_content", ""),
                        "action": action
                    })
                    msg = result.get('message', 'Review required')
                # Check if this is a file rename requiring confirmation
                elif result.get("requires_confirmation") and result.get("action") in ["rename_file", "move_file"]:
                    pending_renames.append({
                        "filepath": result.get("filepath"),
                        "filename": result.get("data", {}).get("filename", Path(result.get("filepath", "")).name),
                        "new_filepath": result.get("data", {}).get("new_filepath", ""),
                        "new_filename": result.get("data", {}).get("new_filename", ""),
                        "action": action
                    })
                    msg = result.get('message', 'Review required')
                # Check if this is a deletion requiring confirmation
                elif result.get("requires_confirmation") and result.get("action") == "delete_file":
                    pending_deletions.append({
                        "filepath": result.get("filepath"),
                        "filename": result.get("data", {}).get("filename", Path(result.get("filepath", "")).name),
                        "action": action
                    })
                    msg = result.get('message', 'Confirmation required')
                else:
                    msg = f"❌ {result.get('message', 'Failed')}"
                    # Show command output even on failure so LLM can see what went wrong
                    if action.get("action") in ["run_shell_command", "execute_code"] and "data" in result and "output" in result["data"]:
                        output = result["data"]["output"]
                        if output:
                            msg += f"\n```\n{output}\n```"
                response_parts.append(msg)
        
        # Store pending modifications, renames, and deletions in session state
        if pending_modifications:
            st.session_state.pending_modifications = pending_modifications
        if pending_renames:
            st.session_state.pending_renames = pending_renames
        if pending_deletions:
            st.session_state.pending_deletions = pending_deletions
        
        response = "\n".join(response_parts) if response_parts else "No actions executed"
        
        # Add assistant response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "actions": actions,
            "results": results
        })
        
        # Handle navigation
        if st.session_state.get("_chatbot_navigate_to"):
            st.info(f"💡 Navigate to {st.session_state._chatbot_navigate_to} page using the sidebar menu")
            # Clear navigation flag after showing message
            del st.session_state._chatbot_navigate_to
        
        st.rerun()

if __name__ == "__main__":
    main()

