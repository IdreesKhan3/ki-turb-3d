"""
Terminal Page
Command-line interface to control the entire app
"""

import streamlit as st
import sys
import io
import os
import subprocess
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from utils.theme_config import inject_theme_css, get_theme_list
st.set_page_config(page_icon="⚫")

def inject_terminal_css():
    """Inject professional terminal CSS styling"""
    st.markdown("""
    <style>
    /* Terminal Styling */
    .terminal-container {
        background: #0d1117;
        border: 2px solid #30363d;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        overflow: hidden;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    .terminal-header {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        padding: 12px 15px;
        border-bottom: 1px solid #30363d;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #c9d1d9;
    }
    
    .terminal-header .title {
        font-weight: 600;
        color: #58a6ff;
        font-size: 14px;
    }
    
    .terminal-header .status {
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 12px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3fb950;
        box-shadow: 0 0 4px #3fb950;
    }
    
    .terminal-output {
        background: #0d1117;
        color: #c9d1d9;
        padding: 15px;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .terminal-prompt {
        color: #58a6ff;
        font-weight: 600;
    }
    
    .terminal-command {
        color: #c9d1d9;
    }
    
    .terminal-output-text {
        color: #c9d1d9;
        margin: 5px 0;
    }
    
    .terminal-error {
        color: #f85149;
    }
    
    .terminal-success {
        color: #3fb950;
    }
    
    .terminal-warning {
        color: #d29922;
    }
    
    .terminal-info {
        color: #58a6ff;
    }
    
    /* Scrollbar */
    .terminal-output::-webkit-scrollbar {
        width: 10px;
    }
    
    .terminal-output::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    .terminal-output::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 5px;
    }
    
    .terminal-output::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    
    /* Input styling */
    div[data-baseweb="input"] {
        background: #0d1117 !important;
    }
    
    div[data-baseweb="input"] input {
        background: #0d1117 !important;
        color: #c9d1d9 !important;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 13px !important;
        border: 1px solid #30363d !important;
        border-radius: 4px !important;
        padding: 10px !important;
        caret-color: #58a6ff !important;
    }
    
    div[data-baseweb="input"] input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }
    
    /* Command history */
    .command-history-item {
        padding: 4px 0;
        border-bottom: 1px solid #21262d;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .command-history-item:hover {
        background: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

def get_prompt():
    """Generate terminal prompt"""
    project_root = Path(__file__).parent.parent
    try:
        cwd = Path.cwd()
        if cwd.is_relative_to(project_root):
            cwd_display = str(cwd.relative_to(project_root))
        else:
            cwd_display = str(cwd)
    except:
        cwd_display = "APP"
    
    user = os.getenv('USER', os.getenv('USERNAME', 'user'))
    return f"{user}@turbulence-dashboard:{cwd_display}$"

def execute_app_command(command: str, args: list) -> str:
    """Execute app-specific commands"""
    cmd = command.lower()
    
    if cmd == "help" or cmd == "?":
        return """Available Commands:
  help, ?              - Show this help message
  clear, cls           - Clear terminal output
  load <path>          - Load data directory
  load-multi <path...> - Load multiple directories
  theme <name>         - Change theme
  themes               - List available themes
  state                - Show session state
  state <key>          - Show specific state value
  set <key> <value>    - Set session state value
  pages                - List available pages
  goto <page>          - Navigate to page (requires manual navigation)
  pwd                  - Print working directory
  ls, dir              - List files in current directory
  cd <path>            - Change directory
  python <code>        - Execute Python code
  exit, quit           - Exit terminal (refresh page)
"""
    
    elif cmd == "clear" or cmd == "cls":
        if 'terminal_history' in st.session_state:
            st.session_state.terminal_history = []
        return "Terminal cleared."
    
    elif cmd == "load":
        if not args:
            return "Error: Please provide a directory path.\nUsage: load <path>"
        path = Path(args[0])
        if path.exists() and path.is_dir():
            st.session_state.data_directory = str(path)
            st.session_state.data_directories = [str(path)]
            st.session_state.data_loaded = True
            return f"✅ Loaded data directory: {path}"
        else:
            return f"❌ Error: Directory not found: {path}"
    
    elif cmd == "load-multi":
        if not args:
            return "Error: Please provide directory paths.\nUsage: load-multi <path1> <path2> ..."
        valid_paths = []
        for path_str in args:
            path = Path(path_str)
            if path.exists() and path.is_dir():
                valid_paths.append(str(path))
            else:
                return f"❌ Error: Directory not found: {path}"
        st.session_state.data_directories = valid_paths
        st.session_state.data_loaded = True
        return f"✅ Loaded {len(valid_paths)} directories:\n" + "\n".join(f"  - {p}" for p in valid_paths)
    
    elif cmd == "theme":
        if not args:
            return f"Current theme: {st.session_state.get('theme', 'Light Scientific')}\nUsage: theme <name>"
        theme_name = " ".join(args)
        available_themes = get_theme_list()
        if theme_name in available_themes:
            st.session_state.theme = theme_name
            if 'plot_style' in st.session_state:
                del st.session_state.plot_style
            return f"✅ Theme changed to: {theme_name}"
        else:
            return f"❌ Error: Theme '{theme_name}' not found.\nAvailable themes: {', '.join(available_themes)}"
    
    elif cmd == "themes":
        themes = get_theme_list()
        return "Available themes:\n" + "\n".join(f"  - {t}" for t in themes)
    
    elif cmd == "state":
        if not args:
            # Show all session state
            state_items = []
            for key, value in st.session_state.items():
                if key.startswith('_'):
                    continue  # Skip internal keys
                if isinstance(value, (str, int, float, bool, type(None))):
                    state_items.append(f"  {key} = {value}")
                else:
                    state_items.append(f"  {key} = {type(value).__name__}")
            return "Session State:\n" + "\n".join(state_items) if state_items else "Session state is empty."
        else:
            key = args[0]
            if key in st.session_state:
                value = st.session_state[key]
                return f"{key} = {value}"
            else:
                return f"❌ Error: Key '{key}' not found in session state."
    
    elif cmd == "set":
        if len(args) < 2:
            return "Error: Please provide key and value.\nUsage: set <key> <value>"
        key = args[0]
        value = " ".join(args[1:])
        # Try to convert to appropriate type
        try:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except:
                    pass  # Keep as string
        except:
            pass
        st.session_state[key] = value
        return f"✅ Set {key} = {value}"
    
    elif cmd == "pages":
        pages = [
            "01_Overview", "02_Theory_Equations", "03_Multi_Method_Support",
            "04_Other_Turbulence_Stats", "05_Real_Isotropy", "06_Spectral_Isotropy",
            "07_Energy_Spectra", "08_Flatness", "09_Structure_Functions",
            "10_LES_Metrics", "11_Comparison", "12_Reynolds_Transition",
            "13_3D_Slice_Viewer", "14_Citation",
            "16_Report_Builder", "17_Terminal"
        ]
        return "Available pages:\n" + "\n".join(f"  - {p}" for p in pages)
    
    elif cmd == "goto":
        if not args:
            return "Error: Please provide a page name.\nUsage: goto <page>"
        page = args[0]
        st.session_state['_navigate_to'] = page
        return f"💡 Navigate to '{page}' page using the sidebar menu."
    
    elif cmd == "pwd":
        return str(Path.cwd())
    
    elif cmd == "ls" or cmd == "dir":
        try:
            cwd = Path.cwd()
            items = []
            for item in sorted(cwd.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    items.append(f"📄 {item.name}")
            return "\n".join(items) if items else "Directory is empty."
        except Exception as e:
            return f"❌ Error: {e}"
    
    elif cmd == "cd":
        if not args:
            return str(Path.cwd())
        try:
            new_path = Path(args[0])
            if new_path.is_absolute():
                os.chdir(new_path)
            else:
                os.chdir(Path.cwd() / new_path)
            return f"Changed directory to: {Path.cwd()}"
        except Exception as e:
            return f"❌ Error: {e}"
    
    elif cmd == "python" or cmd == "py":
        if not args:
            return "Error: Please provide Python code.\nUsage: python <code>"
        code = " ".join(args)
        return execute_python_code(code)
    
    elif cmd == "exit" or cmd == "quit":
        return "💡 Use browser refresh to exit terminal."
    
    else:
        return f"❌ Unknown command: {command}\nType 'help' for available commands."

def execute_python_code(code: str) -> str:
    """Execute Python code and return output"""
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = output_buffer
    sys.stderr = error_buffer
    
    try:
        exec_globals = {
            '__builtins__': __builtins__,
            'st': st,
            'np': np,
            'pd': pd,
            'go': go,
            'px': px,
            'Path': Path,
            'print': print,
        }
        
        # Add session state variables
        for key in st.session_state:
            if not key.startswith('_'):
                exec_globals[key] = st.session_state[key]
        
        result = eval(code, exec_globals) if '\n' not in code.strip() else None
        if result is None:
            exec(code, exec_globals)
        
        stdout_text = output_buffer.getvalue()
        stderr_text = error_buffer.getvalue()
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        output = ""
        if stdout_text:
            output += stdout_text
        if stderr_text:
            output += f"⚠️ {stderr_text}"
        if result is not None:
            output += f"\nResult: {result}"
        
        return output if output else "✅ Code executed successfully."
    
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

def main():
    inject_theme_css()
    inject_terminal_css()
    
    st.title("💻 Terminal")
    st.markdown("**Command-line interface to control the entire app**")
    
    # Initialize terminal history
    if 'terminal_history' not in st.session_state:
        st.session_state.terminal_history = []
    
    if 'terminal_command_history' not in st.session_state:
        st.session_state.terminal_command_history = []
    
    # Sidebar
    with st.sidebar:
        st.subheader("📚 Quick Commands")
        st.code("""
load <path>
theme <name>
state
pages
help
        """)
        
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown("""
- Use `help` to see all commands
- `load <path>` to load data
- `theme <name>` to change theme
- `state` to view session state
- `python <code>` to run Python
        """)
        
        if st.button("🗑️ Clear Terminal"):
            st.session_state.terminal_history = []
            st.session_state.terminal_command_history = []
            st.rerun()
    
    # Terminal interface
    st.markdown("""
    <div class="terminal-container">
        <div class="terminal-header">
            <span class="title">💻 Terminal</span>
            <div class="status">
                <span class="status-dot"></span>
                <span>Ready</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display terminal history
    if st.session_state.terminal_history:
        terminal_output = "\n".join(st.session_state.terminal_history)
        st.markdown(f'<div class="terminal-output">{terminal_output}</div>', unsafe_allow_html=True)
    else:
        welcome_msg = f"""Welcome to Turbulence Dashboard Terminal
Type 'help' for available commands.

{get_prompt()}"""
        st.markdown(f'<div class="terminal-output">{welcome_msg}</div>', unsafe_allow_html=True)
    
    # Command input
    col1, col2 = st.columns([10, 1])
    
    with col1:
        command_input = st.text_input(
            "",
            key="terminal_command",
            placeholder="Enter command...",
            label_visibility="collapsed"
        )
    
    with col2:
        execute_btn = st.button("▶️", width='stretch')
    
    # Execute command
    if execute_btn and command_input.strip():
        prompt = get_prompt()
        command_line = command_input.strip()
        
        # Add command to history
        st.session_state.terminal_command_history.append(command_line)
        if len(st.session_state.terminal_command_history) > 100:
            st.session_state.terminal_command_history.pop(0)
        
        # Parse command
        parts = command_line.split()
        if not parts:
            st.rerun()
        
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        # Execute command
        if command.startswith("python ") or command.startswith("py "):
            code = command_line.replace("python ", "").replace("py ", "")
            result = execute_python_code(code)
        else:
            result = execute_app_command(command, args)
        
        # Add to terminal history
        new_entry = f'<span class="terminal-prompt">{prompt}</span> <span class="terminal-command">{command_line}</span>\n<span class="terminal-output-text">{result}</span>'
        st.session_state.terminal_history.append(new_entry)
        
        # Keep history manageable
        if len(st.session_state.terminal_history) > 50:
            st.session_state.terminal_history.pop(0)
        
        st.rerun()
    
    # Command history navigation
    if st.session_state.terminal_command_history:
        st.markdown("---")
        with st.expander("📜 Command History", expanded=False):
            for idx, cmd in enumerate(reversed(st.session_state.terminal_command_history[-20:]), 1):
                if st.button(f"{cmd}", key=f"hist_{idx}", width='stretch'):
                    st.session_state.terminal_command = cmd
                    st.rerun()

if __name__ == "__main__":
    main()

