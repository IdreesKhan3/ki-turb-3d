"""
Python Interpreter Page
Interactive Python console for data analysis and testing
Professional code editor interface for physicists
"""

import streamlit as st
import sys
import io
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.theme_config import inject_theme_css

def inject_code_editor_css():
    """Inject professional code editor CSS styling"""
    st.markdown("""
    <style>
    /* Professional Code Editor Styling */
    .code-editor-container {
        border: 2px solid #2d2d2d;
        border-radius: 8px;
        background: #1e1e1e;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        overflow: hidden;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    .code-editor-header {
        background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
        padding: 10px 15px;
        border-bottom: 1px solid #3d3d3d;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #d4d4d4;
        font-size: 13px;
    }
    
    .code-editor-header .title {
        font-weight: 600;
        color: #4ec9b0;
    }
    
    .code-editor-header .status {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ec9b0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Style the text area to look like a code editor */
    div[data-baseweb="textarea"] {
        background: #1e1e1e !important;
        border: none !important;
    }
    
    div[data-baseweb="textarea"] textarea {
        background: #1e1e1e !important;
        color: #d4d4d4 !important;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        padding: 15px !important;
        border: none !important;
        caret-color: #4ec9b0 !important;
    }
    
    div[data-baseweb="textarea"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Line number styling (simulated with CSS) */
    .code-line-numbers {
        position: absolute;
        left: 0;
        top: 0;
        padding: 15px 10px;
        color: #6a6a6a;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        user-select: none;
        pointer-events: none;
        background: #252526;
        border-right: 1px solid #3d3d3d;
    }
    
    /* Syntax highlighting hints (basic) */
    .code-hint {
        color: #569cd6; /* Blue for keywords */
    }
    
    .code-string {
        color: #ce9178; /* Orange for strings */
    }
    
    .code-comment {
        color: #6a9955; /* Green for comments */
    }
    
    /* Editor controls */
    .editor-controls {
        display: flex;
        gap: 10px;
        padding: 10px 15px;
        background: #252526;
        border-top: 1px solid #3d3d3d;
        align-items: center;
    }
    
    .editor-controls button {
        background: #0e639c;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        transition: background 0.2s;
    }
    
    .editor-controls button:hover {
        background: #1177bb;
    }
    
    /* Output styling */
    .code-output {
        background: #1e1e1e;
        border: 2px solid #2d2d2d;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        color: #d4d4d4;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .code-output pre {
        margin: 0;
        color: #d4d4d4;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    /* Scrollbar styling */
    .code-output::-webkit-scrollbar {
        width: 10px;
    }
    
    .code-output::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    .code-output::-webkit-scrollbar-thumb {
        background: #424242;
        border-radius: 5px;
    }
    
    .code-output::-webkit-scrollbar-thumb:hover {
        background: #4e4e4e;
    }
    
    /* Success/Error indicators */
    .exec-status {
        padding: 8px 12px;
        border-radius: 4px;
        margin: 10px 0;
        font-weight: 500;
    }
    
    .exec-success {
        background: rgba(78, 201, 176, 0.2);
        color: #4ec9b0;
        border-left: 3px solid #4ec9b0;
    }
    
    .exec-error {
        background: rgba(244, 67, 54, 0.2);
        color: #f44336;
        border-left: 3px solid #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    inject_theme_css()
    inject_code_editor_css()
    
    st.title("🐍 Python Interpreter")
    st.markdown("**Professional code editor for scientific computing and data analysis**")
    
    # Initialize session state for interpreter
    if 'interpreter_history' not in st.session_state:
        st.session_state.interpreter_history = []
    if 'interpreter_output' not in st.session_state:
        st.session_state.interpreter_output = ""
    
    # Sidebar with info and controls
    with st.sidebar:
        st.subheader("📚 Available Imports")
        st.code("""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
""")
        
        st.markdown("---")
        st.subheader("🔧 Session State")
        st.markdown("Access session state variables:")
        st.code("""
st.session_state.data_directory
st.session_state.theme
st.session_state.plot_style
# ... and more
""")
        
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown("""
- Use `print()` to display output
- Access loaded data via session state
- Plotly figures will display automatically
- Clear output to start fresh
        """)
        
        if st.button("🗑️ Clear History"):
            st.session_state.interpreter_history = []
            st.session_state.interpreter_output = ""
            st.rerun()
    
    # Main code input area with professional editor styling
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Editor header
        st.markdown("""
        <div class="code-editor-container">
            <div class="code-editor-header">
                <span class="title">📝 Python Editor</span>
                <div class="status">
                    <span class="status-dot"></span>
                    <span>Ready</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle example code injection
    default_code = """# Example: Access data directory
data_dir_path = st.session_state.get('data_directory')
if data_dir_path:
    data_dir = Path(data_dir_path)
    print(f"Data directory: {data_dir}")
else:
    print("No data directory loaded. Please load data from the Overview page.")

# Example: Create a simple plot
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
fig.update_layout(title='Simple Plot', xaxis_title='x', yaxis_title='y')
fig
"""
    
    # Initialize text area code - use injected code if available, otherwise default
    if 'interpreter_code' in st.session_state:
        # Code was injected by example button - set it directly in the text area key
        if 'interpreter_code_input' not in st.session_state:
            st.session_state.interpreter_code_input = ""
        st.session_state.interpreter_code_input = st.session_state.interpreter_code
        del st.session_state.interpreter_code
    elif 'interpreter_code_input' not in st.session_state:
        # First time - use default code
        st.session_state.interpreter_code_input = default_code
    
    # Code input with enhanced styling
    code_input = st.text_area(
        "",
        value=st.session_state.interpreter_code_input,
        height=400,
        key="interpreter_code_input",
        help="Write Python code here. Use 'print()' for text output, and Plotly figures will display automatically.",
        label_visibility="collapsed"
    )
    
    # Editor footer with controls
    st.markdown("""
    <div class="editor-controls">
        <span style="color: #6a6a6a; font-size: 12px;">💡 Press Ctrl+Enter to run | Use print() for output</span>
    </div>
    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        st.subheader("⚡ Execute")
        execute_button = st.button("▶️ Run Code", type="primary", use_container_width=True)
        st.markdown("---")
        
        # Quick stats
        if code_input:
            lines = len(code_input.split('\n'))
            chars = len(code_input)
            st.caption(f"📊 {lines} lines")
            st.caption(f"📝 {chars} characters")
        
        st.markdown("---")
        st.caption("**Shortcuts:**")
        st.caption("Ctrl+Enter: Run")
        st.caption("Ctrl+/: Comment")
    
    # Execute code when button is clicked
    if execute_button:
        if code_input.strip():
            # Capture stdout and stderr
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            # Redirect stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = output_buffer
            sys.stderr = error_buffer
            
            try:
                # Prepare execution environment
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
                
                # Add session state variables to globals for easier access
                if 'data_directory' in st.session_state:
                    exec_globals['data_directory'] = st.session_state.data_directory
                if 'theme' in st.session_state:
                    exec_globals['theme'] = st.session_state.theme
                
                # Execute the code and capture return value
                result = None
                try:
                    # Try to evaluate as expression first (for single-line returns)
                    if '\n' not in code_input.strip() and not code_input.strip().startswith('#'):
                        try:
                            result = eval(code_input, exec_globals)
                        except:
                            exec(code_input, exec_globals)
                    else:
                        exec(code_input, exec_globals)
                except Exception as exec_error:
                    raise exec_error
                
                # Get output
                stdout_text = output_buffer.getvalue()
                stderr_text = error_buffer.getvalue()
                
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # Display output with professional styling
                if stdout_text or result is not None:
                    st.markdown('<div class="exec-status exec-success">✅ Execution successful</div>', unsafe_allow_html=True)
                    if stdout_text:
                        st.markdown('<div class="code-output"><pre>' + stdout_text.replace('\n', '<br>') + '</pre></div>', unsafe_allow_html=True)
                    
                    # Display result if it's a Plotly figure
                    if result is not None:
                        if hasattr(result, '__class__'):
                            class_name = result.__class__.__name__
                            if 'Figure' in class_name or 'graph_objs' in str(type(result)):
                                st.plotly_chart(result, use_container_width=True)
                            else:
                                st.markdown('<div class="code-output"><pre>' + str(result).replace('\n', '<br>') + '</pre></div>', unsafe_allow_html=True)
                
                if stderr_text:
                    st.markdown('<div class="exec-status exec-error">⚠️ Warnings/Errors</div>', unsafe_allow_html=True)
                    st.markdown('<div class="code-output"><pre style="color: #f44336;">' + stderr_text.replace('\n', '<br>') + '</pre></div>', unsafe_allow_html=True)
                
                # Add to history
                st.session_state.interpreter_history.append({
                    'code': code_input,
                    'output': stdout_text,
                    'error': stderr_text,
                    'success': not stderr_text
                })
                
            except Exception as e:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # Get error details
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                
                st.markdown(f'<div class="exec-status exec-error">❌ Execution error: {error_msg}</div>', unsafe_allow_html=True)
                st.markdown('<div class="code-output"><pre style="color: #f44336;">' + error_traceback.replace('\n', '<br>') + '</pre></div>', unsafe_allow_html=True)
                
                # Add to history
                st.session_state.interpreter_history.append({
                    'code': code_input,
                    'output': output_buffer.getvalue(),
                    'error': error_traceback,
                    'success': False
                })
    
    # Display history
    if st.session_state.interpreter_history:
        st.markdown("---")
        st.subheader("📜 Execution History")
        
        # Show last N entries (most recent first)
        history_to_show = st.session_state.interpreter_history[-10:][::-1]
        
        for idx, entry in enumerate(history_to_show):
            status_icon = "✅" if entry['success'] else "❌"
            status_class = "exec-success" if entry['success'] else "exec-error"
            with st.expander(
                f"{status_icon} Code #{len(st.session_state.interpreter_history) - idx}",
                expanded=False
            ):
                st.code(entry['code'], language='python')
                if entry['output']:
                    st.markdown('<div class="code-output"><pre>' + entry['output'].replace('\n', '<br>') + '</pre></div>', unsafe_allow_html=True)
                if entry['error']:
                    st.markdown('<div class="code-output"><pre style="color: #f44336;">' + entry['error'].replace('\n', '<br>') + '</pre></div>', unsafe_allow_html=True)
    
    # Quick examples
    st.markdown("---")
    st.subheader("📝 Quick Examples")
    
    example_cols = st.columns(3)
    
    with example_cols[0]:
        if st.button("📊 Simple Plot", use_container_width=True, key="ex_plot"):
            example_code = """import numpy as np
import plotly.graph_objects as go
st.set_page_config(page_icon="⚫")

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
fig.update_layout(title='Sine Wave', xaxis_title='x', yaxis_title='y')
fig"""
            st.session_state.interpreter_code = example_code
            st.rerun()
    
    with example_cols[1]:
        st.markdown("**📁 Check Data Dir**")
        st.caption("Loads code to check what files are in your loaded data directory")
        if st.button("Load Example", use_container_width=True, key="ex_data"):
            example_code = """from pathlib import Path

# Check what data directory is loaded
data_dir_path = st.session_state.get('data_directory')
if data_dir_path:
    data_dir = Path(data_dir_path)
    print(f"Data directory: {data_dir}")
    print(f"Exists: {data_dir.exists()}")
    
    if data_dir.exists():
        # List .dat files
        dat_files = list(data_dir.glob('*.dat'))[:10]
        print(f"\\nFound {len(dat_files)} .dat files (showing first 10):")
        for f in dat_files:
            print(f"  - {f.name}")
        
        # List .txt files
        txt_files = list(data_dir.glob('*.txt'))[:10]
        print(f"\\nFound {len(txt_files)} .txt files (showing first 10):")
        for f in txt_files:
            print(f"  - {f.name}")
        
        # List .csv files
        csv_files = list(data_dir.glob('*.csv'))[:10]
        print(f"\\nFound {len(csv_files)} .csv files (showing first 10):")
        for f in csv_files:
            print(f"  - {f.name}")
else:
    print("No data directory loaded. Please load data from the Overview page.")"""
            st.session_state.interpreter_code = example_code
            st.rerun()
    
    with example_cols[2]:
        if st.button("🔢 NumPy Example", use_container_width=True, key="ex_numpy"):
            example_code = """import numpy as np

arr = np.random.randn(1000)
print(f"Array shape: {arr.shape}")
print(f"Mean: {arr.mean():.4f}")
print(f"Std: {arr.std():.4f}")
print(f"Min: {arr.min():.4f}")
print(f"Max: {arr.max():.4f}")"""
            st.session_state.interpreter_code = example_code
            st.rerun()

if __name__ == "__main__":
    main()

