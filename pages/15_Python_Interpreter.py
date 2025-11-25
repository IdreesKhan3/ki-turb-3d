"""
Python Interpreter Page
Interactive Python console for data analysis and testing
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

def main():
    inject_theme_css()
    
    st.title("🐍 Python Interpreter")
    st.markdown("Interactive Python console for data analysis and testing")
    
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
    
    # Main code input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Code Input")
    
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
    
    code_input = st.text_area(
        "Enter Python code:",
        value=st.session_state.interpreter_code_input,
        height=300,
        key="interpreter_code_input",
        help="Write Python code here. Use 'print()' for text output, and Plotly figures will display automatically."
    )
    
    with col2:
        st.subheader("Execute")
        execute_button = st.button("▶️ Run Code", type="primary", use_container_width=True)
        st.markdown("---")
        st.caption("Press Ctrl+Enter to run")
    
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
                
                # Display output
                if stdout_text or result is not None:
                    st.success("✅ Execution successful")
                    if stdout_text:
                        with st.expander("📤 Output", expanded=True):
                            st.text(stdout_text)
                    
                    # Display result if it's a Plotly figure
                    if result is not None:
                        if hasattr(result, '__class__'):
                            class_name = result.__class__.__name__
                            if 'Figure' in class_name or 'graph_objs' in str(type(result)):
                                st.plotly_chart(result, use_container_width=True)
                            else:
                                with st.expander("📤 Result", expanded=True):
                                    st.write(result)
                
                if stderr_text:
                    st.warning("⚠️ Warnings/Errors")
                    with st.expander("⚠️ Error Output", expanded=True):
                        st.text(stderr_text)
                
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
                
                st.error(f"❌ Execution error: {error_msg}")
                with st.expander("🔍 Error Details", expanded=True):
                    st.code(error_traceback, language='python')
                
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
            with st.expander(
                f"{'✅' if entry['success'] else '❌'} Code #{len(st.session_state.interpreter_history) - idx}",
                expanded=False
            ):
                st.code(entry['code'], language='python')
                if entry['output']:
                    st.text("Output:")
                    st.text(entry['output'])
                if entry['error']:
                    st.text("Error:")
                    st.code(entry['error'], language='python')
    
    # Quick examples
    st.markdown("---")
    st.subheader("📝 Quick Examples")
    
    example_cols = st.columns(3)
    
    with example_cols[0]:
        if st.button("📊 Simple Plot", use_container_width=True, key="ex_plot"):
            example_code = """import numpy as np
import plotly.graph_objects as go

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

