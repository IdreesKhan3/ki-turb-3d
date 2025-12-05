"""
Multi-Method Support Page
Demonstrates that the app works with both MRT and SRT data
Primary focus: MRT | Also supports: SRT (BGK)
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
st.set_page_config(page_icon="⚫")

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    st.title("🔄 Multi-Method Support")
    
    st.markdown("""
    ### Purpose
    
    This page demonstrates that the dashboard is **not limited to MRT simulations**.
    While the primary focus is on **MRT-based DNS/LES**, the app can also process
    and analyze data from **SRT (BGK) simulations**.
    
    **Note:** This is not a comparison of method performance, but rather a demonstration
    of the app's flexibility to work with different LBM collision operators.
    """)
    
    if 'data_directory' not in st.session_state or not st.session_state.data_directory:
        st.warning("Please select a data directory from the main page.")
        st.info("""
        **To see multi-method examples:**
        - Load MRT simulation data (primary use case)
        - Optionally load SRT simulation data to see side-by-side plots
        - The app will automatically detect and display both
        """)
        return
    
    st.info("""
    **Implementation Status:** This page will show energy spectra from both MRT and SRT 
    simulations side-by-side when both datasets are available. The implementation will 
    focus on demonstrating the app's capability to handle multiple LBM methods, not on 
    comparing their relative performance.
    """)

if __name__ == "__main__":
    main()

