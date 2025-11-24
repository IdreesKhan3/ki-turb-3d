"""
Turbulence Statistics Dashboard
Main Streamlit application entry point
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="LBM-MRT Turbulence Analysis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_directory' not in st.session_state:
    st.session_state.data_directory = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    """Main application entry point"""
    
    # Sidebar - Data Selection
    with st.sidebar:
        st.title("🌊 Turbulence Dashboard")
        st.markdown("---")
        
        st.subheader("📁 Data Selection")
        
        # Primary: User data directory selection
        st.markdown("**Select your simulation output folder:**")
        user_dir = st.text_input(
            "Directory path:",
            value="",
            help="Enter the path to your simulation output directory"
        )
        
        if st.button("Load Data", type="primary"):
            if user_dir and Path(user_dir).exists():
                st.session_state.data_directory = user_dir
                st.session_state.data_loaded = True
                st.success(f"Data loaded from: {user_dir}")
            else:
                st.error("Directory not found. Please check the path.")
        
        # Optional: Example data
        st.markdown("---")
        st.markdown("**Or try example data:**")
        if st.button("Try Example Data"):
            example_dir = project_root / "examples" / "showcase"
            if example_dir.exists():
                st.session_state.data_directory = str(example_dir)
                st.session_state.data_loaded = True
                st.success("Example data loaded")
            else:
                st.warning("Example data not available")
        
        # Navigation
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        st.markdown("""
        - [Overview](#overview)
        - [Energy Spectra](#energy-spectra)
        - [Structure Functions](#structure-functions)
        - [Flatness](#flatness)
        - [LES Metrics](#les-metrics)
        - [Isotropy](#isotropy)
        - [Comparison](#comparison)
        - [Reynolds Transition](#reynolds-transition)
        - [Theory & Equations](#theory-equations)
        - [Multi-Method Support](#multi-method-support)
        """)
    
    # Main content area
    st.title("🌊 LBM-MRT Turbulence Analysis Dashboard")
    
    if not st.session_state.data_loaded:
        st.info("👈 Please select a data directory from the sidebar to begin analysis.")
        st.markdown("""
        ### Welcome!
        
        This dashboard allows you to analyze LBM-based DNS/LES turbulence simulation data.
        **Primary focus:** MRT (Multiple Relaxation Time) | **Also supports:** SRT (BGK)
        
        **Features:**
        - Interactive energy spectra visualization
        - Structure functions and ESS analysis
        - Flatness factors
        - Isotropy validation (spectral and real-space)
        - LES metrics
        - Energy balance analysis
        - Multi-simulation comparison
        - Multi-method support (MRT and SRT)
        
        **Getting Started:**
        1. Select your simulation output folder from the sidebar
        2. Or try the example data to see the dashboard in action
        3. Navigate through the pages using the sidebar
        """)
    else:
        st.success(f"✅ Data loaded from: `{st.session_state.data_directory}`")
        st.info("Navigate through the pages using the sidebar menu.")

if __name__ == "__main__":
    main()

