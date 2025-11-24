"""
Overview Page
Displays simulation metadata, parameters, and time series statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.csv_reader import read_csv_data, read_eps_validation_csv
from data_readers.parameter_reader import read_parameters, format_parameters_for_display
from utils.file_detector import detect_simulation_files
from visualizations.time_series import plot_time_series

def main():
    st.title("📊 Overview")
    
    if 'data_directory' not in st.session_state or not st.session_state.data_directory:
        st.warning("Please select a data directory from the main page.")
        return
    
    data_dir = st.session_state.data_directory
    
    # Detect available files
    files = detect_simulation_files(data_dir)
    
    # Load parameters
    if files['parameters']:
        params = read_parameters(str(files['parameters'][0]))
        formatted_params = format_parameters_for_display(params)
        
        st.header("Simulation Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Grid Parameters")
            for label, info in formatted_params.items():
                if 'Grid' in label or 'Size' in label or 'Interval' in label or 'Tag' in label:
                    st.text(f"{label}: {info['value']} {info['unit']}")
        
        with col2:
            st.subheader("Physical Parameters")
            for label, info in formatted_params.items():
                if 'Viscosity' in label or 'Velocity' in label or 'Relaxation' in label or 'Forcing' in label or 'Perturbation' in label:
                    st.text(f"{label}: {info['value']} {info['unit']}")
        
        with col3:
            st.subheader("LBM Parameters")
            for label, info in formatted_params.items():
                if 'Lattice' in label or 'Speed' in label or 'Length' in label or 'Smagorinsky' in label:
                    st.text(f"{label}: {info['value']} {info['unit']}")
    
    # Load CSV data
    if files['csv']:
        df = read_csv_data(str(files['csv'][0]))
        
        st.header("Turbulence Statistics")
        
        # Latest values table
        st.subheader("Latest Values")
        latest = df.iloc[-1]
        st.dataframe(latest.to_frame().T, use_container_width=True)
        
        # Full time series table
        st.subheader("Time Series Data")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Time series plots
        st.subheader("Time Evolution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'TKE' in df.columns:
                fig = plot_time_series(df['iter'].values, df['TKE'].values, "TKE", "blue", "TKE")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'u_rms' in df.columns:
                fig = plot_time_series(df['iter'].values, df['u_rms'].values, "u_rms", "green", "u_rms")
                st.plotly_chart(fig, use_container_width=True)
    
    # File availability checklist
    st.header("Data Availability")
    checklist = {
        'CSV Statistics': len(files['csv']) > 0,
        'Energy Spectra': len(files['spectrum']) > 0,
        'Normalized Spectra': len(files['norm_spectrum']) > 0,
        'Structure Functions': len(files['structure_functions_txt']) > 0 or len(files['structure_functions_bin']) > 0,
        'Flatness': len(files['flatness']) > 0,
        'Isotropy': len(files['isotropy']) > 0,
        'Energy Balance Validation': len(files['eps_validation']) > 0,
    }
    
    for item, available in checklist.items():
        status = "✅" if available else "❌"
        st.markdown(f"{status} {item}")

if __name__ == "__main__":
    main()

