"""
Energy Spectra Page
Time-averaged spectra with standard deviation, Pope model validation, time evolution
"""

import streamlit as st
import numpy as np
import glob
import re
from pathlib import Path
from collections import defaultdict
import sys
import plotly.graph_objects as go

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.spectrum_reader import read_spectrum_file
from data_readers.norm_spectrum_reader import read_norm_spectrum_file
from utils.file_detector import natural_sort_key, group_files_by_simulation
from utils.data_processor import compute_energy_variance
from visualizations.spectra import plot_energy_spectrum, plot_time_evolution_spectrum

def main():
    st.title("📈 Energy Spectra")
    
    if 'data_directory' not in st.session_state or not st.session_state.data_directory:
        st.warning("Please select a data directory from the main page.")
        return
    
    data_dir = Path(st.session_state.data_directory)
    
    # File selection
    st.sidebar.header("Options")
    view_mode = st.sidebar.radio("View Mode", ["Time-Averaged", "Time Evolution"])
    
    # Find spectrum files
    spectrum_files = sorted(glob.glob(str(data_dir / "spectrum*.dat")), key=natural_sort_key)
    norm_files = sorted(glob.glob(str(data_dir / "norm*.dat")), key=natural_sort_key)
    
    if not spectrum_files and not norm_files:
        st.error("No spectrum files found in the selected directory.")
        return
    
    if view_mode == "Time-Averaged":
        # Time-averaging mode (replicates plot_spectra.ipynb)
        st.header("Time-Averaged Energy Spectra")
        
        # Group files by simulation type
        sim_groups = group_files_by_simulation(spectrum_files, r'(spectrum\d+)_\d+\.dat')
        
        # Iteration range selection
        if sim_groups:
            sample_group = list(sim_groups.values())[0]
            total_files = len(sample_group)
            start_iter = st.sidebar.slider("Start Iteration", 1, total_files, 20)
            end_iter = st.sidebar.slider("End Iteration", start_iter, total_files, min(2000000, total_files))
        
        # Process each simulation group
        fig = None
        colors = ['blue', 'green', 'purple', 'orange', 'brown', 'gray']
        
        for idx, (sim_prefix, files) in enumerate(sorted(sim_groups.items())):
            selected_files = files[start_iter-1:min(end_iter, len(files))]
            
            if not selected_files:
                continue
            
            # Time-averaging computation (same as notebook)
            energy_accum = None
            energy_sq_accum = None
            count = 0
            k_vals = None
            
            for fname in selected_files:
                try:
                    k, E = read_spectrum_file(str(fname))
                    
                    if k_vals is None:
                        k_vals = k
                        energy_accum = np.zeros_like(E)
                        energy_sq_accum = np.zeros_like(E)
                    
                    energy_accum += E
                    energy_sq_accum += E**2
                    count += 1
                except Exception as e:
                    st.warning(f"Error reading {fname}: {e}")
                    continue
            
            if count == 0:
                continue
            
            # Compute mean and std (same as notebook)
            energy_avg = energy_accum / count
            energy_var = (energy_sq_accum / count) - energy_avg**2
            energy_std = np.sqrt(np.maximum(energy_var, 0.0))
            
            # Plot
            if fig is None:
                fig = plot_energy_spectrum(k_vals, energy_avg, energy_std, sim_prefix, colors[idx % len(colors)])
            else:
                fig.add_trace(go.Scatter(
                    x=k_vals,
                    y=energy_avg,
                    mode='lines',
                    name=sim_prefix,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Time evolution mode
        st.header("Time Evolution of Energy Spectra")
        st.info("Time evolution mode - showing all iterations")
        # Implementation for time evolution plot
        st.info("Time evolution visualization coming soon...")

if __name__ == "__main__":
    main()

