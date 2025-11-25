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

from data_readers.csv_reader import read_eps_validation_csv
from data_readers.parameter_reader import read_parameters, format_parameters_for_display
from data_readers.binary_reader import read_tau_analysis_file
from utils.file_detector import detect_simulation_files
from utils.theme_config import inject_theme_css

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
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
    
    # Compute Mach and Knudsen numbers (physics validation)
    mach_number = None
    knudsen_number = None
    mach_warning = False
    
    if files['eps_validation']:
        # Load validation data for u_rms_real
        val_df = read_eps_validation_csv(str(files['eps_validation'][0]))
        if 'u_rms_real' in val_df.columns and len(val_df) > 0:
            u_rms_latest = val_df['u_rms_real'].iloc[-1]
            c_s = 1.0 / np.sqrt(3.0)  # Lattice sound speed
            mach_number = u_rms_latest / c_s
            mach_warning = mach_number > 0.1  # Warning if Ma > 0.1
    
    # Determine if simulation is DNS or LES
    is_les = False
    if files['parameters']:
        params = read_parameters(str(files['parameters'][0]))
        smogc = params.get('SmogC', 0.0)
        is_les = smogc > 0.0  # LES if Smagorinsky constant is set
    
    # Compute Knudsen number (different for DNS vs LES)
    if files['parameters']:
        params = read_parameters(str(files['parameters'][0]))
        nu = params.get('nu', None)
        if nu is not None:
            c_s2 = 1.0 / 3.0
            
            if is_les:
                # LES: Kn_t = ((τ_e - 1/2) * √3 * Δx) / Δx = (τ_e - 1/2) * √3
                # Read τ_e from tau_analysis_*.bin files
                # tau_local = 1.0 / s9_field = τ_e, tau_offset = τ_e - 0.5, so τ_e = tau_offset + 0.5
                if files['tau_analysis'] and files['parameters']:
                    params = read_parameters(str(files['parameters'][0]))
                    nx = params.get('nx', None)
                    ny = params.get('ny', None)
                    nz = params.get('nz', None)
                    
                    if nx and ny and nz:
                        tau_file = str(files['tau_analysis'][-1])
                        try:
                            tau_e = read_tau_analysis_file(tau_file, nx, ny, nz)
                            dx = 1.0
                            sqrt3 = np.sqrt(3.0)
                            knudsen_number = ((tau_e - 0.5) * sqrt3 * dx) / dx
                            knudsen_type = "LES"
                            knudsen_help = "Kn_t = ((τ_e - 1/2) * √3 * Δx) / Δx = (τ_e - 1/2) * √3"
                        except Exception:
                            knudsen_number = None
                            knudsen_type = None
                            knudsen_help = None
                    else:
                        knudsen_number = None
                        knudsen_type = None
                        knudsen_help = None
                else:
                    knudsen_number = None
                    knudsen_type = None
                    knudsen_help = None
            else:
                # DNS: Kn = (c_s * (τ₀ - 1/2) * Δx) / Δx = c_s * (τ₀ - 1/2)
                # Classical Knudsen number for DNS/continuum check
                # Using smallest length scale (grid spacing) as characteristic length
                tau_0 = nu / c_s2 + 0.5
                c_s = 1.0 / np.sqrt(3.0)  # Lattice sound speed
                dx = 1.0  # Grid spacing in lattice units (Δx)
                # Use grid spacing as characteristic length (smallest resolved scale)
                knudsen_number = (c_s * (tau_0 - 0.5) * dx) / dx
                knudsen_type = "DNS"
                knudsen_help = "Kn = (c_s * (τ₀ - 1/2) * Δx) / Δx = c_s * (τ₀ - 1/2) (using smallest length scale)"
    
    # Display physics validation
    if mach_number is not None or knudsen_number is not None:
        st.header("🔬 Physics Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if mach_number is not None:
                # Mach number traffic light logic
                if mach_number > 0.1:
                    status_color = "🔴"
                    status_text = "**Invalid:** Ma > 0.1"
                    status_msg = "Compressibility effects are significant. LBM weakly compressible approximation may be invalid."
                    delta_color = "inverse"
                elif mach_number > 0.05:
                    status_color = "🟡"
                    status_text = "**Warning:** 0.05 < Ma < 0.1"
                    status_msg = "Approaching compressibility limit. Monitor for compressibility artifacts."
                    delta_color = "normal"
                else:
                    status_color = "🟢"
                    status_text = "**Valid:** Ma < 0.1"
                    status_msg = "Incompressible flow regime. Navier-Stokes approximation is valid."
                    delta_color = "off"
                
                st.metric(
                    "Mach Number", 
                    f"{mach_number:.4f}",
                    delta=None,
                    help="Ma = u_rms / c_s, where c_s = 1/√3 is the lattice sound speed"
                )
                st.markdown(f"{status_color} {status_text}")
                st.caption(status_msg)
        
        with col2:
            if knudsen_number is not None:
                # Knudsen number traffic light logic
                if knudsen_number > 0.1:
                    status_color = "🔴"
                    status_text = "**Invalid:** Kn > 0.1"
                    status_msg = "Transition regime. Boltzmann equation is not recovering Navier-Stokes hydrodynamics correctly for this scale."
                    delta_color = "inverse"
                elif knudsen_number > 0.01:
                    status_color = "🟡"
                    status_text = "**Warning:** 0.01 < Kn < 0.1"
                    status_msg = "Slip regime. Boundary conditions might be inaccurate; fine for some bulk flows but risky for DNS."
                    delta_color = "normal"
                else:
                    status_color = "🟢"
                    status_text = "**Valid:** Kn < 0.01"
                    status_msg = "Continuum regime. Navier-Stokes valid."
                    delta_color = "off"
                
                st.metric(
                    "Knudsen Number", 
                    f"{knudsen_number:.6f}",
                    delta=None,
                    help=knudsen_help if 'knudsen_help' in locals() else "Kn = (τ - 0.5) / L"
                )
                if 'knudsen_type' in locals() and knudsen_type:
                    st.caption(f"*{knudsen_type} formulation*")
                st.markdown(f"{status_color} {status_text}")
                st.caption(status_msg)
    
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

