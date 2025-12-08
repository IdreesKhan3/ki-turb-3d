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
from data_readers.vti_reader import read_vti_file, compute_compressibility_metrics as compute_compressibility_vti
from data_readers.hdf5_reader import read_hdf5_file, compute_compressibility_metrics as compute_compressibility_h5
from utils.file_detector import detect_simulation_files
from utils.theme_config import inject_theme_css

st.set_page_config(page_icon="⚫")

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("📊 Overview")
    
    # Get data directories from session state (support multiple directories)
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        # Fallback to single directory for backward compatibility
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the main page.")
        return
    
    # Show which directories are loaded
    if len(data_dirs) > 1:
        st.info(f"📁 **Multiple simulations loaded:** {len(data_dirs)} directories")
        with st.expander("View loaded directories", expanded=False):
            for i, data_dir_path in enumerate(data_dirs, 1):
                data_dir = Path(data_dir_path)
                try:
                    rel_path = data_dir.relative_to(project_root)
                    st.markdown(f"**{i}.** `APP/{rel_path}`")
                except ValueError:
                    st.markdown(f"**{i}.** `{data_dir_path}`")
        st.markdown("---")
    
    # Process each directory
    all_simulations_data = []
    
    for data_dir_path in data_dirs:
        data_dir = Path(data_dir_path)
        dir_name = data_dir.name if len(data_dirs) > 1 else "Simulation"
    
        # Detect available files
        files = detect_simulation_files(str(data_dir))
    
        # Collect data for this simulation
        sim_data = {
            'directory': dir_name,
            'path': str(data_dir),
            'files': files,
            'params': None,
            'mach_number': None,
            'knudsen_number': None,
            'is_les': False,
        }
    
        # Determine if LES: only directories in \APP\user\LES are treated as LES
        # Check if path contains APP/user/LES (case-insensitive, cross-platform)
        path_str = str(data_dir).replace('\\', '/')
        is_les_dir = '/APP/user/LES' in path_str.upper() or '\\APP\\user\\LES' in str(data_dir).upper()
        
        # Load parameters
        if files['parameters']:
            params = read_parameters(str(files['parameters'][0]))
            formatted_params = format_parameters_for_display(params)
            sim_data['params'] = formatted_params
            sim_data['is_les'] = is_les_dir  # Use directory-based detection
        
        all_simulations_data.append(sim_data)
    
    # Display parameters - show comparison if multiple, single view if one
    if len(data_dirs) > 1:
        st.header("📊 Simulation Parameters Comparison")
        
        # Create comparison table
        comparison_data = []
        for sim in all_simulations_data:
            if sim['params']:
                row = {'Directory': sim['directory']}
                for label, info in sim['params'].items():
                    row[label] = f"{info['value']} {info['unit']}"
                comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width='stretch', hide_index=True)
    else:
        # Single simulation - original detailed view
        if all_simulations_data[0]['params']:
            st.header("Simulation Parameters")
            formatted_params = all_simulations_data[0]['params']
        
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
    
    # Compute Mach and Knudsen numbers for each simulation
    for sim in all_simulations_data:
        files = sim['files']
        mach_number = None
        knudsen_number = None
        is_les = sim['is_les']
        
        if files['spectral_turb_stats']:
            # Load validation data for u_rms_real
            val_df = read_eps_validation_csv(str(files['spectral_turb_stats'][0]))
            if 'u_rms_real' in val_df.columns and len(val_df) > 0:
                u_rms_latest = val_df['u_rms_real'].iloc[-1]
                c_s = 1.0 / np.sqrt(3.0)  # Lattice sound speed
                mach_number = u_rms_latest / c_s
        
        # Compute Knudsen number
        # LES (only in \APP\user\LES): Use effective/turbulent tau (τ_e) from tau_analysis for Kn_t
        # DNS (all other directories): Use molecular tau (τ₀) from parameters file
        if files['parameters']:
            params = read_parameters(str(files['parameters'][0]))
            nu = params.get('nu', None)
            if nu is not None:
                c_s2 = 1.0 / 3.0
                
                if is_les:
                    # LES: Compute turbulent Knudsen number Kn_t using effective tau from tau_analysis
                    # Kn_t = ((τ_e - 1/2) * √3 * Δx) / Δx = (τ_e - 1/2) * √3
                    if files['tau_analysis']:
                        params = read_parameters(str(files['parameters'][0]))
                        nx = params.get('nx', None)
                        ny = params.get('ny', None)
                        nz = params.get('nz', None)
                        
                        if nx and ny and nz:
                            tau_file = str(files['tau_analysis'][-1])
                            try:
                                tau_e = read_tau_analysis_file(tau_file, nx, ny, nz)  # Effective tau
                                dx = 1.0
                                sqrt3 = np.sqrt(3.0)
                                knudsen_number = ((tau_e - 0.5) * sqrt3 * dx) / dx
                            except Exception:
                                knudsen_number = None
                        else:
                            knudsen_number = None
                    else:
                        knudsen_number = None
                else:
                    # DNS: Use molecular tau (τ₀) from parameters file (input file)
                    # Kn = (c_s * (τ₀ - 1/2) * Δx) / Δx = c_s * (τ₀ - 1/2)
                    tau_0 = nu / c_s2 + 0.5  # Molecular tau from viscosity in input file
                    c_s = 1.0 / np.sqrt(3.0)  # Lattice sound speed
                    dx = 1.0  # Grid spacing in lattice units (Δx)
                    knudsen_number = (c_s * (tau_0 - 0.5) * dx) / dx
        
        sim['mach_number'] = mach_number
        sim['knudsen_number'] = knudsen_number
        
        # Compute compressibility from velocity field files
        compressibility_metrics = None
        if files['velocity_vti']:
            # Try to read first VTI file
            try:
                vti_data = read_vti_file(str(files['velocity_vti'][0]))
                velocity = vti_data['velocity']
                compressibility_metrics = compute_compressibility_vti(velocity)
            except Exception as e:
                compressibility_metrics = None
        elif files['velocity_h5']:
            # Try to read first H5 file
            try:
                h5_data = read_hdf5_file(str(files['velocity_h5'][0]))
                velocity = h5_data['velocity']
                compressibility_metrics = compute_compressibility_h5(velocity)
            except Exception as e:
                compressibility_metrics = None
        
        sim['compressibility'] = compressibility_metrics
    
    # Display physics validation
    has_validation = any(sim['mach_number'] is not None or sim['knudsen_number'] is not None or sim['compressibility'] is not None for sim in all_simulations_data)
    
    if has_validation:
        st.header("🔬 Physics Validation")
        
        if len(data_dirs) > 1:
            # Comparison table for multiple simulations
            validation_data = []
            for sim in all_simulations_data:
                row = {'Directory': sim['directory']}
                files = sim['files']
                
                # Mach Number with reason if N/A
                if sim['mach_number'] is not None:
                    row['Mach Number'] = f"{sim['mach_number']:.4f}"
                else:
                    if not files['spectral_turb_stats']:
                        row['Mach Number'] = "N/A (no eps_real_validation*.csv)"
                    else:
                        row['Mach Number'] = "N/A (missing u_rms_real)"
                
                # Knudsen Number with reason if N/A
                if sim['knudsen_number'] is not None:
                    row['Knudsen Number'] = f"{sim['knudsen_number']:.6f}"
                else:
                    if not files['parameters']:
                        row['Knudsen Number'] = "N/A (no simulation.input)"
                    elif sim['is_les'] and not files['tau_analysis']:
                        row['Knudsen Number'] = "N/A (LES: no tau_analysis*.bin)"
                    else:
                        row['Knudsen Number'] = "N/A (computation failed)"
                
                # Compressibility (Max Divergence) with reason if N/A
                if sim['compressibility'] is not None:
                    max_div = sim['compressibility']['max_divergence']
                    row['Max Divergence |∇·u|'] = f"{max_div:.6e}"
                else:
                    if not files['velocity_vti'] and not files['velocity_h5']:
                        row['Max Divergence |∇·u|'] = "N/A (no .vti or .h5 files)"
                    else:
                        row['Max Divergence |∇·u|'] = "N/A (computation failed)"
                
                validation_data.append(row)
            
            if validation_data:
                validation_df = pd.DataFrame(validation_data)
                st.dataframe(validation_df, width='stretch', hide_index=True)
                st.caption("💡 N/A values indicate missing required files or data for computation")
        else:
            # Single simulation - detailed view
            sim = all_simulations_data[0]
            mach_number = sim['mach_number']
            knudsen_number = sim['knudsen_number']
            compressibility = sim['compressibility']
            
            col1, col2, col3 = st.columns(3)
        
        with col1:
            if mach_number is not None:
                # Mach number traffic light logic
                if mach_number > 0.1:
                    status_color = "🔴"
                    status_text = "**Invalid:** Ma > 0.1"
                    status_msg = "Compressibility effects are significant. LBM weakly compressible approximation may be invalid."
                elif mach_number > 0.05:
                    status_color = "🟡"
                    status_text = "**Warning:** 0.05 < Ma < 0.1"
                    status_msg = "Approaching compressibility limit. Monitor for compressibility artifacts."
                else:
                    status_color = "🟢"
                    status_text = "**Valid:** Ma < 0.1"
                    status_msg = "Incompressible flow regime. Navier-Stokes approximation is valid."
                
                st.metric(
                    "Mach Number", 
                    f"{mach_number:.4f}",
                    delta=None
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
                    elif knudsen_number > 0.01:
                        status_color = "🟡"
                        status_text = "**Warning:** 0.01 < Kn < 0.1"
                        status_msg = "Slip regime. Boundary conditions might be inaccurate; fine for some bulk flows but risky for DNS."
                    else:
                        status_color = "🟢"
                        status_text = "**Valid:** Kn < 0.01"
                        status_msg = "Continuum regime. Navier-Stokes valid."
                    
                    st.metric(
                        "Knudsen Number", 
                        f"{knudsen_number:.6f}",
                        delta=None
                    )
                    st.markdown(f"{status_color} {status_text}")
                    st.caption(status_msg)
            
            with col3:
                if compressibility is not None:
                    max_div = compressibility['max_divergence']
                    rms_div = compressibility['rms_divergence']
                    
                    # Compressibility traffic light logic
                    # For incompressible flow, divergence should be very small
                    # Typical threshold: |∇·u| < 1e-6 for well-resolved DNS
                    if max_div > 1e-3:
                        status_color = "🔴"
                        status_text = "**Invalid:** |∇·u| > 1e-3"
                        status_msg = "Significant compressibility. Flow violates incompressibility assumption."
                    elif max_div > 1e-5:
                        status_color = "🟡"
                        status_text = "**Warning:** 1e-5 < |∇·u| < 1e-3"
                        status_msg = "Moderate compressibility. May indicate numerical errors or compressibility effects."
                    else:
                        status_color = "🟢"
                        status_text = "**Valid:** |∇·u| < 1e-5"
                        status_msg = "Incompressible flow. Divergence is within acceptable limits."
                    
                    st.metric(
                        "Max Divergence |∇·u|", 
                        f"{max_div:.6e}",
                        delta=None
                    )
                    st.caption(f"RMS: {rms_div:.6e}")
                    st.markdown(f"{status_color} {status_text}")
                    st.caption(status_msg)
                else:
                    files = all_simulations_data[0]['files']
                    if not files['velocity_vti'] and not files['velocity_h5']:
                        st.metric("Max Divergence |∇·u|", "N/A")
                        st.caption("No .vti or .h5 velocity field files found")
                    else:
                        st.metric("Max Divergence |∇·u|", "N/A")
                        st.caption("Failed to compute compressibility")
    
    # File availability checklist
    st.header("Data Availability")
    
    if len(data_dirs) > 1:
        # Comparison table for multiple simulations
        availability_data = []
        for sim in all_simulations_data:
            files = sim['files']
            row = {'Directory': sim['directory']}
            row['Real Turbulence Stats'] = "✅" if len(files['real_turb_stats']) > 0 else "❌"
            row['Energy Spectra'] = "✅" if len(files['spectrum']) > 0 else "❌"
            row['Normalized Spectra'] = "✅" if len(files['norm_spectrum']) > 0 else "❌"
            row['Structure Functions'] = "✅" if (len(files['structure_functions_txt']) > 0 or len(files['structure_functions_bin']) > 0) else "❌"
            row['Flatness'] = "✅" if len(files['flatness']) > 0 else "❌"
            row['Isotropy'] = "✅" if len(files['isotropy']) > 0 else "❌"
            row['Spectral Turbulence Stats'] = "✅" if len(files['spectral_turb_stats']) > 0 else "❌"
            availability_data.append(row)
        
        availability_df = pd.DataFrame(availability_data)
        st.dataframe(availability_df, width='stretch', hide_index=True)
        st.caption("💡 ❌ indicates the file type is not found in that directory. ✅ means files are available.")
    else:
        # Single simulation - original checklist
        files = all_simulations_data[0]['files']
    checklist = {
        'Real Turbulence Stats': len(files['real_turb_stats']) > 0,
        'Energy Spectra': len(files['spectrum']) > 0,
        'Normalized Spectra': len(files['norm_spectrum']) > 0,
        'Structure Functions': len(files['structure_functions_txt']) > 0 or len(files['structure_functions_bin']) > 0,
        'Flatness': len(files['flatness']) > 0,
        'Isotropy': len(files['isotropy']) > 0,
        'Spectral Turbulence Stats': len(files['spectral_turb_stats']) > 0,
    }
    
    for item, available in checklist.items():
        status = "✅" if available else "❌"
        st.markdown(f"{status} {item}")
    
    # Theory Equations Section
    st.markdown("---")
    st.header("📚 Theory Equations")
    
    with st.expander("**Physics Validation Equations**", expanded=False):
        st.markdown("**Mach Number:**")
        st.latex(r"""
        \text{Ma} = \frac{u_{\text{rms}}}{c_s}
        """)
        st.markdown(r"""
        where $u_{\text{rms}} = \sqrt{\langle u_x^2 + u_y^2 + u_z^2 \rangle}$ is the root-mean-square velocity and $c_s = 1/\sqrt{3}$ is the lattice sound speed. For incompressible flow: $\text{Ma} < 0.1$
        """)
        
        st.markdown("---")
        st.markdown("**Knudsen Number (DNS/Continuum Regime):**")
        st.latex(r"""
        \text{Kn} = \frac{c_s (\tau_0 - 1/2) \Delta x}{\Delta x} = c_s \left(\tau_0 - \frac{1}{2}\right)
        """)
        st.markdown(r"""
        where $\tau_0 = \nu_0/c_s^2 + 1/2$ is the molecular relaxation time from the input parameters, $\nu_0$ is the molecular viscosity, and $c_s = 1/\sqrt{3}$ is the lattice sound speed. Continuum regime: $\text{Kn} < 0.01$
        """)
        
        st.markdown("---")
        st.markdown("**Knudsen Number (LES/Turbulent Regime):**")
        st.latex(r"""
        \text{Kn}_t = \frac{(\tau_e - 1/2) \sqrt{3} \Delta x}{\Delta x} = \sqrt{3} \left(\tau_e - \frac{1}{2}\right)
        """)
        st.markdown(r"""
        where $\tau_e$ is the effective relaxation time computed from the turbulent viscosity analysis. For LES simulations, this uses the effective tau from tau_analysis files. Continuum regime: $\text{Kn}_t < 0.01$
        """)
        
        st.markdown("---")
        st.markdown("**Velocity Divergence (Compressibility Check):**")
        st.latex(r"""
        \nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z}
        """)
        st.markdown(r"""
        For incompressible flow, the divergence should be zero: $\nabla \cdot \mathbf{u} = 0$. The maximum absolute divergence $|\nabla \cdot \mathbf{u}|_{\max}$ is used to validate incompressibility.
        """)
        
        st.markdown("**Compressibility Metrics:**")
        st.latex(r"""
        \begin{align}
        |\nabla \cdot \mathbf{u}|_{\max} &= \max_{x,y,z} |\nabla \cdot \mathbf{u}| \\
        \text{RMS}(\nabla \cdot \mathbf{u}) &= \sqrt{\frac{1}{V} \int_V (\nabla \cdot \mathbf{u})^2 \, dV}
        \end{align}
        """)
        st.markdown(r"""
        Validation thresholds: $|\nabla \cdot \mathbf{u}|_{\max} < 10^{-5}$ (valid), $10^{-5} < |\nabla \cdot \mathbf{u}|_{\max} < 10^{-3}$ (warning), $|\nabla \cdot \mathbf{u}|_{\max} > 10^{-3}$ (invalid)
        """)

if __name__ == "__main__":
    main()

