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
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from data_readers.csv_reader import read_eps_validation_csv
from data_readers.parameter_reader import read_parameters, format_parameters_for_display
from data_readers.binary_reader import read_tau_analysis_file
from data_readers.hdf5_reader import read_hdf5_file, compute_compressibility_metrics as compute_compressibility_h5
from utils.file_detector import detect_simulation_files
from utils.theme_config import inject_theme_css

st.set_page_config(page_icon="⚫")

@st.cache_data
def read_parameters_cached(filepath: str, mtime: float):
    """
    Cached parameter reader keyed by file path + modification time.
    
    Args:
        filepath: Path to simulation.input file
        mtime: File modification time (for cache invalidation)
    
    Returns:
        Dictionary of parameters
    """
    return read_parameters(filepath)

@st.cache_data
def compute_compressibility_from_slice(filepath: str, mtime: float, max_size: int = 128):
    """
    Compute compressibility metrics from a subsampled slice of velocity field (.h5 only).
    Reads only a slice from disk (memory-efficient for large files).
    Uses caching keyed by file path + modification time.
    
    Args:
        filepath: Path to .h5 velocity file
        mtime: File modification time (for cache invalidation)
        max_size: Maximum size for subsampling (default 128^3)
    
    Returns:
        Dictionary with compressibility metrics or None if error
    """
    try:
        import h5py
        import numpy as np
        
        with h5py.File(filepath, 'r') as f:
            # Find velocity dataset
            if 'velocity' in f:
                velocity_ds = f['velocity']
            elif 'Velocity' in f:
                velocity_ds = f['Velocity']
            elif 'u' in f:
                velocity_ds = f['u']
            else:
                # Try to find any 4D dataset
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data = f[key]
                        if len(data.shape) == 4 and (data.shape[3] == 3 or data.shape[0] == 3):
                            velocity_ds = data
                            break
                else:
                    raise ValueError("Could not find velocity data in HDF5 file")
            
            # Get original shape (before any transpose)
            orig_shape = velocity_ds.shape
            
            # Determine format and dimensions
            if len(orig_shape) == 4:
                if orig_shape[0] == 3:
                    # Format: (3, nx, ny, nz) - need to transpose
                    ncomp, nx, ny, nz = orig_shape
                    needs_transpose = True
                elif orig_shape[3] == 3:
                    # Format: (nx, ny, nz, 3)
                    nx, ny, nz, ncomp = orig_shape
                    needs_transpose = False
                else:
                    raise ValueError(f"Unexpected velocity shape: {orig_shape}")
            else:
                raise ValueError(f"Expected 4D velocity array, got shape {orig_shape}")
            
            if ncomp != 3:
                raise ValueError(f"Expected 3 velocity components, got {ncomp}")
            
            # Determine if we need to subsample
            total_points = nx * ny * nz
            needs_subsample = total_points > max_size ** 3
            
            if needs_subsample:
                # Read only a slice from disk (memory-efficient)
                # Use middle slice in z-direction (most representative)
                z_mid = nz // 2
                
                # Calculate subsampling steps
                y_step = max(1, ny // max_size) if ny > max_size else 1
                x_step = max(1, nx // max_size) if nx > max_size else 1
                
                # Read slice directly from disk
                if orig_shape[0] == 3:
                    # Format: (3, nx, ny, nz) - read slice in z (last dimension)
                    velocity_slice = velocity_ds[:, ::x_step, ::y_step, z_mid:z_mid+1]
                    # Transpose to (z, y, x, 3) format
                    velocity_slice = np.transpose(velocity_slice, (3, 2, 1, 0))
                else:
                    # Format: (nx, ny, nz, 3) - read slice in z (third dimension)
                    velocity_slice = velocity_ds[::x_step, ::y_step, z_mid:z_mid+1, :]
                    # Transpose to (z, y, x, 3) format to match expected format
                    velocity_slice = np.transpose(velocity_slice, (2, 1, 0, 3))
            else:
                # Small enough - read full array but still transpose if needed
                if orig_shape[0] == 3:
                    velocity_slice = np.array(velocity_ds)
                    velocity_slice = np.transpose(velocity_slice, (3, 2, 1, 0))
                else:
                    velocity_slice = np.array(velocity_ds)
                    velocity_slice = np.transpose(velocity_slice, (2, 1, 0, 3))
            
            # Compute compressibility on slice
            return compute_compressibility_h5(velocity_slice)
    except Exception:
        return None

def is_examples_les_dir(data_dir: Path, project_root: Path) -> bool:
    """
    Strict path-based LES detection: only examples/LES/* directories are considered LES.
    
    Args:
        data_dir: Path to simulation directory
        project_root: Path to project root
    
    Returns:
        True if directory is under examples/LES/, False otherwise
    """
    try:
        rel = data_dir.resolve().relative_to(project_root.resolve())
    except Exception:
        return False
    parts = [p.lower() for p in rel.parts]
    return len(parts) >= 2 and parts[0] == "examples" and parts[1] == "les"

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("Overview")
    
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
                    st.markdown(f"**{i}.** `{rel_path}`")
                except ValueError:
                    st.markdown(f"**{i}.** `{data_dir_path}`")
        st.markdown("---")
    
    # Process each directory
    all_simulations_data = []
    
    for data_dir_path in data_dirs:
        # Resolve path to ensure it works regardless of how it was stored
        try:
            data_dir = Path(data_dir_path).resolve()
            if not data_dir.exists() or not data_dir.is_dir():
                st.warning(f"Directory not found or invalid: {data_dir_path}")
                continue
        except Exception as e:
            st.warning(f"Error processing directory {data_dir_path}: {str(e)}")
            continue
        
        dir_name = data_dir.name if len(data_dirs) > 1 else "Simulation"
    
        # Detect available files - each directory is processed independently
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
    
        # Load parameters and cache in sim_data to avoid re-reading
        params = None
        if files['parameters']:
            param_file = str(files['parameters'][0])
            try:
                mtime = Path(param_file).stat().st_mtime
                params = read_parameters_cached(param_file, mtime)
            except Exception:
                params = None
            
            if params:
                formatted_params = format_parameters_for_display(params)
                sim_data['params'] = formatted_params
                sim_data['raw_params'] = params  # Store raw params for later use
        
        # Strict path-based LES detection: only examples/LES/* directories
        sim_data['is_les'] = is_examples_les_dir(data_dir, project_root)
        
        all_simulations_data.append(sim_data)
    
    # Guard: Check if any valid directories were processed
    if not all_simulations_data:
        st.error("No valid simulation directories could be processed.")
        st.info("Please check that the directories exist and contain simulation files.")
        return
    
    # Display parameters - show comparison if multiple, single view if one
    if len(data_dirs) > 1:
        st.header("Simulation Parameters Comparison")
        
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
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
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
        
        # Improved Mach computation: iterate over all candidate CSVs, prefer newest
        if files.get('spectral_turb_stats'):
            candidates = sorted(
                [Path(p) for p in files['spectral_turb_stats']],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            for csv_path in candidates:
                try:
                    df = read_eps_validation_csv(str(csv_path))
                    if 'u_rms_real' in df.columns and len(df) > 0:
                        u_rms_latest = df['u_rms_real'].iloc[-1]
                        c_s = 1.0 / np.sqrt(3.0)  # Lattice sound speed
                        mach_number = u_rms_latest / c_s
                        break
                except Exception:
                    continue
        
        # Compute Knudsen number
        # Always compute molecular Kn if nu exists
        # Only if is_les is True, attempt turbulent and override
        params = sim.get('raw_params', None)
        if params is None and files['parameters']:
            # Fallback: read if not cached (shouldn't happen, but safe)
            param_file = str(files['parameters'][0])
            try:
                mtime = Path(param_file).stat().st_mtime
                params = read_parameters_cached(param_file, mtime)
                sim['raw_params'] = params  # Cache for future use
            except Exception:
                params = None
        
        if params:
            nu = params.get('nu', None)
            if nu is not None:
                c_s2 = 1.0 / 3.0
                c_s = 1.0 / np.sqrt(3.0)  # Lattice sound speed
                dx = 1.0  # Grid spacing in lattice units (Δx)
                
                # Always compute molecular Kn first
                tau_0 = nu / c_s2 + 0.5  # Molecular tau from viscosity in input file
                knudsen_number = (c_s * (tau_0 - 0.5) * dx) / dx
                
                # Only override with turbulent Kn if this is a strict LES directory
                if is_les and files['tau_analysis']:
                    nx = params.get('nx', None)
                    ny = params.get('ny', None)
                    nz = params.get('nz', None)
                    
                    if nx and ny and nz:
                        tau_file = str(files['tau_analysis'][-1])
                        try:
                            tau_e = read_tau_analysis_file(tau_file, nx, ny, nz)  # Effective tau
                            sqrt3 = np.sqrt(3.0)
                            knudsen_number = ((tau_e - 0.5) * sqrt3 * dx) / dx
                        except Exception:
                            pass  # Keep molecular Kn if turbulent computation fails
        
        sim['mach_number'] = mach_number
        sim['knudsen_number'] = knudsen_number
        
        # Compute compressibility from velocity field files (using cached subsample, .h5 only)
        compressibility_metrics = None
        if files['velocity_h5']:
            filepath = str(files['velocity_h5'][0])
            try:
                mtime = Path(filepath).stat().st_mtime
                compressibility_metrics = compute_compressibility_from_slice(filepath, mtime)
            except Exception:
                compressibility_metrics = None
        
        sim['compressibility'] = compressibility_metrics
    
    # Display physics validation
    has_validation = any(sim['mach_number'] is not None or sim['knudsen_number'] is not None or sim['compressibility'] is not None for sim in all_simulations_data)
    
    if has_validation:
        st.header("Physics Validation")
        
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
                    kn_label = " (turbulent)" if sim['is_les'] else " (molecular)"
                    row['Knudsen Number'] = f"{sim['knudsen_number']:.6f}{kn_label}"
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
                    if not files['velocity_h5']:
                        row['Max Divergence |∇·u|'] = "N/A (no .h5 files)"
                    else:
                        row['Max Divergence |∇·u|'] = "N/A (computation failed)"
                
                validation_data.append(row)
            
            if validation_data:
                validation_df = pd.DataFrame(validation_data)
                st.dataframe(validation_df, use_container_width=True, hide_index=True)
                st.caption("N/A values indicate missing required files or data for computation")
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
                    is_les = sim['is_les']
                    kn_label = "Kn_t" if is_les else "Kn"
                    kn_type = "turbulent" if is_les else "molecular"
                    
                    if knudsen_number > 0.1:
                        status_color = "🔴"
                        status_text = f"**Invalid:** {kn_label} > 0.1"
                        status_msg = "Transition regime. Boltzmann equation is not recovering Navier-Stokes hydrodynamics correctly for this scale."
                    elif knudsen_number > 0.01:
                        status_color = "🟡"
                        status_text = f"**Warning:** 0.01 < {kn_label} < 0.1"
                        status_msg = "Slip regime. Boundary conditions might be inaccurate; fine for some bulk flows but risky for DNS."
                    else:
                        status_color = "🟢"
                        status_text = f"**Valid:** {kn_label} < 0.01"
                        status_msg = "Continuum regime. Navier-Stokes valid."
                    
                    metric_label = f"Knudsen Number ({kn_type})"
                    st.metric(
                        metric_label, 
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
                    if not files['velocity_h5']:
                        st.metric("Max Divergence |∇·u|", "N/A")
                        st.caption("No .h5 velocity field files found")
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
            row['Real Turbulence Stats'] = "Yes" if len(files['real_turb_stats']) > 0 else "No"
            row['Energy Spectra'] = "Yes" if len(files['spectrum']) > 0 else "No"
            row['Normalized Spectra'] = "Yes" if len(files['norm_spectrum']) > 0 else "No"
            row['Structure Functions'] = "Yes" if (len(files['structure_functions_txt']) > 0 or len(files['structure_functions_bin']) > 0) else "No"
            row['Flatness'] = "Yes" if len(files['flatness']) > 0 else "No"
            row['Isotropy'] = "Yes" if len(files['isotropy']) > 0 else "No"
            row['Spectral Turbulence Stats'] = "Yes" if len(files['spectral_turb_stats']) > 0 else "No"
            availability_data.append(row)
        
        availability_df = pd.DataFrame(availability_data)
        st.dataframe(availability_df, use_container_width=True, hide_index=True)
        st.caption("No indicates the file type is not found in that directory. Yes means files are available.")
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
            status = "Yes" if available else "No"
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

