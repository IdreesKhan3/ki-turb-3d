"""
Vorticity & Enstrophy Statistical Analysis
Module for computing and visualizing vorticity magnitude and enstrophy PDFs
"""

import streamlit as st
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from utils.iso_surfaces import compute_vorticity_vector


def compute_vorticity_pdf(velocity, bins=100, dx=1.0, dy=1.0, dz=1.0, normalize=False):
    """
    Compute smooth Probability Density Function for vorticity magnitude |ω|
    Uses Kernel Density Estimation (KDE) to produce smooth curves
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        bins: Number of evaluation points for smooth curve
        dx, dy, dz: Grid spacing (default 1.0)
        normalize: If True, normalize by RMS (|ω|/σ_|ω|)
        
    Returns:
        omega_grid, pdf_omega (arrays) - smooth PDF curve
    """
    # Compute vorticity vector
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    
    # Compute magnitude: |ω| = √(ωx² + ωy² + ωz²)
    omega_mag = np.sqrt(
        vorticity[:, :, :, 0]**2 + 
        vorticity[:, :, :, 1]**2 + 
        vorticity[:, :, :, 2]**2
    )
    
    # Flatten and remove NaN/Inf
    omega_flat = omega_mag.flatten()
    omega_flat = omega_flat[np.isfinite(omega_flat)]
    
    if len(omega_flat) == 0:
        return np.array([]), np.array([])
    
    # Normalize by RMS if requested (standard for vorticity: |ω|/σ_|ω|)
    normalization_factor = 1.0
    if normalize:
        rms_omega = np.sqrt(np.mean(omega_flat**2))
        if rms_omega > 0:
            omega_flat = omega_flat / rms_omega
            normalization_factor = rms_omega
    
    # Determine range
    omega_min = omega_flat.min()
    omega_max = omega_flat.max()
    
    # Add padding for smooth evaluation at edges
    omega_range = omega_max - omega_min
    omega_min -= 0.1 * omega_range
    omega_max += 0.1 * omega_range
    
    # Create fine grid for smooth curve evaluation
    omega_grid = np.linspace(omega_min, omega_max, bins)
    
    # Compute KDE for smooth PDF curve
    try:
        kde = gaussian_kde(omega_flat)
        pdf_omega = kde(omega_grid)
    except:
        # Fallback to histogram if KDE fails
        counts, edges = np.histogram(omega_flat, bins=bins, range=(omega_min, omega_max), density=True)
        pdf_omega = counts
        omega_grid = (edges[:-1] + edges[1:]) / 2
    
    # Normalize Y-axis: multiply by normalization_factor to preserve area = 1
    if normalize and normalization_factor > 0:
        pdf_omega = pdf_omega * normalization_factor
    
    return omega_grid, pdf_omega


def compute_vorticity_statistics(velocity, dx=1.0, dy=1.0, dz=1.0):
    """
    Compute statistical moments (mean, RMS, skewness, kurtosis) for vorticity magnitude.
    
    Returns:
        mean, rms, skewness, kurtosis (floats)
    """
    from utils.iso_surfaces import compute_vorticity_vector
    from utils.velocity_magnitude_stats import compute_skewness_kurtosis
    
    # Compute vorticity vector
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    
    # Compute magnitude: |ω| = √(ωx² + ωy² + ωz²)
    omega_mag = np.sqrt(
        vorticity[:, :, :, 0]**2 + 
        vorticity[:, :, :, 1]**2 + 
        vorticity[:, :, :, 2]**2
    )
    
    # Flatten
    omega_flat = omega_mag.flatten()
    
    # Compute statistics
    mean, rms, skewness, kurtosis = compute_skewness_kurtosis(omega_flat)
    
    return mean, rms, skewness, kurtosis


def compute_enstrophy_pdf(velocity, bins=100, dx=1.0, dy=1.0, dz=1.0, normalize=False):
    """
    Compute smooth Probability Density Function for enstrophy Ω = |ω|²
    Uses Kernel Density Estimation (KDE) to produce smooth curves
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        bins: Number of evaluation points for smooth curve
        dx, dy, dz: Grid spacing (default 1.0)
        normalize: If True, normalize by mean (Ω/⟨Ω⟩)
        
    Returns:
        enstrophy_grid, pdf_enstrophy (arrays) - smooth PDF curve
    """
    # Compute vorticity vector
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    
    # Compute enstrophy: Ω = |ω|² = ωx² + ωy² + ωz²
    enstrophy = (
        vorticity[:, :, :, 0]**2 + 
        vorticity[:, :, :, 1]**2 + 
        vorticity[:, :, :, 2]**2
    )
    
    # Flatten and remove NaN/Inf
    enstrophy_flat = enstrophy.flatten()
    enstrophy_flat = enstrophy_flat[np.isfinite(enstrophy_flat)]
    
    if len(enstrophy_flat) == 0:
        return np.array([]), np.array([])
    
    # Normalize by mean if requested (standard for enstrophy: Ω/⟨Ω⟩)
    normalization_factor = 1.0
    if normalize:
        mean_enstrophy = np.mean(enstrophy_flat)
        if mean_enstrophy > 0:
            enstrophy_flat = enstrophy_flat / mean_enstrophy
            normalization_factor = mean_enstrophy
    
    # Determine range
    enstrophy_min = enstrophy_flat.min()
    enstrophy_max = enstrophy_flat.max()
    
    # Add padding for smooth evaluation at edges
    enstrophy_range = enstrophy_max - enstrophy_min
    enstrophy_min -= 0.1 * enstrophy_range
    enstrophy_max += 0.1 * enstrophy_range
    
    # Create fine grid for smooth curve evaluation
    enstrophy_grid = np.linspace(enstrophy_min, enstrophy_max, bins)
    
    # Compute KDE for smooth PDF curve
    try:
        kde = gaussian_kde(enstrophy_flat)
        pdf_enstrophy = kde(enstrophy_grid)
    except:
        # Fallback to histogram if KDE fails
        counts, edges = np.histogram(enstrophy_flat, bins=bins, range=(enstrophy_min, enstrophy_max), density=True)
        pdf_enstrophy = counts
        enstrophy_grid = (edges[:-1] + edges[1:]) / 2
    
    # Normalize Y-axis: multiply by normalization_factor to preserve area = 1
    if normalize and normalization_factor > 0:
        pdf_enstrophy = pdf_enstrophy * normalization_factor
    
    return enstrophy_grid, pdf_enstrophy


def compute_enstrophy_statistics(velocity, dx=1.0, dy=1.0, dz=1.0):
    """
    Compute statistical moments (mean, RMS, skewness, kurtosis) for enstrophy.
    
    Returns:
        mean, rms, skewness, kurtosis (floats)
    """
    from utils.iso_surfaces import compute_vorticity_vector
    from utils.velocity_magnitude_stats import compute_skewness_kurtosis
    
    # Compute vorticity vector
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    
    # Compute enstrophy: Ω = |ω|² = ωx² + ωy² + ωz²
    enstrophy = (
        vorticity[:, :, :, 0]**2 + 
        vorticity[:, :, :, 1]**2 + 
        vorticity[:, :, :, 2]**2
    )
    
    # Flatten
    enstrophy_flat = enstrophy.flatten()
    
    # Compute statistics
    mean, rms, skewness, kurtosis = compute_skewness_kurtosis(enstrophy_flat)
    
    return mean, rms, skewness, kurtosis


def render_vorticity_stats_tab(data_dir_or_dirs, load_velocity_file_func,
                               get_plot_style_func=None, apply_plot_style_func=None,
                               get_palette_func=None, resolve_line_style_func=None,
                               export_panel_func=None, capture_button_func=None):
    """
    Render the Vorticity & Enstrophy PDFs tab content
    
    Args:
        data_dir_or_dirs: Path to data directory (Path) or list of directories (list of Path/str)
        load_velocity_file_func: Function to load velocity files (takes filepath)
        get_plot_style_func: Optional function to get plot style (plot_name) -> style_dict
        apply_plot_style_func: Optional function to apply plot style (fig, style_dict) -> fig
        get_palette_func: Optional function to get color palette (style_dict) -> color_list
        resolve_line_style_func: Optional function to resolve line style for files
        export_panel_func: Optional function to show export panel (fig, out_dir, base_name)
        capture_button_func: Optional function to add capture button (fig, title, source_page)
    """
    import glob
    from pathlib import Path
    from utils.file_detector import natural_sort_key
    
    st.header("Vorticity & Enstrophy PDFs")
    st.markdown("Compare vorticity magnitude and enstrophy PDFs across different simulations/methods.")
    
    # Handle both single directory and multiple directories
    if isinstance(data_dir_or_dirs, (list, tuple)):
        data_dirs = [Path(d).resolve() for d in data_dir_or_dirs]
        data_dir = data_dirs[0]  # Use first for metadata
    else:
        data_dirs = [Path(data_dir_or_dirs).resolve()]
        data_dir = data_dirs[0]
    
    # Find velocity files from ALL directories independently
    all_vti_files = []
    all_hdf5_files = []
    
    for dir_path in data_dirs:
        if dir_path.exists() and dir_path.is_dir():
            dir_vti = sorted(
                glob.glob(str(dir_path / "*.vti")) + 
                glob.glob(str(dir_path / "*.VTI")),
                key=natural_sort_key
            )
            dir_hdf5 = sorted(
                glob.glob(str(dir_path / "*.h5")) + 
                glob.glob(str(dir_path / "*.H5")) +
                glob.glob(str(dir_path / "*.hdf5")) + 
                glob.glob(str(dir_path / "*.HDF5")),
                key=natural_sort_key
            )
            all_vti_files.extend(dir_vti)
            all_hdf5_files.extend(dir_hdf5)
    
    all_files = all_vti_files + all_hdf5_files
    
    if not all_files:
        st.error("No velocity files found. Expected: `*.vti`, `*.h5`, or `*.hdf5`")
        return
    
    # Create mapping from filename to full path (handle files from different directories)
    filename_to_path = {Path(f).name: f for f in all_files}
    
    # File selection - independent for each plot
    st.sidebar.header("📁 File Selection")
    st.sidebar.caption(f"Found {len(all_files)} velocity files")
    
    # Separate file selection for each plot
    selected_files_vort = st.sidebar.multiselect(
        "Vorticity PDF files:",
        options=[Path(f).name for f in all_files],
        default=[Path(f).name for f in all_files[:min(2, len(all_files))]],
        help="Select files for Vorticity Magnitude PDF plot (left)",
        key="vorticity_file_select"
    )
    
    selected_files_enst = st.sidebar.multiselect(
        "Enstrophy PDF files:",
        options=[Path(f).name for f in all_files],
        default=[Path(f).name for f in all_files[:min(2, len(all_files))]],
        help="Select files for Enstrophy PDF plot (right)",
        key="enstrophy_file_select"
    )
    
    if not selected_files_vort and not selected_files_enst:
        st.warning("Please select at least one file for either plot.")
        return
    
    # Plot parameters
    st.sidebar.header("📊 Plot Parameters")
    pdf_bins = st.sidebar.slider("PDF bins", 50, 500, 100, 10, key="vorticity_pdf_bins")
    normalize_pdf = st.sidebar.checkbox(
        "Normalize PDFs",
        value=False,
        help="Normalize: |ω| by RMS, Ω by mean",
        key="vorticity_normalize"
    )
    
    # Load and compute data independently for each plot
    vort_data = {}
    enst_data = {}
    
    # Load data for Vorticity PDF plot
    if selected_files_vort:
        for filename in selected_files_vort:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for Vorticity PDF..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"⚠️ {filename}: Invalid velocity shape")
                        continue
                    
                    # Compute PDF
                    omega_grid, pdf_omega = compute_vorticity_pdf(velocity, bins=pdf_bins, dx=1.0, dy=1.0, dz=1.0, normalize=normalize_pdf)
                    vort_data[filename] = (omega_grid, pdf_omega)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for Vorticity PDF: {e}")
                continue
    
    # Load data for Enstrophy PDF plot
    if selected_files_enst:
        for filename in selected_files_enst:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for Enstrophy PDF..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"⚠️ {filename}: Invalid velocity shape")
                        continue
                    
                    # Compute PDF
                    enstrophy_grid, pdf_enstrophy = compute_enstrophy_pdf(velocity, bins=pdf_bins, dx=1.0, dy=1.0, dz=1.0, normalize=normalize_pdf)
                    enst_data[filename] = (enstrophy_grid, pdf_enstrophy)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for Enstrophy PDF: {e}")
                continue
    
    if not vort_data and not enst_data:
        st.error("No valid velocity data loaded.")
        return
    
    # ============================================
    # Statistics Section
    # ============================================
    with st.expander("📈 Statistical Moments (Skewness & Kurtosis)", expanded=False):
        from utils.velocity_magnitude_stats import display_statistics_table
        
        # Compute statistics from first available file
        stats_dict = {}
        if selected_files_vort:
            first_file = selected_files_vort[0]
            filepath = data_dir / first_file
            try:
                vti_data = load_velocity_file_func(str(filepath))
                velocity = vti_data['velocity']
                if velocity is not None and len(velocity.shape) == 4:
                    mean, rms, skew, kurt = compute_vorticity_statistics(velocity, dx=1.0, dy=1.0, dz=1.0)
                    stats_dict['vorticity'] = (mean, rms, skew, kurt)
            except:
                pass
        
        if selected_files_enst:
            first_file = selected_files_enst[0]
            filepath = data_dir / first_file
            try:
                vti_data = load_velocity_file_func(str(filepath))
                velocity = vti_data['velocity']
                if velocity is not None and len(velocity.shape) == 4:
                    mean, rms, skew, kurt = compute_enstrophy_statistics(velocity, dx=1.0, dy=1.0, dz=1.0)
                    stats_dict['enstrophy'] = (mean, rms, skew, kurt)
            except:
                pass
        
        if stats_dict:
            display_statistics_table(stats_dict, title="Vorticity & Enstrophy Statistics")
        else:
            st.info("Statistics will be computed when files are loaded.")
    
    st.markdown("---")
    
    # Create side-by-side plots
    col1, col2 = st.columns(2)
    
    # ============================================
    # Left: Vorticity Magnitude PDF
    # ============================================
    with col1:
        st.subheader("Vorticity Magnitude PDF")
        
        if not vort_data:
            st.info("Select files in the sidebar to plot Vorticity Magnitude PDF")
        else:
            plot_name_vort = "Vorticity PDF"
            ps_vort = get_plot_style_func(plot_name_vort) if get_plot_style_func else {}
            colors_vort = get_palette_func(ps_vort) if get_palette_func else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            line_width = ps_vort.get("line_width", 2.4) if ps_vort else 2.0
            
            fig_vort = go.Figure()
            
            for idx, (filename, (omega_grid, pdf_omega)) in enumerate(vort_data.items()):
                if len(omega_grid) == 0:
                    continue
                
                label_base = Path(filename).stem
                
                if resolve_line_style_func:
                    color, lw, dash = resolve_line_style_func(
                        filename, idx, colors_vort, ps_vort,
                        style_key="per_sim_style_comparison",
                        include_marker=False,
                        default_marker="circle"
                    )
                else:
                    color = colors_vort[idx % len(colors_vort)]
                    lw = line_width
                    dash = "solid"
                
                fig_vort.add_trace(go.Scatter(
                    x=omega_grid,
                    y=pdf_omega,
                    mode='lines',
                    name=label_base,
                    line=dict(color=color, width=lw, dash=dash),
                    hovertemplate=f"|ω| = %{{x:.4f}}<br>PDF = %{{y:.4e}}<extra>{label_base}</extra>"
                ))
            
            x_label_vort = "|ω| / σ<sub>|ω|</sub>" if normalize_pdf else "|ω|"
            y_label_vort = "σ<sub>|ω|</sub> P(|ω| / σ<sub>|ω|</sub>)" if normalize_pdf else "P(|ω|)"
            layout_kwargs = dict(
                xaxis_title=x_label_vort,
                yaxis_title=y_label_vort,
                height=ps_vort.get("figure_height", 500) if ps_vort else 500,
                hovermode='x unified',
                legend=dict(x=1.02, y=1)
            )
            
            if ps_vort:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs = apply_axis_limits(layout_kwargs, ps_vort)
                layout_kwargs = apply_figure_size(layout_kwargs, ps_vort)
            
            fig_vort.update_layout(**layout_kwargs)
            
            if apply_plot_style_func and ps_vort:
                fig_vort = apply_plot_style_func(fig_vort, ps_vort)
            
            st.plotly_chart(
                fig_vort, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "vorticity_pdf",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_vort, title="Vorticity Magnitude PDF", source_page="PDFs")
            
            if export_panel_func:
                export_panel_func(fig_vort, data_dir, "vorticity_pdf")
    
    # ============================================
    # Right: Enstrophy PDF
    # ============================================
    with col2:
        st.subheader("Enstrophy PDF")
        
        if not enst_data:
            st.info("Select files in the sidebar to plot Enstrophy PDF")
        else:
            plot_name_enst = "Enstrophy PDF"
            ps_enst = get_plot_style_func(plot_name_enst) if get_plot_style_func else {}
            colors_enst = get_palette_func(ps_enst) if get_palette_func else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            line_width = ps_enst.get("line_width", 2.4) if ps_enst else 2.0
            
            fig_enst = go.Figure()
            
            for idx, (filename, (enstrophy_grid, pdf_enstrophy)) in enumerate(enst_data.items()):
                if len(enstrophy_grid) == 0:
                    continue
                
                label_base = Path(filename).stem
                
                if resolve_line_style_func:
                    color, lw, dash = resolve_line_style_func(
                        filename, idx, colors_enst, ps_enst,
                        style_key="per_sim_style_comparison",
                        include_marker=False,
                        default_marker="circle"
                    )
                else:
                    color = colors_enst[idx % len(colors_enst)]
                    lw = line_width
                    dash = "solid"
                
                fig_enst.add_trace(go.Scatter(
                    x=enstrophy_grid,
                    y=pdf_enstrophy,
                    mode='lines',
                    name=label_base,
                    line=dict(color=color, width=lw, dash=dash),
                    hovertemplate=f"Ω = %{{x:.4f}}<br>PDF = %{{y:.4e}}<extra>{label_base}</extra>"
                ))
            
            x_label_enst = "Ω / ⟨Ω⟩" if normalize_pdf else "Ω"
            y_label_enst = "⟨Ω⟩ P(Ω / ⟨Ω⟩)" if normalize_pdf else "P(Ω)"
            layout_kwargs = dict(
                xaxis_title=x_label_enst,
                yaxis_title=y_label_enst,
                height=ps_enst.get("figure_height", 500) if ps_enst else 500,
                hovermode='x unified',
                legend=dict(x=1.02, y=1)
            )
            
            if ps_enst:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs = apply_axis_limits(layout_kwargs, ps_enst)
                layout_kwargs = apply_figure_size(layout_kwargs, ps_enst)
            
            fig_enst.update_layout(**layout_kwargs)
            
            if apply_plot_style_func and ps_enst:
                fig_enst = apply_plot_style_func(fig_enst, ps_enst)
            
            st.plotly_chart(
                fig_enst, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "enstrophy_pdf",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_enst, title="Enstrophy PDF", source_page="PDFs")
            
            if export_panel_func:
                export_panel_func(fig_enst, data_dir, "enstrophy_pdf")
    
    # Theory & Equations
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("### Vorticity Magnitude PDF")
        st.markdown("**Probability Density Function of vorticity magnitude:**")
        st.latex(r"P(|\omega|) = \frac{1}{N \Delta |\omega|} \sum_{i=1}^{N} \delta(|\omega| - |\omega_i|)")
        st.markdown("where $|\omega| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$ is the vorticity magnitude,")
        st.markdown("$\\omega = \\nabla \\times \\mathbf{u}$ is the vorticity vector, and $N$ is the total number of grid points.")
        
        st.markdown("### Enstrophy PDF")
        st.markdown("**Probability Density Function of enstrophy:**")
        st.latex(r"P(\Omega) = \frac{1}{N \Delta \Omega} \sum_{i=1}^{N} \delta(\Omega - \Omega_i)")
        st.markdown("where $\\Omega = |\\omega|^2 = \\omega_x^2 + \\omega_y^2 + \\omega_z^2$ is the enstrophy,")
        st.markdown("which represents the local rotational intensity of the flow.")
        st.markdown("Enstrophy is a key quantity in turbulence, related to energy dissipation and vortex dynamics.")

