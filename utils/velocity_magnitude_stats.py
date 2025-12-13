"""
Velocity Magnitude Statistical Analysis
Module for computing and visualizing velocity magnitude PDFs
"""

import streamlit as st
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


# ==========================================================
# Higher-order moments: Skewness and Kurtosis
# ==========================================================
def compute_skewness_kurtosis(data):
    """
    Compute skewness and kurtosis for a data array.
    
    Skewness: S = ⟨u'³⟩/⟨u'²⟩^(3/2)
    Kurtosis: K = ⟨u'⁴⟩/⟨u'²⟩²
    
    Args:
        data: 1D array of values (already flattened and cleaned)
        
    Returns:
        mean, rms, skewness, kurtosis (floats)
    """
    # Remove NaN/Inf
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Compute mean
    mean = np.mean(data_clean)
    
    # Compute fluctuating component: u' = u - ⟨u⟩
    u_prime = data_clean - mean
    
    # Compute moments
    u2_mean = np.mean(u_prime**2)  # ⟨u'²⟩
    u3_mean = np.mean(u_prime**3)  # ⟨u'³⟩
    u4_mean = np.mean(u_prime**4)  # ⟨u'⁴⟩
    
    # Compute RMS
    rms = np.sqrt(u2_mean) if u2_mean > 0 else 0.0
    
    # Compute Skewness: S = ⟨u'³⟩/⟨u'²⟩^(3/2)
    if u2_mean > 0:
        skewness = u3_mean / (u2_mean**(3/2))
    else:
        skewness = 0.0
    
    # Compute Kurtosis: K = ⟨u'⁴⟩/⟨u'²⟩²
    if u2_mean > 0:
        kurtosis = u4_mean / (u2_mean**2)
    else:
        kurtosis = 0.0
    
    return mean, rms, skewness, kurtosis


def compute_velocity_magnitude_pdf(velocity, bins=100, normalize=False):
    """
    Compute smooth Probability Density Function for velocity magnitude |u|
    Uses Kernel Density Estimation (KDE) to produce smooth curves
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        bins: Number of evaluation points for smooth curve
        normalize: If True, normalize by RMS (|u|/σ_|u|)
        
    Returns:
        u_mag_grid, pdf_u_mag (arrays) - smooth PDF curve
    """
    # Compute velocity magnitude: |u| = √(ux² + uy² + uz²)
    u_mag = np.sqrt(
        velocity[:, :, :, 0]**2 + 
        velocity[:, :, :, 1]**2 + 
        velocity[:, :, :, 2]**2
    )
    
    # Flatten and remove NaN/Inf
    u_mag_flat = u_mag.flatten()
    u_mag_flat = u_mag_flat[np.isfinite(u_mag_flat)]
    
    if len(u_mag_flat) == 0:
        return np.array([]), np.array([])
    
    # Normalize by RMS if requested (standard for velocity: |u|/σ_|u|)
    normalization_factor = 1.0
    if normalize:
        rms_u = np.sqrt(np.mean(u_mag_flat**2))
        if rms_u > 0:
            u_mag_flat = u_mag_flat / rms_u
            normalization_factor = rms_u
    
    # Determine range
    u_mag_min = u_mag_flat.min()
    u_mag_max = u_mag_flat.max()
    
    # Add padding for smooth evaluation at edges
    u_mag_range = u_mag_max - u_mag_min
    u_mag_min -= 0.1 * u_mag_range
    u_mag_max += 0.1 * u_mag_range
    
    # Create fine grid for smooth curve evaluation
    u_mag_grid = np.linspace(u_mag_min, u_mag_max, bins)
    
    # Compute KDE for smooth PDF curve
    try:
        kde = gaussian_kde(u_mag_flat)
        pdf_u_mag = kde(u_mag_grid)
    except:
        # Fallback to histogram if KDE fails
        counts, edges = np.histogram(u_mag_flat, bins=bins, range=(u_mag_min, u_mag_max), density=True)
        pdf_u_mag = counts
        u_mag_grid = (edges[:-1] + edges[1:]) / 2
    
    # Normalize Y-axis: multiply by normalization_factor to preserve area = 1
    if normalize and normalization_factor > 0:
        pdf_u_mag = pdf_u_mag * normalization_factor
    
    return u_mag_grid, pdf_u_mag


def compute_velocity_magnitude_statistics(velocity):
    """
    Compute statistical moments (mean, RMS, skewness, kurtosis) for velocity magnitude.
    
    Returns:
        mean, rms, skewness, kurtosis (floats)
    """
    # Compute velocity magnitude: |u| = √(ux² + uy² + uz²)
    u_mag = np.sqrt(
        velocity[:, :, :, 0]**2 + 
        velocity[:, :, :, 1]**2 + 
        velocity[:, :, :, 2]**2
    )
    
    # Flatten
    u_mag_flat = u_mag.flatten()
    
    # Compute statistics
    mean, rms, skewness, kurtosis = compute_skewness_kurtosis(u_mag_flat)
    
    return mean, rms, skewness, kurtosis


def compute_velocity_pdf(velocity, bins=100, normalize=False):
    """
    Compute smooth Probability Density Function for each velocity component (u, v, w)
    Uses Kernel Density Estimation (KDE) to produce smooth curves like reference figures
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        bins: Number of evaluation points for smooth curve
        normalize: If True, normalize by RMS (u/σ_u)
        
    Returns:
        u_grid, pdf_u, pdf_v, pdf_w (all arrays) - smooth PDF curves
    """
    # Extract each component
    ux = velocity[:, :, :, 0].flatten()
    uy = velocity[:, :, :, 1].flatten()
    uz = velocity[:, :, :, 2].flatten()
    
    # Remove NaN/Inf from each
    ux = ux[np.isfinite(ux)]
    uy = uy[np.isfinite(uy)]
    uz = uz[np.isfinite(uz)]
    
    if len(ux) == 0 or len(uy) == 0 or len(uz) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Normalize by RMS if requested (standard for velocity: u/σ_u)
    normalization_factor = 1.0
    if normalize:
        # Use combined RMS across all components
        all_u = np.concatenate([ux, uy, uz])
        rms_u = np.sqrt(np.mean(all_u**2))
        if rms_u > 0:
            ux = ux / rms_u
            uy = uy / rms_u
            uz = uz / rms_u
            normalization_factor = rms_u
    
    # Find common range across all components for consistent comparison
    u_min = min(ux.min(), uy.min(), uz.min())
    u_max = max(ux.max(), uy.max(), uz.max())
    
    # Add padding for smooth evaluation at edges
    u_range = u_max - u_min
    u_min -= 0.1 * u_range
    u_max += 0.1 * u_range
    
    # Create fine grid for smooth curve evaluation
    u_grid = np.linspace(u_min, u_max, bins)
    
    # Compute KDE for each component to get smooth PDF curves
    try:
        kde_u = gaussian_kde(ux)
        pdf_u = kde_u(u_grid)
    except:
        # Fallback to histogram if KDE fails
        counts, edges = np.histogram(ux, bins=bins, range=(u_min, u_max), density=True)
        pdf_u = counts
        u_grid = (edges[:-1] + edges[1:]) / 2
    
    try:
        kde_v = gaussian_kde(uy)
        pdf_v = kde_v(u_grid)
    except:
        counts, edges = np.histogram(uy, bins=bins, range=(u_min, u_max), density=True)
        pdf_v = counts
        if len(u_grid) != len(pdf_v):
            u_grid = (edges[:-1] + edges[1:]) / 2
    
    try:
        kde_w = gaussian_kde(uz)
        pdf_w = kde_w(u_grid)
    except:
        counts, edges = np.histogram(uz, bins=bins, range=(u_min, u_max), density=True)
        pdf_w = counts
        if len(u_grid) != len(pdf_w):
            u_grid = (edges[:-1] + edges[1:]) / 2
    
    # Normalize Y-axis: multiply by normalization_factor to preserve area = 1
    if normalize and normalization_factor > 0:
        pdf_u = pdf_u * normalization_factor
        pdf_v = pdf_v * normalization_factor
        pdf_w = pdf_w * normalization_factor
    
    return u_grid, pdf_u, pdf_v, pdf_w


def compute_velocity_component_statistics(velocity):
    """
    Compute statistical moments (mean, RMS, skewness, kurtosis) for each velocity component (u, v, w).
    
    Returns:
        Dictionary with keys 'u', 'v', 'w', each containing (mean, rms, skewness, kurtosis)
    """
    # Extract each component
    ux = velocity[:, :, :, 0].flatten()
    uy = velocity[:, :, :, 1].flatten()
    uz = velocity[:, :, :, 2].flatten()
    
    # Compute statistics for each component
    stats_u = compute_skewness_kurtosis(ux)
    stats_v = compute_skewness_kurtosis(uy)
    stats_w = compute_skewness_kurtosis(uz)
    
    return {
        'u': stats_u,
        'v': stats_v,
        'w': stats_w
    }


def display_statistics_table(statistics_dict, title="Statistical Moments"):
    """
    Display statistics in a formatted table.
    
    Args:
        statistics_dict: Dictionary with keys as variable names and values as (mean, rms, skewness, kurtosis) tuples
        title: Title for the statistics section
    """
    st.markdown(f"### {title}")
    st.markdown("**Higher-order moments:**")
    st.markdown("- **Skewness**: $S = \\langle u'^3 \\rangle / \\langle u'^2 \\rangle^{3/2}$ (asymmetry)")
    st.markdown("- **Kurtosis**: $K = \\langle u'^4 \\rangle / \\langle u'^2 \\rangle^2$ (tail heaviness)")
    st.markdown("")
    
    # Create table data
    table_data = []
    for var_name, (mean, rms, skewness, kurtosis) in statistics_dict.items():
        table_data.append({
            "Variable": var_name.upper(),
            "Mean": f"{mean:.6e}",
            "RMS": f"{rms:.6e}",
            "Skewness": f"{skewness:.4f}",
            "Kurtosis": f"{kurtosis:.4f}"
        })
    
    if table_data:
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', hide_index=True)


def render_velocity_magnitude_tab(data_dir_or_dirs, load_velocity_file_func,
                                   get_plot_style_func=None, apply_plot_style_func=None,
                                   get_palette_func=None, resolve_line_style_func=None,
                                   export_panel_func=None, capture_button_func=None):
    """
    Render the Velocity Magnitude PDF tab content
    
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
    
    st.header("Velocity Magnitude PDF")
    st.markdown("Compare velocity component PDFs and velocity magnitude PDFs across different simulations/methods.")
    
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
    
    selected_files_pdf = st.sidebar.multiselect(
        "Velocity PDF files:",
        options=[Path(f).name for f in all_files],
        default=[Path(f).name for f in all_files[:min(2, len(all_files))]],
        help="Select files for Velocity PDF plot (left)",
        key="velocity_pdf_file_select"
    )
    
    selected_files_mag = st.sidebar.multiselect(
        "Velocity Magnitude PDF files:",
        options=[Path(f).name for f in all_files],
        default=[Path(f).name for f in all_files[:min(2, len(all_files))]],
        help="Select files for Velocity Magnitude PDF plot (right)",
        key="velocity_mag_file_select"
    )
    
    if not selected_files_pdf and not selected_files_mag:
        st.warning("Please select at least one file for either plot.")
        return
    
    # Plot parameters
    st.sidebar.header("📊 Plot Parameters")
    pdf_bins = st.sidebar.slider("Velocity PDF bins", 50, 500, 100, 10, key="velocity_pdf_bins")
    mag_bins = st.sidebar.slider("Velocity Magnitude PDF bins", 50, 500, 100, 10, key="velocity_mag_pdf_bins")
    normalize_pdf = st.sidebar.checkbox(
        "Normalize Velocity PDF (u/σ_u)",
        value=False,
        help="Normalize velocity components by RMS for comparison with literature",
        key="velocity_pdf_normalize"
    )
    normalize_mag = st.sidebar.checkbox(
        "Normalize by RMS (|u|/σ_|u|)",
        value=False,
        help="Normalize velocity magnitude by RMS for comparison with literature",
        key="velocity_mag_normalize"
    )
    
    # Load and compute data independently for each plot
    pdf_data = {}
    mag_data = {}
    
    # Load data for Velocity PDF plot
    if selected_files_pdf:
        for filename in selected_files_pdf:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for PDF..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"⚠️ {filename}: Invalid velocity shape")
                        continue
                    
                    # Compute PDF for each component
                    u_bins, pdf_u, pdf_v, pdf_w = compute_velocity_pdf(velocity, bins=pdf_bins, normalize=normalize_pdf)
                    pdf_data[filename] = (u_bins, pdf_u, pdf_v, pdf_w)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for PDF: {e}")
                continue
    
    # Load data for Velocity Magnitude PDF plot
    if selected_files_mag:
        for filename in selected_files_mag:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for Magnitude PDF..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"⚠️ {filename}: Invalid velocity shape")
                        continue
                    
                    # Compute PDF
                    u_mag_grid, pdf_u_mag = compute_velocity_magnitude_pdf(velocity, bins=mag_bins, normalize=normalize_mag)
                    mag_data[filename] = (u_mag_grid, pdf_u_mag)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for Magnitude PDF: {e}")
                continue
    
    if not pdf_data and not mag_data:
        st.error("No valid velocity data loaded.")
        return
    
    # ============================================
    # Statistics Section
    # ============================================
    with st.expander("📈 Statistical Moments (Skewness & Kurtosis)", expanded=False):
        # Compute statistics from first available file
        stats_dict = {}
        if selected_files_pdf or selected_files_mag:
            first_file = (selected_files_pdf + selected_files_mag)[0] if (selected_files_pdf or selected_files_mag) else None
            if first_file:
                filepath = data_dir / first_file
                try:
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    if velocity is not None and len(velocity.shape) == 4:
                        # Velocity components statistics
                        comp_stats = compute_velocity_component_statistics(velocity)
                        stats_dict['u'] = comp_stats['u']
                        stats_dict['v'] = comp_stats['v']
                        stats_dict['w'] = comp_stats['w']
                        
                        # Velocity magnitude statistics
                        mean, rms, skew, kurt = compute_velocity_magnitude_statistics(velocity)
                        stats_dict['|u|'] = (mean, rms, skew, kurt)
                except:
                    pass
        
        if stats_dict:
            display_statistics_table(stats_dict, title="Velocity Statistics")
        else:
            st.info("Statistics will be computed when files are loaded.")
    
    st.markdown("---")
    
    # Create side-by-side plots
    col1, col2 = st.columns(2)
    
    # ============================================
    # Left: Velocity PDF
    # ============================================
    with col1:
        st.subheader("Velocity PDF")
        
        if not pdf_data:
            st.info("Select files in the sidebar to plot Velocity PDF")
        else:
            plot_name_pdf = "Velocity PDF"
            ps_pdf = get_plot_style_func(plot_name_pdf) if get_plot_style_func else {}
            colors_pdf = get_palette_func(ps_pdf) if get_palette_func else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            component_colors = ['#1f77b4', '#2ca02c', '#d62728']
            line_width = ps_pdf.get("line_width", 2.4) if ps_pdf else 2.0
            
            fig_pdf = go.Figure()
            
            for idx, (filename, (u_bins, pdf_u, pdf_v, pdf_w)) in enumerate(pdf_data.items()):
                if len(u_bins) == 0:
                    continue
                
                label_base = Path(filename).stem
                
                if resolve_line_style_func:
                    color_u, lw_u, dash_u = resolve_line_style_func(
                        filename, idx, colors_pdf, ps_pdf,
                        style_key="per_sim_style_comparison",
                        include_marker=False,
                        default_marker="circle"
                    )
                else:
                    color_u = colors_pdf[idx % len(colors_pdf)]
                    lw_u = line_width
                    dash_u = "solid"
                
                fig_pdf.add_trace(go.Scatter(
                    x=u_bins,
                    y=pdf_u,
                    mode='lines',
                    name=f"{label_base} - u",
                    line=dict(color=color_u, width=lw_u, dash=dash_u),
                    hovertemplate=f"u = %{{x:.4f}}<br>PDF = %{{y:.4e}}<extra>{label_base} - u</extra>"
                ))
                
                if resolve_line_style_func:
                    color_v, lw_v, dash_v = resolve_line_style_func(
                        filename, idx, colors_pdf, ps_pdf,
                        style_key="per_sim_style_comparison",
                        include_marker=False,
                        default_marker="circle"
                    )
                else:
                    color_v = component_colors[1]
                    lw_v = line_width
                    dash_v = "solid"
                
                fig_pdf.add_trace(go.Scatter(
                    x=u_bins,
                    y=pdf_v,
                    mode='lines',
                    name=f"{label_base} - v",
                    line=dict(color=color_v, width=lw_v, dash=dash_v),
                    hovertemplate=f"v = %{{x:.4f}}<br>PDF = %{{y:.4e}}<extra>{label_base} - v</extra>"
                ))
                
                if resolve_line_style_func:
                    color_w, lw_w, dash_w = resolve_line_style_func(
                        filename, idx, colors_pdf, ps_pdf,
                        style_key="per_sim_style_comparison",
                        include_marker=False,
                        default_marker="circle"
                    )
                else:
                    color_w = component_colors[2]
                    lw_w = line_width
                    dash_w = "solid"
                
                fig_pdf.add_trace(go.Scatter(
                    x=u_bins,
                    y=pdf_w,
                    mode='lines',
                    name=f"{label_base} - w",
                    line=dict(color=color_w, width=lw_w, dash=dash_w),
                    hovertemplate=f"w = %{{x:.4f}}<br>PDF = %{{y:.4e}}<extra>{label_base} - w</extra>"
                ))
            
            x_label_vel = "u / σ<sub>u</sub>" if normalize_pdf else "u"
            y_label_vel = "σ<sub>u</sub> P(u / σ<sub>u</sub>)" if normalize_pdf else "P(u)"
            layout_kwargs = dict(
                xaxis_title=x_label_vel,
                yaxis_title=y_label_vel,
                height=ps_pdf.get("figure_height", 500) if ps_pdf else 500,
                hovermode='x unified',
                legend=dict(x=1.02, y=1)
            )
            
            if ps_pdf:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs = apply_axis_limits(layout_kwargs, ps_pdf)
                layout_kwargs = apply_figure_size(layout_kwargs, ps_pdf)
            
            fig_pdf.update_layout(**layout_kwargs)
            
            if apply_plot_style_func and ps_pdf:
                fig_pdf = apply_plot_style_func(fig_pdf, ps_pdf)
            
            st.plotly_chart(
                fig_pdf, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "velocity_pdf",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_pdf, title="Velocity PDF", source_page="PDFs")
    
            if export_panel_func:
                export_panel_func(fig_pdf, data_dir, "velocity_pdf")
    
    # ============================================
    # Right: Velocity Magnitude PDF
    # ============================================
    with col2:
        st.subheader("Velocity Magnitude PDF")
        
        if not mag_data:
            st.info("Select files in the sidebar to plot Velocity Magnitude PDF")
        else:
            plot_name_mag = "Velocity Magnitude PDF"
            ps_mag = get_plot_style_func(plot_name_mag) if get_plot_style_func else {}
            colors_mag = get_palette_func(ps_mag) if get_palette_func else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            line_width = ps_mag.get("line_width", 2.4) if ps_mag else 2.0
    
            fig_mag = go.Figure()
    
            for idx, (filename, (u_mag_grid, pdf_u_mag)) in enumerate(mag_data.items()):
                if len(u_mag_grid) == 0:
                    continue
                
                label_base = Path(filename).stem
                
                if resolve_line_style_func:
                    color, lw, dash = resolve_line_style_func(
                        filename, idx, colors_mag, ps_mag,
                        style_key="per_sim_style_comparison",
                        include_marker=False,
                        default_marker="circle"
                    )
                else:
                    color = colors_mag[idx % len(colors_mag)]
                    lw = line_width
                    dash = "solid"
                
                fig_mag.add_trace(go.Scatter(
                    x=u_mag_grid,
                    y=pdf_u_mag,
                    mode='lines',
                    name=label_base,
                    line=dict(color=color, width=lw, dash=dash),
                    hovertemplate=f"|u| = %{{x:.4f}}<br>PDF = %{{y:.4e}}<extra>{label_base}</extra>"
                ))
            
            x_label = "|u| / σ<sub>|u|</sub>" if normalize_mag else "|u|"
            y_label = "σ<sub>|u|</sub> P(|u| / σ<sub>|u|</sub>)" if normalize_mag else "P(|u|)"
            layout_kwargs = dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=ps_mag.get("figure_height", 500) if ps_mag else 500,
                hovermode='x unified',
                legend=dict(x=1.02, y=1)
            )
            
            if ps_mag:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs = apply_axis_limits(layout_kwargs, ps_mag)
                layout_kwargs = apply_figure_size(layout_kwargs, ps_mag)
            
            fig_mag.update_layout(**layout_kwargs)
            
            if apply_plot_style_func and ps_mag:
                fig_mag = apply_plot_style_func(fig_mag, ps_mag)
            
            st.plotly_chart(
                fig_mag, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "velocity_magnitude_pdf",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_mag, title="Velocity Magnitude PDF", source_page="PDFs")
            
            if export_panel_func:
                export_panel_func(fig_mag, data_dir, "velocity_magnitude_pdf")
    
    # Theory & Equations
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("### Velocity PDF")
        st.markdown("**Probability Density Function of velocity:**")
        st.latex(r"P(u) = \frac{1}{N \Delta u} \sum_{i=1}^{N} \delta(u - u_i)")
        st.markdown("where $N$ is the total number of grid points and $\\Delta u$ is the bin width.")
        st.markdown("The semi-logarithmic scale reveals the tails of the distribution, showing rare high-velocity events.")
        
        st.markdown("### Velocity Magnitude PDF")
        st.markdown("**Probability Density Function of velocity magnitude:**")
        st.latex(r"P(|\mathbf{u}|) = \frac{1}{N \Delta |\mathbf{u}|} \sum_{i=1}^{N} \delta(|\mathbf{u}| - |\mathbf{u}_i|)")
        st.markdown("where $|\\mathbf{u}| = \\sqrt{u_x^2 + u_y^2 + u_z^2}$ is the velocity magnitude,")
        st.markdown("and $N$ is the total number of grid points.")
        st.markdown("The velocity magnitude PDF provides information about the overall speed distribution")
        st.markdown("in the flow, complementing the component-wise velocity PDFs.")

