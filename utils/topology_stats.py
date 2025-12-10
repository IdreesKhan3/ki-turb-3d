"""
Topological & Statistical Distribution Analysis
Module for computing and visualizing velocity PDFs and R-Q topological space
"""

import streamlit as st
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from utils.iso_surfaces import compute_q_invariant, compute_r_invariant


def compute_velocity_pdf(velocity, bins=100):
    """
    Compute smooth Probability Density Function for each velocity component (u, v, w)
    Uses Kernel Density Estimation (KDE) to produce smooth curves like reference figures
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        bins: Number of evaluation points for smooth curve
        
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
    
    return u_grid, pdf_u, pdf_v, pdf_w


def compute_rq_joint_pdf(velocity, r_bins=100, q_bins=100, r_range=None, q_range=None):
    """
    Compute Joint PDF of Q and R invariants
    
    Q and R are normalized by <S_ij S_ij> (mean strain rate squared) to match literature conventions.
    This normalization brings Q values to typical ranges of -30 to 30 as seen in turbulence literature.
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        r_bins: Number of bins for R axis
        q_bins: Number of bins for Q axis
        r_range: (r_min, r_max) for R axis, or None for auto
        q_range: (q_min, q_max) for Q axis, or None for auto
        
    Returns:
        R_centers, Q_centers, joint_pdf (2D array)
    """
    # Compute Q and R invariants
    Q = compute_q_invariant(velocity)
    R = compute_r_invariant(velocity)
    
    # Compute normalization factor: <S_ij S_ij> (mean strain rate squared)
    from utils.iso_surfaces import compute_rotation_deformation_tensors
    _, S = compute_rotation_deformation_tensors(velocity)
    S_squared_sum = np.einsum('ijklm,ijklm->ijk', S, S)
    # Use all finite values for mean (S_ij S_ij is always non-negative)
    valid_S = S_squared_sum[np.isfinite(S_squared_sum)]
    mean_S_squared = np.mean(valid_S) if len(valid_S) > 0 else 1.0
    
    # Normalize Q and R by <S_ij S_ij>
    # Q* = Q / <S_ij S_ij>
    # R* = R / <S_ij S_ij>^(3/2)
    # This normalization brings values to typical literature ranges (-30 to 30 for Q)
    if mean_S_squared > 0:
        Q_normalized = Q / mean_S_squared
        R_normalized = R / (mean_S_squared ** 1.5)
    else:
        # Fallback: use raw values if normalization fails
        Q_normalized = Q
        R_normalized = R
    
    # Flatten
    Q_flat = Q_normalized.flatten()
    R_flat = R_normalized.flatten()
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(Q_flat) & np.isfinite(R_flat)
    Q_flat = Q_flat[valid_mask]
    R_flat = R_flat[valid_mask]
    
    if len(Q_flat) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Determine ranges
    if r_range is None:
        r_range = (R_flat.min(), R_flat.max())
    if q_range is None:
        q_range = (Q_flat.min(), Q_flat.max())
    
    # Compute 2D histogram
    joint_hist, r_edges, q_edges = np.histogram2d(
        R_flat, Q_flat,
        bins=[r_bins, q_bins],
        range=[r_range, q_range],
        density=False
    )
    
    # Normalize to joint PDF
    bin_area = (r_edges[1] - r_edges[0]) * (q_edges[1] - q_edges[0])
    joint_pdf = joint_hist / (len(R_flat) * bin_area)
    
    # Bin centers
    R_centers = (r_edges[:-1] + r_edges[1:]) / 2
    Q_centers = (q_edges[:-1] + q_edges[1:]) / 2
    
    return R_centers, Q_centers, joint_pdf.T  # Transpose for correct orientation


def compute_discriminant_line(r_values):
    """
    Compute Q values for the zero-discriminant line D = (Q/3)^3 + (R/2)^2 = 0
    
    The discriminant D = 0 separates regions with real eigenvalues (inside V) 
    from complex eigenvalues (outside V). Solving for Q:
    (Q/3)^3 = -(R/2)^2
    Q = -3 * (R^2/4)^(1/3)
    
    Q is always negative on this boundary, regardless of R's sign.
    This gives a V-shape (cusp) with vertex at (0,0) extending downward.
    
    Args:
        r_values: Array of R values
        
    Returns:
        Q values for the discriminant line (always negative, forming V-shape)
    """
    q_values = -3 * np.power(np.abs(r_values) / 2.0, 2.0/3.0)
    return q_values


def render_topology_stats_tab(data_dir, load_velocity_file_func,
                               get_plot_style_func=None, apply_plot_style_func=None,
                               get_palette_func=None, resolve_line_style_func=None,
                               export_panel_func=None, capture_button_func=None):
    """
    Render the Topological & Statistical Distribution tab content
    
    Args:
        data_dir: Path to data directory
        load_velocity_file_func: Function to load velocity files (takes filepath)
        get_plot_style_func: Optional function to get plot style (plot_name) -> style_dict
        apply_plot_style_func: Optional function to apply plot style (fig, style_dict) -> fig
        get_palette_func: Optional function to get color palette (style_dict) -> color_list
        resolve_line_style_func: Optional function to resolve line style for files
        export_panel_func: Optional function to show export panel (fig, out_dir, base_name)
        capture_button_func: Optional function to add capture button (fig, title, source_page)
    """
    import glob
    from utils.file_detector import natural_sort_key
    
    st.header("Topological & Statistical Distribution")
    st.markdown("Compare velocity PDFs and R-Q topological space across different simulations/methods.")
    
    # Find velocity files
    vti_files = sorted(
        glob.glob(str(data_dir / "*.vti")) + 
        glob.glob(str(data_dir / "*.VTI")),
        key=natural_sort_key
    )
    hdf5_files = sorted(
        glob.glob(str(data_dir / "*.h5")) + 
        glob.glob(str(data_dir / "*.H5")) +
        glob.glob(str(data_dir / "*.hdf5")) + 
        glob.glob(str(data_dir / "*.HDF5")),
        key=natural_sort_key
    )
    
    all_files = vti_files + hdf5_files
    
    if not all_files:
        st.error("No velocity files found. Expected: `*.vti`, `*.h5`, or `*.hdf5`")
        return
    
    # File selection - independent for each plot
    st.sidebar.header("📁 File Selection")
    st.sidebar.caption(f"Found {len(all_files)} velocity files")
    
    # Separate file selection for each plot
    selected_files_pdf = st.sidebar.multiselect(
        "Velocity PDF files:",
        options=[Path(f).name for f in all_files],
        default=[Path(f).name for f in all_files[:min(2, len(all_files))]],
        help="Select files for Velocity PDF plot (left)"
    )
    
    selected_files_rq = st.sidebar.multiselect(
        "R-Q Topological Space files:",
        options=[Path(f).name for f in all_files],
        default=[Path(f).name for f in all_files[:min(3, len(all_files))]],
        help="Select files for R-Q plot (right)"
    )
    
    if not selected_files_pdf and not selected_files_rq:
        st.warning("Please select at least one file for either plot.")
        return
    
    
    # Plot parameters
    st.sidebar.header("📊 Plot Parameters")
    pdf_bins = st.sidebar.slider("Velocity PDF bins", 50, 500, 100, 10)
    rq_bins = st.sidebar.slider("R-Q space bins", 50, 200, 100, 10)
    use_log_scale = st.sidebar.checkbox(
        "Log scale (R-Q PDF)",
        value=True,
        help="Use logarithmic scale to visualize the tear-drop shape (Vieillefosse tail)"
    )
    
    # Load and compute data independently for each plot
    pdf_data = {}
    rq_data = {}
    
    # Load data for Velocity PDF plot
    if selected_files_pdf:
        for filename in selected_files_pdf:
            filepath = data_dir / filename
            try:
                with st.spinner(f"Loading {filename} for PDF..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"⚠️ {filename}: Invalid velocity shape")
                        continue
                    
                    # Compute PDF for each component
                    u_bins, pdf_u, pdf_v, pdf_w = compute_velocity_pdf(velocity, bins=pdf_bins)
                    pdf_data[filename] = (u_bins, pdf_u, pdf_v, pdf_w)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for PDF: {e}")
                continue
    
    # Load data for R-Q plot
    if selected_files_rq:
        for filename in selected_files_rq:
            filepath = data_dir / filename
            try:
                with st.spinner(f"Loading {filename} for R-Q..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"⚠️ {filename}: Invalid velocity shape")
                        continue
                    
                    # Compute R-Q joint PDF
                    R_centers, Q_centers, joint_pdf = compute_rq_joint_pdf(velocity, r_bins=rq_bins, q_bins=rq_bins)
                    rq_data[filename] = (R_centers, Q_centers, joint_pdf)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for R-Q: {e}")
                continue
    
    if not pdf_data and not rq_data:
        st.error("No valid velocity data loaded.")
        return
    
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
            
            layout_kwargs = dict(
                xaxis_title="Velocity",
                yaxis_title="PDF",
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
                capture_button_func(fig_pdf, title="Velocity PDF", source_page="Comparison")
            
            if export_panel_func:
                export_panel_func(fig_pdf, data_dir, "velocity_pdf")
    
    # ============================================
    # Right: R-Q Topological Space
    # ============================================
    with col2:
        st.subheader("R-Q Topological Space")
        
        if not rq_data:
            st.info("Select files in the sidebar to plot R-Q Topological Space")
        else:
            # Determine common range for comparison
            all_R = []
            all_Q = []
            for R_centers, Q_centers, _ in rq_data.values():
                if len(R_centers) > 0 and len(Q_centers) > 0:
                    all_R.extend([R_centers.min(), R_centers.max()])
                    all_Q.extend([Q_centers.min(), Q_centers.max()])
            
            if all_R and all_Q:
                r_range = (min(all_R), max(all_R))
                q_range = (min(all_Q), max(all_Q))
            else:
                r_range = None
                q_range = None
            
            # Create subplot for multiple datasets
            fig_rq = go.Figure()
            
            # Plot each dataset
            for idx, (filename, (R_centers, Q_centers, joint_pdf)) in enumerate(rq_data.items()):
                if len(R_centers) == 0 or len(Q_centers) == 0 or joint_pdf.size == 0:
                    continue
                
                label = Path(filename).stem
                
                # Apply log scaling if requested (essential for seeing tear-drop shape)
                if use_log_scale:
                    safe_pdf = joint_pdf.copy()
                    safe_pdf[safe_pdf <= 0] = np.nan
                    plot_data = np.log10(safe_pdf)
                    z_label = "log₁₀(PDF)"
                    colorscale = 'Jet'
                else:
                    plot_data = joint_pdf
                    z_label = "PDF"
                    colorscale = 'Viridis'
                
                # Create contour plot
                fig_rq.add_trace(go.Contour(
                    x=R_centers,
                    y=Q_centers,
                    z=plot_data,
                    name=label,
                    colorscale=colorscale,
                    showscale=(idx == 0),
                    colorbar=dict(title=z_label) if idx == 0 else None,
                    hovertemplate=f"R = %{{x:.4f}}<br>Q = %{{y:.4f}}<br>{z_label} = %{{z:.2f}}<extra>{label}</extra>"
                ))
            
            # Add discriminant line (V-shaped black line)
            if r_range:
                # Use more points for smooth V-shape
                r_line = np.linspace(r_range[0], r_range[1], 1000)
                q_line = compute_discriminant_line(r_line)
                fig_rq.add_trace(go.Scatter(
                    x=r_line,
                    y=q_line,
                    mode='lines',
                    name='D = 0',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hovertemplate="R = %{x:.4f}<br>Q = %{y:.4f}<extra>Discriminant</extra>"
                ))
            
            plot_name_rq = "R-Q Topological Space"
            ps_rq = get_plot_style_func(plot_name_rq) if get_plot_style_func else {}
            
            layout_kwargs_rq = dict(
                xaxis_title="R",
                yaxis_title="Q",
                height=ps_rq.get("figure_height", 500) if ps_rq else 500,
                hovermode='closest',
                legend=dict(x=1.02, y=1)
            )
            
            if ps_rq:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs_rq = apply_axis_limits(layout_kwargs_rq, ps_rq)
                layout_kwargs_rq = apply_figure_size(layout_kwargs_rq, ps_rq)
            
            fig_rq.update_layout(**layout_kwargs_rq)
            
            if apply_plot_style_func and ps_rq:
                fig_rq = apply_plot_style_func(fig_rq, ps_rq)
            
            st.plotly_chart(
                fig_rq, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "rq_topological_space",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_rq, title="R-Q Topological Space", source_page="Comparison")
            
            if export_panel_func:
                export_panel_func(fig_rq, data_dir, "rq_topological_space")
    
    # Theory & Equations
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("### Velocity PDF")
        st.markdown("**Probability Density Function of velocity:**")
        st.latex(r"P(u) = \frac{1}{N \Delta u} \sum_{i=1}^{N} \delta(u - u_i)")
        st.markdown("where $N$ is the total number of grid points and $\\Delta u$ is the bin width.")
        st.markdown("The semi-logarithmic scale reveals the tails of the distribution, showing rare high-velocity events.")
        
        st.markdown("### R-Q Topological Space")
        st.markdown("**Joint PDF of velocity gradient tensor invariants:**")
        st.markdown("**Second Invariant Q:**")
        st.latex(r"Q = \frac{1}{4}(\omega_i\omega_i - 2S_{ij}S_{ij})")
        st.markdown("**Third Invariant R:**")
        st.latex(r"R = -\frac{1}{3}\left(S_{ij}S_{jk}S_{ki} + \frac{3}{4}\omega_i\omega_j S_{ij}\right)")
        st.markdown("**Zero Discriminant Line:**")
        st.latex(r"D = \left(\frac{Q}{3}\right)^3 + \left(\frac{R}{2}\right)^2 = 0")
        st.latex(r"Q = -3\left(\frac{R}{2}\right)^{2/3}")
        st.markdown("This line separates regions with real eigenvalues (above) from complex eigenvalues (below) of the velocity gradient tensor.")
        st.markdown("**Visualization:** The logarithmic scale is essential to visualize the **Vieillefosse tail** (tear-drop shape) in the bottom-right quadrant ($R>0, Q<0$), where probability densities are orders of magnitude lower than the core region.")

