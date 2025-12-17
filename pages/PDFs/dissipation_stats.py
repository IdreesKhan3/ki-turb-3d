"""
Dissipation Rate Statistical Analysis
Module for computing and visualizing dissipation rate PDFs
"""

import streamlit as st
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from utils.iso_surfaces import compute_rotation_deformation_tensors
from data_readers.parameter_reader import read_parameters


def compute_dissipation_pdf(velocity, nu=1.0, bins=100, dx=1.0, dy=1.0, dz=1.0, normalize=False):
    """
    Compute smooth Probability Density Function for dissipation rate ε = 2ν S_ij S_ij
    Uses Kernel Density Estimation (KDE) to produce smooth curves
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        nu: Kinematic viscosity (default 1.0)
        bins: Number of evaluation points for smooth curve
        dx, dy, dz: Grid spacing (default 1.0)
        normalize: If True, normalize by mean dissipation (ε/⟨ε⟩)
        
    Returns:
        eps_grid, pdf_eps (arrays) - smooth PDF curve
    """
    # Compute strain rate tensor S_ij
    _, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    
    # Compute dissipation: ε = 2ν S_ij S_ij
    # S_ij S_ij is the double contraction (sum over i and j)
    S_squared_sum = np.einsum('ijklm,ijklm->ijk', S, S)
    dissipation = 2.0 * nu * S_squared_sum
    
    # Flatten and remove NaN/Inf
    eps_flat = dissipation.flatten()
    eps_flat = eps_flat[np.isfinite(eps_flat)]
    
    # Remove negative values (dissipation should be non-negative)
    eps_flat = eps_flat[eps_flat >= 0]
    
    if len(eps_flat) == 0:
        return np.array([]), np.array([])
    
    # Normalize by mean if requested (standard for dissipation: ε/⟨ε⟩)
    normalization_factor = 1.0
    if normalize:
        mean_eps = np.mean(eps_flat)
        if mean_eps > 0:
            eps_flat = eps_flat / mean_eps
            normalization_factor = mean_eps
    
    # Determine range
    eps_min = eps_flat.min()
    eps_max = eps_flat.max()
    
    # Add padding for smooth evaluation at edges
    eps_range = eps_max - eps_min
    if eps_range > 0:
        eps_min -= 0.1 * eps_range
        eps_max += 0.1 * eps_range
    else:
        eps_min = max(0, eps_min - 0.01)
        eps_max = eps_max + 0.01
    
    # Create fine grid for smooth curve evaluation
    eps_grid = np.linspace(eps_min, eps_max, bins)
    
    # Compute KDE for smooth PDF curve
    try:
        kde = gaussian_kde(eps_flat)
        pdf_eps = kde(eps_grid)
    except:
        # Fallback to histogram if KDE fails
        counts, edges = np.histogram(eps_flat, bins=bins, range=(eps_min, eps_max), density=True)
        pdf_eps = counts
        eps_grid = (edges[:-1] + edges[1:]) / 2
    
    # Normalize Y-axis: multiply by normalization_factor to preserve area = 1
    if normalize and normalization_factor > 0:
        pdf_eps = pdf_eps * normalization_factor
    
    return eps_grid, pdf_eps


def compute_dissipation_statistics(velocity, nu=1.0, dx=1.0, dy=1.0, dz=1.0):
    """
    Compute statistical moments (mean, RMS, skewness, kurtosis) for dissipation rate.
    
    Returns:
        mean, rms, skewness, kurtosis (floats)
    """
    from utils.iso_surfaces import compute_rotation_deformation_tensors
    from .velocity_magnitude_stats import compute_skewness_kurtosis
    
    # Compute strain rate tensor S_ij
    _, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    
    # Compute dissipation: ε = 2ν S_ij S_ij
    S_squared_sum = np.einsum('ijklm,ijklm->ijk', S, S)
    dissipation = 2.0 * nu * S_squared_sum
    
    # Flatten and remove NaN/Inf and negative values
    eps_flat = dissipation.flatten()
    eps_flat = eps_flat[np.isfinite(eps_flat)]
    eps_flat = eps_flat[eps_flat >= 0]
    
    # Compute statistics
    mean, rms, skewness, kurtosis = compute_skewness_kurtosis(eps_flat)
    
    return mean, rms, skewness, kurtosis


def render_dissipation_tab(data_dir_or_dirs, load_velocity_file_func,
                            get_plot_style_func=None, apply_plot_style_func=None,
                            get_palette_func=None, resolve_line_style_func=None,
                            export_panel_func=None, capture_button_func=None):
    """
    Render the Dissipation Rate PDF tab content
    
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
    
    st.header("Dissipation Rate PDF")
    st.markdown("Compare dissipation rate PDFs across different simulations/methods.")
    
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
    
    # Physical parameters (show first, always visible)
    st.sidebar.header("⚙️ Physical Parameters")
    
    # Always try to read viscosity from parameter file first
    param_file = data_dir / "simulation.input"
    nu_from_file = None
    if param_file.exists():
        try:
            params = read_parameters(str(param_file))
            if 'nu' in params:
                nu_from_file = params['nu']
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error reading simulation.input: {e}")
    
    # Set default value: use file value if available, otherwise use a reasonable default
    default_nu = nu_from_file if nu_from_file is not None else 0.004
    
    # Show status message
    if nu_from_file is not None:
        st.sidebar.info(f"📄 Viscosity from simulation.input: {nu_from_file:.6f}")
    else:
        st.sidebar.warning("⚠️ Viscosity not found in simulation.input. Please enter manually or check parameter file.")
    
    nu_help = "Kinematic viscosity used in dissipation calculation: ε = 2ν S_ij S_ij"
    if nu_from_file is not None:
        nu_help += f" (loaded from simulation.input, can be overridden)"
    else:
        nu_help += " (enter manually)"
    
    # User can always override manually
    nu = st.sidebar.number_input(
        "ν (Kinematic Viscosity)",
        value=default_nu,
        min_value=0.0001,
        step=0.0001,
        format="%.6f",
        help=nu_help,
        key="dissipation_nu_input"
    )
    
    if not all_files:
        st.error("No velocity files found. Expected: `*.vti`, `*.h5`, or `*.hdf5`")
        return
    
    # Create mapping from filename to full path (handle files from different directories)
    filename_to_path = {Path(f).name: f for f in all_files}
    
    # File selection
    st.sidebar.header("📁 File Selection")
    st.sidebar.caption(f"Found {len(all_files)} velocity files")
    
    selected_files = st.sidebar.multiselect(
        "Dissipation PDF files:",
        options=[Path(f).name for f in all_files],
        default=[Path(f).name for f in all_files[:min(3, len(all_files))]],
        help="Select files for Dissipation Rate PDF plot",
        key="dissipation_file_select"
    )
    
    if not selected_files:
        st.warning("Please select at least one file.")
        return
    
    # Plot parameters
    st.sidebar.header("📊 Plot Parameters")
    pdf_bins = st.sidebar.slider("PDF bins", 50, 500, 100, 10, key="dissipation_pdf_bins")
    normalize_pdf = st.sidebar.checkbox(
        "Normalize by mean (ε/⟨ε⟩)",
        value=False,
        help="Normalize dissipation by mean value for comparison with literature",
        key="dissipation_normalize"
    )
    
    # Load and compute data
    pdf_data = {}
    
    for filename in selected_files:
        # Use full path from mapping (handles files from different directories)
        filepath = filename_to_path.get(filename)
        if not filepath:
            st.warning(f"File not found: {filename}")
            continue
        try:
            with st.spinner(f"Loading {filename}..."):
                vti_data = load_velocity_file_func(str(filepath))
                velocity = vti_data['velocity']
                
                if velocity is None or len(velocity.shape) != 4:
                    st.warning(f"⚠️ {filename}: Invalid velocity shape")
                    continue
                
                # Try to get viscosity from metadata, parameter file, or use sidebar value
                metadata = vti_data.get('metadata', {})
                file_nu = metadata.get('nu', metadata.get('viscosity', None))
                if file_nu is None:
                    # Try parameter file
                    if param_file.exists():
                        try:
                            params = read_parameters(str(param_file))
                            file_nu = params.get('nu', nu)
                        except:
                            file_nu = nu
                    else:
                        file_nu = nu
                
                # Compute PDF
                eps_grid, pdf_eps = compute_dissipation_pdf(
                    velocity, 
                    nu=file_nu, 
                    bins=pdf_bins, 
                    dx=1.0, 
                    dy=1.0, 
                    dz=1.0,
                    normalize=normalize_pdf
                )
                pdf_data[filename] = (eps_grid, pdf_eps)
                
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            continue
    
    if not pdf_data:
        st.error("No valid velocity data loaded.")
        return
    
    # ============================================
    # Statistics Section
    # ============================================
    with st.expander("📈 Statistical Moments (Skewness & Kurtosis)", expanded=False):
        from .velocity_magnitude_stats import display_statistics_table
        
        # Compute statistics from first available file
        stats_dict = {}
        if selected_files:
            first_file = selected_files[0]
            filepath = data_dir / first_file
            try:
                vti_data = load_velocity_file_func(str(filepath))
                velocity = vti_data['velocity']
                if velocity is not None and len(velocity.shape) == 4:
                    mean, rms, skew, kurt = compute_dissipation_statistics(velocity, nu=nu, dx=1.0, dy=1.0, dz=1.0)
                    stats_dict['dissipation'] = (mean, rms, skew, kurt)
            except:
                pass
        
        if stats_dict:
            display_statistics_table(stats_dict, title="Dissipation Rate Statistics")
        else:
            st.info("Statistics will be computed when files are loaded.")
    
    st.markdown("---")
    
    # Create plot
    st.subheader("Dissipation Rate PDF")
    
    plot_name = "Dissipation PDF"
    ps = get_plot_style_func(plot_name) if get_plot_style_func else {}
    colors = get_palette_func(ps) if get_palette_func else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_width = ps.get("line_width", 2.4) if ps else 2.0
    
    fig = go.Figure()
    
    for idx, (filename, (eps_grid, pdf_eps)) in enumerate(pdf_data.items()):
        if len(eps_grid) == 0:
            continue
        
        label_base = Path(filename).stem
        
        if resolve_line_style_func:
            color, lw, dash = resolve_line_style_func(
                filename, idx, colors, ps,
                style_key="per_sim_style_comparison",
                include_marker=False,
                default_marker="circle"
            )
        else:
            color = colors[idx % len(colors)]
            lw = line_width
            dash = "solid"
        
        fig.add_trace(go.Scatter(
            x=eps_grid,
            y=pdf_eps,
            mode='lines',
            name=label_base,
            line=dict(color=color, width=lw, dash=dash),
            hovertemplate=f"ε = %{{x:.4e}}<br>PDF = %{{y:.4e}}<extra>{label_base}</extra>"
        ))
    
    x_label = "ε / ⟨ε⟩" if normalize_pdf else "ε"
    y_label = "⟨ε⟩ P(ε / ⟨ε⟩)" if normalize_pdf else "P(ε)"
    layout_kwargs = dict(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=ps.get("figure_height", 500) if ps else 500,
        hovermode='x unified',
        legend=dict(x=1.02, y=1)
    )
    
    if ps:
        from utils.plot_style import apply_axis_limits, apply_figure_size
        layout_kwargs = apply_axis_limits(layout_kwargs, ps)
        layout_kwargs = apply_figure_size(layout_kwargs, ps)
    
    fig.update_layout(**layout_kwargs)
    
    if apply_plot_style_func and ps:
        fig = apply_plot_style_func(fig, ps)
    
    st.plotly_chart(
        fig, 
        width='stretch',
        config={
            "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "dissipation_pdf",
                "height": None,
                "width": None,
                "scale": 2
            }
        }
    )
    
    if capture_button_func:
        capture_button_func(fig, title="Dissipation Rate PDF", source_page="PDFs")
    
    if export_panel_func:
        export_panel_func(fig, data_dir, "dissipation_pdf")
    
    # Theory & Equations
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("### Dissipation Rate PDF")
        st.markdown("**Probability Density Function of dissipation rate:**")
        st.latex(r"P(\varepsilon) = \frac{1}{N \Delta \varepsilon} \sum_{i=1}^{N} \delta(\varepsilon - \varepsilon_i)")
        st.markdown("where the dissipation rate is defined as:")
        st.latex(r"\varepsilon = 2\nu S_{ij} S_{ij}")
        st.markdown("where:")
        st.markdown("- $\\nu$ is the kinematic viscosity")
        st.markdown("- $S_{ij} = \\frac{1}{2}\\left(\\frac{\\partial u_i}{\\partial x_j} + \\frac{\\partial u_j}{\\partial x_i}\\right)$ is the strain rate tensor")
        st.markdown("- $S_{ij} S_{ij}$ is the double contraction (sum over $i$ and $j$)")
        st.markdown("The dissipation rate represents the rate at which kinetic energy is converted to internal energy")
        st.markdown("through viscous effects. The PDF of dissipation is typically log-normal in turbulence,")
        st.markdown("reflecting the intermittent nature of energy dissipation.")

