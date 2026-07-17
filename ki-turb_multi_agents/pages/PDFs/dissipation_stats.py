"""
Dissipation Rate Statistical Analysis
Module for computing and visualizing dissipation rate PDFs
"""

import streamlit as st
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from core_physics import compute_dissipation_pdf, compute_dissipation_statistics
from data_readers.parameter_reader import read_parameters


def render_dissipation_tab(data_dir_or_dirs, load_velocity_file_func,
                            get_plot_style_func=None, apply_plot_style_func=None,
                            get_palette_func=None, resolve_line_style_func=None,
                            export_panel_func=None, capture_button_func=None,
                            dx=1.0, dy=1.0, dz=1.0, nu=0.004):
    """Render the Dissipation Rate PDF tab content"""
    import glob
    from utils.file_detector import natural_sort_key
    
    st.header("Dissipation Rate PDF")
    st.markdown("Compare dissipation rate PDFs across different simulations/methods.")
    axis_labels = st.session_state.get("axis_labels_pdfs", {})
    legend_titles = st.session_state.get("legend_titles_pdfs", {})
    
    # Handle both single directory and multiple directories
    if isinstance(data_dir_or_dirs, (list, tuple)):
        data_dirs = [Path(d).resolve() for d in data_dir_or_dirs]
        data_dir = data_dirs[0]  # Use first for metadata
    else:
        data_dirs = [Path(data_dir_or_dirs).resolve()]
        data_dir = data_dirs[0]
    
    # Find velocity files from ALL directories independently
    from utils.file_detector import list_velocity_field_files
    all_files = list_velocity_field_files(data_dirs)

    if not all_files:
        st.error(
            "No velocity files found. Expected: `velocity_*.vti`, `Velocity_*.vti`, "
            "`velocity_*.h5` / `.hdf5` (density/vorticity dumps are ignored)."
        )
        return
    
    # Create mapping from filename to full path (handle files from different directories)
    filename_to_path = {Path(f).name: f for f in all_files}
    
    # File selection (shared with Autonomous Lab agent workflow)
    st.sidebar.header("📁 File Selection")
    st.sidebar.caption(f"Found {len(all_files)} velocity files")
    file_options = [Path(f).name for f in all_files]
    default_files = [Path(f).name for f in all_files[:min(3, len(all_files))]]
    if "dissipation_file_select" in st.session_state:
        valid = [f for f in st.session_state["dissipation_file_select"] if f in file_options]
        if valid != st.session_state["dissipation_file_select"]:
            st.session_state["dissipation_file_select"] = valid if valid else default_files
    session_selection = st.session_state.get("dissipation_file_select")
    default_selection = [f for f in (session_selection or default_files) if f in file_options] or default_files

    selected_files = st.sidebar.multiselect(
        "Dissipation PDF files:",
        options=file_options,
        default=default_selection,
        help="Select files for Dissipation Rate PDF plot",
        key="dissipation_file_select"
    )
    
    if not selected_files:
        st.warning("Please select at least one file.")
        return
    
    # Plot parameters (shared with Autonomous Lab agent workflow)
    st.sidebar.header("Plot Parameters")
    if "dissipation_pdf_bins" in st.session_state and not (50 <= st.session_state["dissipation_pdf_bins"] <= 500):
        st.session_state["dissipation_pdf_bins"] = 100

    pdf_bins = st.sidebar.slider(
        "PDF bins", 50, 500,
        value=st.session_state.get("dissipation_pdf_bins", 100),
        step=10, key="dissipation_pdf_bins"
    )
    normalize_pdf = st.sidebar.checkbox(
        "Normalize by mean (ε/⟨ε⟩)",
        value=st.session_state.get("dissipation_normalize", False),
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
                    st.warning(f"{filename}: Invalid velocity shape")
                    continue
                
                # Try to get viscosity from metadata, parameter file, or use sidebar value
                metadata = vti_data.get('metadata', {})
                file_nu = metadata.get('nu', metadata.get('viscosity', None))
                if file_nu is None:
                    # Try parameter file (simulation.input or simulation.json) in file's directory
                    file_dir = Path(filepath).parent
                    for candidate in (file_dir / "simulation.input", file_dir / "simulation.json"):
                        if candidate.exists():
                            try:
                                params = read_parameters(str(candidate))
                                file_nu = params.get('nu', nu)
                                break
                            except Exception:
                                pass
                    if file_nu is None:
                        file_nu = nu
                
                # Compute PDF
                eps_grid, pdf_eps = compute_dissipation_pdf(
                    velocity, 
                    nu=file_nu, 
                    bins=pdf_bins, 
                    dx=dx, 
                    dy=dy, 
                    dz=dz,
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
    with st.expander("Statistical Moments (Skewness & Kurtosis)", expanded=False):
        from .velocity_magnitude_stats import display_statistics_table
        
        # Compute statistics from first available file
        stats_dict = {}
        if selected_files:
            first_file = selected_files[0]
            filepath = filename_to_path.get(first_file, data_dir / first_file)
            try:
                vti_data = load_velocity_file_func(str(filepath))
                velocity = vti_data['velocity']
                if velocity is not None and len(velocity.shape) == 4:
                    mean, rms, skew, kurt = compute_dissipation_statistics(velocity, nu=nu, dx=dx, dy=dy, dz=dz)
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
    
    x_label_default = "ε / ⟨ε⟩" if normalize_pdf else "ε"
    y_label_default = "⟨ε⟩ P(ε / ⟨ε⟩)" if normalize_pdf else "P(ε)"
    x_label = axis_labels.get("dissipation_x", x_label_default)
    y_label = axis_labels.get("dissipation_y", y_label_default)
    legend_title = legend_titles.get("dissipation_pdf", "")
    layout_kwargs = dict(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=ps.get("figure_height", 500) if ps else 500,
        hovermode='x unified',
        legend=dict(x=1.02, y=1),
        legend_title_text=legend_title if legend_title else None
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

