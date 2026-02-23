"""
Joint Probability Density Function Analysis
Module for computing and visualizing joint PDFs of turbulence quantities
"""

import streamlit as st
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from core_physics import (
    compute_velocity_dissipation_joint_pdf,
    compute_velocity_enstrophy_joint_pdf,
    compute_dissipation_enstrophy_joint_pdf,
    compute_rq_joint_pdf,
    compute_discriminant_line,
)
from data_readers.parameter_reader import read_parameters


def render_joint_pdf_tab(data_dir_or_dirs, load_velocity_file_func,
                          get_plot_style_func=None, apply_plot_style_func=None,
                          get_palette_func=None, resolve_line_style_func=None,
                          export_panel_func=None, capture_button_func=None,
                          dx=1.0, dy=1.0, dz=1.0, nu=0.004):
    """Render the Joint PDFs tab content"""
    import glob
    from pathlib import Path
    from utils.file_detector import natural_sort_key
    
    st.header("Joint Probability Density Functions")
    st.markdown("Compare joint PDFs of turbulence quantities across different simulations/methods.")
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
    
    # File selection (shared with Autonomous Lab agent workflow)
    st.sidebar.header("📁 File Selection")
    st.sidebar.caption(f"Found {len(all_files)} velocity files")
    file_options = [Path(f).name for f in all_files]
    default_velocity_dissipation = [Path(f).name for f in all_files[:min(2, len(all_files))]]
    default_velocity_enstrophy = [Path(f).name for f in all_files[:min(2, len(all_files))]]
    default_dissipation_enstrophy = [Path(f).name for f in all_files[:min(2, len(all_files))]]
    default_rq = [Path(f).name for f in all_files[:min(3, len(all_files))]]

    def resolve_default_selection(session_value, fallback):
        valid = [f for f in (session_value or fallback) if f in file_options]
        return valid if valid else fallback

    # Validate selection against available files; reset if stale
    for key, fallback in [
        ("joint_pdf_files_ud", default_velocity_dissipation),
        ("joint_pdf_files_uo", default_velocity_enstrophy),
        ("joint_pdf_files_do", default_dissipation_enstrophy),
        ("joint_pdf_files_rq", default_rq),
    ]:
        if key in st.session_state:
            valid = [f for f in st.session_state[key] if f in file_options]
            if valid != st.session_state[key]:
                st.session_state[key] = valid if valid else fallback

    selected_files_ud = st.sidebar.multiselect(
        "Velocity-Dissipation PDF files:",
        options=file_options,
        default=resolve_default_selection(st.session_state.get("joint_pdf_files_ud"), default_velocity_dissipation),
        help="Select files for P(|u|, ε) plot",
        key="joint_pdf_files_ud"
    )
    selected_files_uo = st.sidebar.multiselect(
        "Velocity-Enstrophy PDF files:",
        options=file_options,
        default=resolve_default_selection(st.session_state.get("joint_pdf_files_uo"), default_velocity_enstrophy),
        help="Select files for P(|u|, |ω|) plot",
        key="joint_pdf_files_uo"
    )
    selected_files_do = st.sidebar.multiselect(
        "Dissipation-Enstrophy PDF files:",
        options=file_options,
        default=resolve_default_selection(st.session_state.get("joint_pdf_files_do"), default_dissipation_enstrophy),
        help="Select files for P(ε, |ω|) plot",
        key="joint_pdf_files_do"
    )
    selected_files_rq = st.sidebar.multiselect(
        "R-Q Topological Space files:",
        options=file_options,
        default=resolve_default_selection(st.session_state.get("joint_pdf_files_rq"), default_rq),
        help="Select files for R-Q plot",
        key="joint_pdf_files_rq"
    )
    
    if not selected_files_ud and not selected_files_uo and not selected_files_do and not selected_files_rq:
        st.warning("Please select at least one file for any joint PDF plot.")
        return
    
    # Plot parameters (shared with Autonomous Lab agent workflow)
    st.sidebar.header("Plot Parameters")
    for key, lo, hi, default in [
        ("joint_pdf_bins", 50, 200, 100),
        ("joint_pdf_rq_bins", 50, 200, 100),
    ]:
        if key in st.session_state and not (lo <= st.session_state[key] <= hi):
            st.session_state[key] = default

    pdf_bins = st.sidebar.slider(
        "PDF bins", 50, 200,
        value=st.session_state.get("joint_pdf_bins", 100),
        step=10, key="joint_pdf_bins"
    )
    rq_bins = st.sidebar.slider(
        "R-Q space bins", 50, 200,
        value=st.session_state.get("joint_pdf_rq_bins", 100),
        step=10, key="joint_pdf_rq_bins"
    )
    use_log_scale = st.sidebar.checkbox(
        "Log scale (PDF)",
        value=st.session_state.get("joint_pdf_log_scale", True),
        help="Use logarithmic scale to visualize joint PDF structure",
        key="joint_pdf_log_scale"
    )
    use_log_scale_rq = st.sidebar.checkbox(
        "Log scale (R-Q PDF)",
        value=st.session_state.get("joint_pdf_rq_log_scale", True),
        help="Use logarithmic scale to visualize the tear-drop shape (Vieillefosse tail)",
        key="joint_pdf_rq_log_scale"
    )
    normalize_pdf = st.sidebar.checkbox(
        "Normalize PDFs",
        value=st.session_state.get("joint_pdf_normalize", False),
        help="Normalize: |u| and |ω| by RMS, ε by mean",
        key="joint_pdf_normalize"
    )
    
    # Load and compute data independently for each plot
    ud_data = {}
    uo_data = {}
    do_data = {}
    rq_data = {}
    
    # Load data for Velocity-Dissipation joint PDF
    if selected_files_ud:
        for filename in selected_files_ud:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for P(|u|, ε)..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"{filename}: Invalid velocity shape")
                        continue
                    
                    # Try to get viscosity from metadata, parameter file, or use sidebar value
                    metadata = vti_data.get('metadata', {})
                    file_nu = metadata.get('nu', metadata.get('viscosity', None))
                    if file_nu is None:
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
                    
                    u_centers, eps_centers, joint_pdf = compute_velocity_dissipation_joint_pdf(
                        velocity, nu=file_nu, bins=pdf_bins, dx=dx, dy=dy, dz=dz, normalize=normalize_pdf
                    )
                    ud_data[filename] = (u_centers, eps_centers, joint_pdf)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for P(|u|, ε): {e}")
                continue
    
    # Load data for Velocity-Enstrophy joint PDF
    if selected_files_uo:
        for filename in selected_files_uo:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for P(|u|, |ω|)..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"{filename}: Invalid velocity shape")
                        continue
                    
                    u_centers, omega_centers, joint_pdf = compute_velocity_enstrophy_joint_pdf(
                        velocity, bins=pdf_bins, dx=dx, dy=dy, dz=dz, normalize=normalize_pdf
                    )
                    uo_data[filename] = (u_centers, omega_centers, joint_pdf)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for P(|u|, |ω|): {e}")
                continue
    
    # Load data for Dissipation-Enstrophy joint PDF
    if selected_files_do:
        for filename in selected_files_do:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for P(ε, |ω|)..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"{filename}: Invalid velocity shape")
                        continue
                    
                    # Try to get viscosity from metadata, parameter file, or use sidebar value
                    metadata = vti_data.get('metadata', {})
                    file_nu = metadata.get('nu', metadata.get('viscosity', None))
                    if file_nu is None:
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
                    
                    eps_centers, omega_centers, joint_pdf = compute_dissipation_enstrophy_joint_pdf(
                        velocity, nu=file_nu, bins=pdf_bins, dx=dx, dy=dy, dz=dz, normalize=normalize_pdf
                    )
                    do_data[filename] = (eps_centers, omega_centers, joint_pdf)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for P(ε, |ω|): {e}")
                continue
    
    # Load data for R-Q plot
    if selected_files_rq:
        for filename in selected_files_rq:
            # Use full path from mapping (handles files from different directories)
            filepath = filename_to_path.get(filename)
            if not filepath:
                st.warning(f"File not found: {filename}")
                continue
            try:
                with st.spinner(f"Loading {filename} for R-Q..."):
                    vti_data = load_velocity_file_func(str(filepath))
                    velocity = vti_data['velocity']
                    
                    if velocity is None or len(velocity.shape) != 4:
                        st.warning(f"{filename}: Invalid velocity shape")
                        continue
                    
                    # Compute R-Q joint PDF (uses dx for gradient tensor)
                    R_centers, Q_centers, joint_pdf = compute_rq_joint_pdf(
                        velocity, r_bins=rq_bins, q_bins=rq_bins, dx=dx, dy=dy, dz=dz
                    )
                    rq_data[filename] = (R_centers, Q_centers, joint_pdf)
                    
            except Exception as e:
                st.error(f"Error loading {filename} for R-Q: {e}")
                continue
    
    if not ud_data and not uo_data and not do_data and not rq_data:
        st.error("No valid velocity data loaded.")
        return
    
    # Create plots in a grid layout
    col1, col2 = st.columns(2)
    
    # ============================================
    # Top row: P(|u|, ε) and P(|u|, |ω|)
    # ============================================
    with col1:
        st.subheader("P(|u|, ε)")
        
        if not ud_data:
            st.info("Select files in the sidebar to plot P(|u|, ε)")
        else:
            plot_name_ud = "Velocity-Dissipation Joint PDF"
            ps_ud = get_plot_style_func(plot_name_ud) if get_plot_style_func else {}
            
            fig_ud = go.Figure()
            
            for idx, (filename, (u_centers, eps_centers, joint_pdf)) in enumerate(ud_data.items()):
                if len(u_centers) == 0 or len(eps_centers) == 0 or joint_pdf.size == 0:
                    continue
                
                label = Path(filename).stem
                
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
                
                fig_ud.add_trace(go.Contour(
                    x=u_centers,
                    y=eps_centers,
                    z=plot_data,
                    name=label,
                    colorscale=colorscale,
                    showscale=(idx == 0),
                    colorbar=dict(title=z_label) if idx == 0 else None,
                    hovertemplate=f"|u| = %{{x:.4f}}<br>ε = %{{y:.4e}}<br>{z_label} = %{{z:.2f}}<extra>{label}</extra>"
                ))
            
            x_label_ud_default = "|u| / σ<sub>|u|</sub>" if normalize_pdf else "|u|"
            y_label_ud_default = "ε / ⟨ε⟩" if normalize_pdf else "ε"
            x_label_ud = axis_labels.get("joint_ud_x", x_label_ud_default)
            y_label_ud = axis_labels.get("joint_ud_y", y_label_ud_default)
            legend_title_ud = legend_titles.get("joint_ud_pdf", "")
            layout_kwargs_ud = dict(
                xaxis_title=x_label_ud,
                yaxis_title=y_label_ud,
                height=ps_ud.get("figure_height", 500) if ps_ud else 500,
                hovermode='closest',
                legend=dict(x=1.02, y=1),
                legend_title_text=legend_title_ud if legend_title_ud else None
            )
            
            if ps_ud:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs_ud = apply_axis_limits(layout_kwargs_ud, ps_ud)
                layout_kwargs_ud = apply_figure_size(layout_kwargs_ud, ps_ud)
            
            fig_ud.update_layout(**layout_kwargs_ud)
            
            if apply_plot_style_func and ps_ud:
                fig_ud = apply_plot_style_func(fig_ud, ps_ud)
            
            st.plotly_chart(
                fig_ud, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "velocity_dissipation_joint_pdf",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_ud, title="P(|u|, ε)", source_page="PDFs")
            
            if export_panel_func:
                export_panel_func(fig_ud, data_dir, "velocity_dissipation_joint_pdf")
    
    with col2:
        st.subheader("P(|u|, |ω|)")
        
        if not uo_data:
            st.info("Select files in the sidebar to plot P(|u|, |ω|)")
        else:
            plot_name_uo = "Velocity-Enstrophy Joint PDF"
            ps_uo = get_plot_style_func(plot_name_uo) if get_plot_style_func else {}
            
            fig_uo = go.Figure()
            
            for idx, (filename, (u_centers, omega_centers, joint_pdf)) in enumerate(uo_data.items()):
                if len(u_centers) == 0 or len(omega_centers) == 0 or joint_pdf.size == 0:
                    continue
                
                label = Path(filename).stem
                
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
                
                fig_uo.add_trace(go.Contour(
                    x=u_centers,
                    y=omega_centers,
                    z=plot_data,
                    name=label,
                    colorscale=colorscale,
                    showscale=(idx == 0),
                    colorbar=dict(title=z_label) if idx == 0 else None,
                    hovertemplate=f"|u| = %{{x:.4f}}<br>|ω| = %{{y:.4f}}<br>{z_label} = %{{z:.2f}}<extra>{label}</extra>"
                ))
            
            x_label_uo_default = "|u| / σ<sub>|u|</sub>" if normalize_pdf else "|u|"
            y_label_uo_default = "|ω| / σ<sub>|ω|</sub>" if normalize_pdf else "|ω|"
            x_label_uo = axis_labels.get("joint_uo_x", x_label_uo_default)
            y_label_uo = axis_labels.get("joint_uo_y", y_label_uo_default)
            legend_title_uo = legend_titles.get("joint_uo_pdf", "")
            layout_kwargs_uo = dict(
                xaxis_title=x_label_uo,
                yaxis_title=y_label_uo,
                height=ps_uo.get("figure_height", 500) if ps_uo else 500,
                hovermode='closest',
                legend=dict(x=1.02, y=1),
                legend_title_text=legend_title_uo if legend_title_uo else None
            )
            
            if ps_uo:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs_uo = apply_axis_limits(layout_kwargs_uo, ps_uo)
                layout_kwargs_uo = apply_figure_size(layout_kwargs_uo, ps_uo)
            
            fig_uo.update_layout(**layout_kwargs_uo)
            
            if apply_plot_style_func and ps_uo:
                fig_uo = apply_plot_style_func(fig_uo, ps_uo)
            
            st.plotly_chart(
                fig_uo, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "velocity_enstrophy_joint_pdf",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_uo, title="P(|u|, |ω|)", source_page="PDFs")
            
            if export_panel_func:
                export_panel_func(fig_uo, data_dir, "velocity_enstrophy_joint_pdf")
    
    # ============================================
    # Bottom row: P(ε, |ω|) and R-Q Topological Space
    # ============================================
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("P(ε, |ω|)")
        
        if not do_data:
            st.info("Select files in the sidebar to plot P(ε, |ω|)")
        else:
            plot_name_do = "Dissipation-Enstrophy Joint PDF"
            ps_do = get_plot_style_func(plot_name_do) if get_plot_style_func else {}
            
            fig_do = go.Figure()
            
            for idx, (filename, (eps_centers, omega_centers, joint_pdf)) in enumerate(do_data.items()):
                if len(eps_centers) == 0 or len(omega_centers) == 0 or joint_pdf.size == 0:
                    continue
                
                label = Path(filename).stem
                
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
                
                fig_do.add_trace(go.Contour(
                    x=eps_centers,
                    y=omega_centers,
                    z=plot_data,
                    name=label,
                    colorscale=colorscale,
                    showscale=(idx == 0),
                    colorbar=dict(title=z_label) if idx == 0 else None,
                    hovertemplate=f"ε = %{{x:.4e}}<br>|ω| = %{{y:.4f}}<br>{z_label} = %{{z:.2f}}<extra>{label}</extra>"
                ))
            
                x_label_do_default = "ε / ⟨ε⟩" if normalize_pdf else "ε"
                y_label_do_default = "|ω| / σ<sub>|ω|</sub>" if normalize_pdf else "|ω|"
                x_label_do = axis_labels.get("joint_do_x", x_label_do_default)
                y_label_do = axis_labels.get("joint_do_y", y_label_do_default)
                legend_title_do = legend_titles.get("joint_do_pdf", "")
                layout_kwargs_do = dict(
                    xaxis_title=x_label_do,
                    yaxis_title=y_label_do,
                height=ps_do.get("figure_height", 500) if ps_do else 500,
                hovermode='closest',
                    legend=dict(x=1.02, y=1),
                    legend_title_text=legend_title_do if legend_title_do else None
            )
            
            if ps_do:
                from utils.plot_style import apply_axis_limits, apply_figure_size
                layout_kwargs_do = apply_axis_limits(layout_kwargs_do, ps_do)
                layout_kwargs_do = apply_figure_size(layout_kwargs_do, ps_do)
            
            fig_do.update_layout(**layout_kwargs_do)
            
            if apply_plot_style_func and ps_do:
                fig_do = apply_plot_style_func(fig_do, ps_do)
            
            st.plotly_chart(
                fig_do, 
                width='stretch',
                config={
                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "dissipation_enstrophy_joint_pdf",
                        "height": None,
                        "width": None,
                        "scale": 2
                    }
                }
            )
            
            if capture_button_func:
                capture_button_func(fig_do, title="P(ε, |ω|)", source_page="PDFs")
            
            if export_panel_func:
                export_panel_func(fig_do, data_dir, "dissipation_enstrophy_joint_pdf")
    
    with col4:
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
                if use_log_scale_rq:
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
            
            x_label_rq = axis_labels.get("rq_x", "R")
            y_label_rq = axis_labels.get("rq_y", "Q")
            legend_title_rq = legend_titles.get("rq_pdf", "")
            layout_kwargs_rq = dict(
                xaxis_title=x_label_rq,
                yaxis_title=y_label_rq,
                height=ps_rq.get("figure_height", 500) if ps_rq else 500,
                hovermode='closest',
                legend=dict(x=1.02, y=1),
                legend_title_text=legend_title_rq if legend_title_rq else None
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
                capture_button_func(fig_rq, title="R-Q Topological Space", source_page="PDFs")
            
            if export_panel_func:
                export_panel_func(fig_rq, data_dir, "rq_topological_space")
    
    # Theory & Equations
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("### Joint Probability Density Functions")
        st.markdown("**Joint PDF of two turbulence quantities:**")
        st.latex(r"P(X, Y) = \frac{1}{N \Delta X \Delta Y} \sum_{i=1}^{N} \delta(X - X_i) \delta(Y - Y_i)")
        st.markdown("where $N$ is the total number of grid points and $\\Delta X, \\Delta Y$ are bin widths.")
        
        st.markdown("### P(|u|, ε)")
        st.markdown("Joint PDF of velocity magnitude and dissipation rate.")
        st.markdown("Reveals correlations between flow speed and energy dissipation.")
        
        st.markdown("### P(|u|, |ω|)")
        st.markdown("Joint PDF of velocity magnitude and vorticity magnitude.")
        st.markdown("Shows relationships between flow speed and rotational intensity.")
        
        st.markdown("### P(ε, |ω|)")
        st.markdown("Joint PDF of dissipation rate and vorticity magnitude.")
        st.markdown("Illustrates connections between energy dissipation and vortical structures.")
        
        st.markdown("### R-Q Topological Space")
        st.markdown("**Reference:** [Kareem & Asker (2022)](/Citation#kareem2022) — Simulations of isotropic turbulent flows using lattice Boltzmann method with different forcing functions")
        st.markdown("**Joint PDF of velocity gradient tensor invariants:**")
        st.markdown("**Second Invariant Q:**")
        st.latex(r"Q = -\frac{1}{2} A_{ij} A_{ij} = \frac{1}{4}\left(\omega_i\omega_i - 2S_{ij}S_{ij}\right)")
        st.markdown("**Third Invariant R:**")
        st.latex(r"R = -\frac{1}{3}\left(S_{ij}S_{jk}S_{ki} + \frac{3}{4}\omega_i\omega_j S_{ij}\right)")
        st.markdown("**Zero Discriminant Line:**")
        st.latex(r"D = \left(\frac{Q}{3}\right)^3 + \left(\frac{R}{2}\right)^2 = 0")
        st.latex(r"Q = -3\left(\frac{R}{2}\right)^{2/3}")
        st.markdown("This line separates regions with real eigenvalues (above) from complex eigenvalues (below) of the velocity gradient tensor.")
        st.markdown("**Visualization:** The logarithmic scale is essential to visualize the **Vieillefosse tail** (tear-drop shape) in the bottom-right quadrant ($R>0, Q<0$), where probability densities are orders of magnitude lower than the core region.")

