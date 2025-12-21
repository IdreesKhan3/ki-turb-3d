"""
Other Turbulence Stats Page (Streamlit)

Features:
- Turbulence statistics from turbulence_stats*.csv files
- Energy balance residual from eps_real_validation*.csv files
- Full persistent UI controls:
    * Legend names, axis labels
    * Fonts, tick style, major/minor grids, background colors, theme
    * Palette / custom colors
    * Per-simulation overrides: color/width/dash
- Research-grade export:
    * User can export to: PNG, PDF, SVG, JPG/JPEG, WEBP
- Robust to missing columns/files
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import glob

# --- Project imports ---
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from data_readers.csv_reader import read_csv_data
from utils.file_detector import detect_simulation_files
from utils.theme_config import apply_theme_to_plot_style, inject_theme_css, template_selector
from utils.report_builder import capture_button
from utils.plot_style import (
    resolve_line_style, render_per_sim_style_ui, render_axis_limits_ui, apply_axis_limits, 
    render_figure_size_ui, apply_figure_size, default_plot_style, apply_plot_style as apply_plot_style_base,
    plot_style_sidebar as shared_plot_style_sidebar, _get_palette, _axis_title_font, _tick_font,
    get_tick_format, ensure_per_sim_defaults, render_legend_axis_labels_ui, convert_superscript
)
from utils.export_figs import export_panel
st.set_page_config(page_icon="⚫")

# ==========================================================
# Plot styling (using shared utilities from utils.plot_style)
# ==========================================================

# Alias for backward compatibility
_default_plot_style = default_plot_style

def _get_title_dict(ps, title_text):
    """Get title dict with font color for dark theme compatibility."""
    if not title_text:
        return None
    
    # Get font color from plot style (defaults based on template)
    font_color = ps.get("font_color")
    if font_color is None:
        # Auto-detect from template if font_color not set
        template = ps.get("template", "plotly_white")
        if "dark" in template.lower():
            font_color = "#d4d4d4"
        else:
            font_color = "#000000"
    
    return dict(
        text=convert_superscript(title_text),
        font=dict(
            family=ps.get("font_family", "Arial"),
            size=ps.get("title_size", 16),
            color=font_color
        )
    )

def apply_plot_style(fig, ps):
    # Set default plot_title if not set and show_plot_title is True
    if ps.get("show_plot_title", False) and (not ps.get("plot_title") or ps.get("plot_title") == ""):
        ps["plot_title"] = "Custom Multi-Trace Plot"
    
    # Clear plot_title if show_plot_title is False to prevent centralized function from setting it
    original_plot_title = ps.get("plot_title", "")
    if not ps.get("show_plot_title", False):
        ps["plot_title"] = ""
    
    fig = apply_plot_style_base(fig, ps)
    
    # Restore original plot_title for later use
    ps["plot_title"] = original_plot_title
    
    if not ps.get("show_plot_title", False):
        fig.update_layout(title=None)
    
    # Always set title with correct font color if show_plot_title is True and plot_title exists
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=_get_title_dict(ps, ps["plot_title"]))
    
    return fig

def plot_style_sidebar(data_dir: Path, sim_groups):
    """Plot style sidebar using shared utilities."""
    if "plot_style" not in st.session_state:
        st.session_state.plot_style = default_plot_style()
    ps = st.session_state.plot_style
    ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style_turb_stats", include_marker=True)
    
    def _reset_style():
        st.session_state.plot_style = default_plot_style()
    
    return shared_plot_style_sidebar(
        data_dir=None,
        sim_groups=sim_groups,
        style_key="per_sim_style_turb_stats",
        key_prefix="turb_stats",
        include_marker=True,
        save_callback=None,
        reset_callback=_reset_style,
        theme_selector=template_selector
    )



# ==========================================================
# Page main
# ==========================================================
def main():
    inject_theme_css()
    
    st.title("Other Turbulence Stats")

    # Get data directories from session state (support multiple directories)
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        # Fallback to single directory for backward compatibility
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return

    # Use first directory for metadata storage
    data_dir = Path(data_dirs[0])
    
    
    # Show which directories are loaded
    if len(data_dirs) > 1:
        st.info(f"📁 **Multiple simulations loaded:** {len(data_dirs)} directories")
        with st.expander("View loaded directories", expanded=False):
            project_root = Path(__file__).parent.parent.resolve()
            for i, data_dir_path in enumerate(data_dirs, 1):
                data_dir_obj = Path(data_dir_path)
                try:
                    rel_path = data_dir_obj.relative_to(project_root)
                    st.markdown(f"**{i}.** `{rel_path}`")
                except ValueError:
                    st.markdown(f"**{i}.** `{data_dir_path}`")
        st.markdown("---")
    
    # Apply theme to plot style on page load
    current_theme = st.session_state.get("theme", "Light Scientific")
    if 'plot_style' not in st.session_state:
        st.session_state.plot_style = default_plot_style()
    
    # Always apply current theme (in case theme changed)
    st.session_state.plot_style = apply_theme_to_plot_style(
        st.session_state.plot_style, 
        current_theme
    )

    # Detect available files from all directories
    all_files_dict = {}
    for data_dir_path in data_dirs:
        data_dir_obj = Path(data_dir_path)
        if data_dir_obj.exists():
            dir_files = detect_simulation_files(str(data_dir_obj))
            # Merge files from all directories
            for file_type, file_list in dir_files.items():
                if file_type not in all_files_dict:
                    all_files_dict[file_type] = []
                # Convert Path objects to strings for consistency
                all_files_dict[file_type].extend([str(f) if isinstance(f, Path) else f for f in file_list])
    
    files = all_files_dict

    # Collect all available dataframes (load data first, but don't display tables yet)
    all_dataframes = {}
    available_columns = {}
    table_data = {}  # Store table data for display later
    
    # Load turbulence_stats CSV from all directories
    csv_files = files.get('real_turb_stats', [])
    
    if csv_files:
        if len(data_dirs) > 1:
            # Multiple directories: load all
            for csv_file in csv_files:
                csv_path = Path(csv_file).resolve()  # Make absolute
                # Find which directory this file belongs to
                dir_name = None
                for data_dir_path in data_dirs:
                    data_dir_obj = Path(data_dir_path).resolve()  # Make absolute
                    # Check if csv_path is within data_dir_obj (Python 3.8+ compatible)
                    try:
                        csv_path.relative_to(data_dir_obj)
                        dir_name = Path(data_dir_path).name
                        break
                    except ValueError:
                        # Path is not relative to this directory, try next
                        continue
                
                if not dir_name:
                    dir_name = csv_path.parent.name
                
                try:
                    df_stats = read_csv_data(str(csv_file))
                    key = f"turbulence_stats_{dir_name}"
                    all_dataframes[key] = df_stats
                    available_columns[key] = list(df_stats.columns)
                    table_data[key] = {'df': df_stats, 'dir_name': dir_name, 'type': 'turbulence_stats'}
                except Exception as e:
                    st.warning(f"Could not load {csv_path.name}: {e}")
                    continue
        else:
            # Single directory: original behavior
            try:
                df_stats = read_csv_data(str(files['real_turb_stats'][0]))
                all_dataframes['turbulence_stats'] = df_stats
                available_columns['turbulence_stats'] = list(df_stats.columns)
                table_data['turbulence_stats'] = {'df': df_stats, 'dir_name': None, 'type': 'turbulence_stats'}
            except Exception as e:
                st.warning(f"Could not load turbulence stats: {e}")

    # Load eps_real_validation CSV files from all directories
    eps_files = files.get("spectral_turb_stats", [])
    if not eps_files:
        # Fallback: search in first directory
        eps_files = glob.glob(str(data_dir / "eps_real_validation*.csv"))
    
    if eps_files:
        if len(data_dirs) > 1:
            # Multiple directories: load all and store with directory labels
            for eps_file in eps_files:
                eps_path = Path(eps_file).resolve()  # Make absolute
                # Find which directory this file belongs to
                dir_name = None
                for data_dir_path in data_dirs:
                    data_dir_obj = Path(data_dir_path).resolve()  # Make absolute
                    # Check if eps_path is within data_dir_obj (Python 3.8+ compatible)
                    try:
                        eps_path.relative_to(data_dir_obj)
                        dir_name = Path(data_dir_path).name
                        break
                    except ValueError:
                        # Path is not relative to this directory, try next
                        continue
                
                if not dir_name:
                    dir_name = eps_path.parent.name
                
                try:
                    df_val = pd.read_csv(str(eps_file))
                    key = f"eps_validation_{dir_name}"
                    all_dataframes[key] = df_val
                    available_columns[key] = list(df_val.columns)
                    table_data[key] = {'df': df_val, 'dir_name': dir_name, 'type': 'eps_validation'}
                except Exception as e:
                    st.warning(f"Could not load {eps_path.name} from {dir_name}: {e}")
                    continue
        else:
            # Single directory: original behavior
            try:
                df_val = pd.read_csv(str(eps_files[0]))
                all_dataframes['eps_validation'] = df_val
                available_columns['eps_validation'] = list(df_val.columns)
                table_data['eps_validation'] = {'df': df_val, 'dir_name': None, 'type': 'eps_validation'}
            except Exception:
                pass

    # =========================
    # Custom Plotting Section
    # =========================
    st.header("Custom Plotting")
    
    if not all_dataframes:
        st.info("No CSV files found. Please load data from the Overview page.")
        return
    
    # Create sim_groups structure for plot style sidebar (use full unique dataframe key names)
    sim_groups = {}
    for key in all_dataframes.keys():
        # Use the full unique key name (e.g., "turbulence_stats_768", "eps_validation_768")
        # This allows separate styling for stats vs validation files, like Energy Spectra page
        if key not in sim_groups:
            sim_groups[key] = []  # Empty list is fine, just need the keys
    
    # Add plot style sidebar (fonts, colors, grids, etc.)
    if sim_groups:
        plot_style_sidebar(data_dir, sim_groups)
    
    # Initialize traces in session state
    if 'custom_plot_traces' not in st.session_state:
        st.session_state.custom_plot_traces = []
    
    # Initialize legend names and axis labels in session state
    if 'custom_plot_legend_names' not in st.session_state:
        st.session_state.custom_plot_legend_names = {}
    if 'custom_plot_axis_labels' not in st.session_state:
        st.session_state.custom_plot_axis_labels = {
            'x': 'X',
            'y': 'Y'
        }
    
    # Legend & Axis Labels - using shared utility
    def _reset_legend_axis():
        st.session_state.custom_plot_legend_names = {}
        st.session_state.custom_plot_axis_labels = {'x': 'X', 'y': 'Y'}
    
    legend_names, axis_labels = render_legend_axis_labels_ui(
        data_dir=None,
        traces=st.session_state.custom_plot_traces if st.session_state.custom_plot_traces else None,
        legend_names_key="custom_plot_legend_names",
        axis_labels_key="custom_plot_axis_labels",
        trace_key_func=lambda trace: f"{trace['data_source']}_{trace['x_col']}_{trace['y_col']}",
        save_callback=None,
        reset_callback=_reset_legend_axis,
        key_prefix="custom"
    )
    
    # Sidebar: Plot options (global for all traces)
    st.sidebar.subheader("Plot Options")
    use_abs = st.sidebar.checkbox("Use absolute value (Y-axis)", value=False, key="plot_use_abs")
    smooth_window = st.sidebar.slider(
        "Moving average window (0=off)", 
        0, 500, 0, 10, 
        key="plot_smooth",
        help="Smooths the curve by averaging over N consecutive points. Reduces noise but also reduces the number of data points by (N-1). Example: window=5 averages every 5 points into 1 smoothed point."
    )
    normalize_x = st.sidebar.checkbox("Normalize X-axis", value=False, key="plot_norm_x")
    x_norm = st.sidebar.number_input("X normalization constant", value=1000.0, min_value=1.0, step=100.0, disabled=not normalize_x, key="plot_x_norm")
    normalize_y = st.sidebar.checkbox("Normalize Y-axis by maximum", value=False, key="plot_norm_y", help="Normalize each trace's Y values by its maximum value")
    
    # Main area: Add/Manage Traces
    st.subheader("Add Traces to Plot")
    
    # Add new trace section
    with st.expander("➕ Add New Trace", expanded=len(st.session_state.custom_plot_traces) == 0):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            data_source = st.selectbox(
                "📁 Data Source",
                options=list(all_dataframes.keys()),
                help="Select which CSV file to plot from",
                key="new_trace_source"
            )
        
        df_plot = all_dataframes[data_source]
        numeric_cols = [col for col in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[col])]
        
        if len(numeric_cols) < 2:
            st.warning(f"Not enough numeric columns in {data_source} for plotting.")
        else:
            with col2:
                x_col = st.selectbox(
                    "X-axis Column",
                    options=numeric_cols,
                    index=0 if 'iter' in numeric_cols else 0,
                    help="Select column for X-axis",
                    key="new_trace_x"
                )
            
            with col3:
                y_col = st.selectbox(
                    "Y-axis Column",
                    options=[c for c in numeric_cols if c != x_col],
                    index=0 if 'TKE' in [c for c in numeric_cols if c != x_col] else 0,
                    help="Select column for Y-axis",
                    key="new_trace_y"
                )
            
            with col4:
                trace_label = st.text_input(
                    "Label",
                    value=f"{data_source.split('_')[-1]}: {y_col}",
                    help="Trace label for legend",
                    key="new_trace_label"
                )
            
            if st.button("Add Trace", key="add_trace_btn"):
                trace_config = {
                    'data_source': data_source,
                    'x_col': x_col,
                    'y_col': y_col,
                    'label': trace_label
                }
                st.session_state.custom_plot_traces.append(trace_config)
                st.rerun()
    
    # Display and manage existing traces
    if st.session_state.custom_plot_traces:
        st.subheader("Current Traces")
        for idx, trace in enumerate(st.session_state.custom_plot_traces):
            with st.expander(f"Trace {idx+1}: {trace['label']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"Source: {trace['data_source']}")
                    st.text(f"X: {trace['x_col']} | Y: {trace['y_col']}")
                with col2:
                    if st.button("Remove", key=f"remove_trace_{idx}"):
                        st.session_state.custom_plot_traces.pop(idx)
                        st.rerun()
        
        if st.button("Clear All Traces", key="clear_all_traces"):
            st.session_state.custom_plot_traces = []
            st.rerun()
        
        st.markdown("---")
        
        # Create plot with all traces
        ps = st.session_state.get("plot_style", default_plot_style())
        colors = _get_palette(ps)
        fig = go.Figure()
        
        all_x_labels = set()
        all_y_labels = set()
        
        for idx, trace in enumerate(st.session_state.custom_plot_traces):
            data_source = trace['data_source']
            x_col = trace['x_col']
            y_col = trace['y_col']
            # Use custom legend name if available, otherwise use trace label
            # Use returned legend_names dict to ensure latest state (especially after reset)
            trace_key = f"{data_source}_{x_col}_{y_col}"
            label = legend_names.get(
                trace_key, 
                trace.get('label', f"{data_source.split('_')[-1]}: {y_col}")
            )
            
            if data_source not in all_dataframes:
                continue
            
            df_plot = all_dataframes[data_source]
            
            if x_col not in df_plot.columns or y_col not in df_plot.columns:
                continue
            
            x_data = df_plot[x_col].values
            y_data = df_plot[y_col].values
            
            # Convert to numeric and remove NaN
            x_data = pd.to_numeric(x_data, errors='coerce')
            y_data = pd.to_numeric(y_data, errors='coerce')
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            
            if len(x_data) == 0 or len(y_data) == 0:
                continue
            
            # Apply normalization to X
            if normalize_x:
                x_data = x_data / float(x_norm)
            
            # Apply absolute value to Y
            if use_abs:
                y_data = np.abs(y_data)
            
            # Apply normalization to Y (by maximum value of this trace)
            if normalize_y:
                y_max = np.max(np.abs(y_data)) if len(y_data) > 0 else 1.0
                if y_max > 0:
                    y_data = y_data / y_max
            
            # Filter out non-positive values if log scale is selected (log requires > 0)
            x_axis_type = ps.get("x_axis_type", "linear")
            y_axis_type = ps.get("y_axis_type", "linear")
            
            if x_axis_type == "log":
                # Log scale requires strictly positive values
                log_x_mask = x_data > 0
                x_data = x_data[log_x_mask]
                y_data = y_data[log_x_mask]
            
            if y_axis_type == "log":
                # Log scale requires strictly positive values
                log_y_mask = y_data > 0
                x_data = x_data[log_y_mask]
                y_data = y_data[log_y_mask]
            
            if len(x_data) == 0 or len(y_data) == 0:
                continue
            
            # Build hover template showing original column names and transformations
            hover_x_label = x_col
            if normalize_x:
                hover_x_label = f"{x_col} (normalized)"
            
            hover_y_label = y_col
            if use_abs:
                hover_y_label = f"|{hover_y_label}|"
            if normalize_y:
                hover_y_label = f"{hover_y_label} / max"
            
            # Use full data_source name as simulation identifier for per-simulation styling
            # e.g., "turbulence_stats_768", "eps_validation_768", "turbulence_stats_512", etc.
            sim_prefix = data_source
            
            # Get per-simulation style (color, width, dash, marker, marker_size)
            color, width, dash, marker, marker_size, override_on = resolve_line_style(
                sim_prefix, idx, colors, ps, 
                style_key="per_sim_style_turb_stats", 
                include_marker=True
            )
            
            # Build line style dict
            line_style = dict(width=width, color=color)
            if dash and dash != "solid":
                line_style["dash"] = dash
            
            # Determine mode and marker settings
            mode = "lines"
            marker_dict = None
            if override_on and marker and marker != "none":
                mode = "lines+markers"
                marker_dict = dict(symbol=marker, size=marker_size)
            
            # Apply smoothing - show both original (dim) and smoothed (bright) if smoothing is enabled
            if smooth_window > 1 and len(y_data) > smooth_window:
                # Add original noisy data as a dim background line
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    name=f"{label} (original)",
                    line=dict(width=1.0, color=color),
                    opacity=0.3,  # Set opacity on trace, not line
                    hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label} (original)=%{{y:.4g}}<extra></extra>",
                    showlegend=False,  # Don't clutter legend with original lines
                ))
                
                # Apply smoothing
                kernel = np.ones(int(smooth_window)) / int(smooth_window)
                y_smooth = np.convolve(y_data, kernel, mode="valid")
                x_smooth = x_data[int(smooth_window)//2: int(smooth_window)//2 + len(y_smooth)]
                x_plot, y_plot = x_smooth, y_smooth
                
                # Add smoothed line as main bright line
                hover_y_label_smooth = f"{hover_y_label} (smoothed)"
                scatter_kwargs = dict(
                    x=x_plot,
                    y=y_plot,
                    mode=mode,
                    name=label,
                    line=line_style,
                    hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label_smooth}=%{{y:.4g}}<extra></extra>"
                )
                if marker_dict:
                    scatter_kwargs["marker"] = marker_dict
                fig.add_trace(go.Scatter(**scatter_kwargs))
            else:
                # No smoothing - just plot original data
                x_plot, y_plot = x_data, y_data
                scatter_kwargs = dict(
                    x=x_plot,
                    y=y_plot,
                    mode=mode,
                    name=label,
                    line=line_style,
                    hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label}=%{{y:.4g}}<extra></extra>",
                    showlegend=True  # Explicitly show in legend
                )
                if marker_dict:
                    scatter_kwargs["marker"] = marker_dict
                fig.add_trace(go.Scatter(**scatter_kwargs))
            
            all_x_labels.add(x_col)
            all_y_labels.add(y_col)
        
        if len(fig.data) == 0:
            st.warning("No valid traces to plot. Please add traces with valid data.")
        else:
            # Set axis labels - use returned values from render_legend_axis_labels_ui (ensures latest state after reset)
            custom_x_label = axis_labels.get('x', 'X')
            custom_y_label = axis_labels.get('y', 'Y')
            
            if custom_x_label and custom_x_label != 'X':
                x_label = custom_x_label
                if normalize_x:
                    x_label = f"{x_label} / {x_norm}"
            else:
                # Use default logic
                x_label = list(all_x_labels)[0] if len(all_x_labels) == 1 else "X"
                if normalize_x and len(all_x_labels) == 1:
                    x_label = f"{x_label} / {x_norm}"
                elif len(all_x_labels) > 1:
                    x_label = "X (multiple columns)"
            
            if custom_y_label and custom_y_label != 'Y':
                y_label = custom_y_label
                if use_abs:
                    y_label = f"|{y_label}|"
                if normalize_y:
                    y_label = f"{y_label} / max"
            else:
                # Use default logic
                y_label = list(all_y_labels)[0] if len(all_y_labels) == 1 else "Y"
                if len(all_y_labels) > 1:
                    y_label = "Y (multiple columns)"
                else:
                    # Single column: show original name, add transformation indicators
                    if use_abs:
                        y_label = f"|{y_label}|"
                    if normalize_y:
                        y_label = f"{y_label} / max({y_label})"
            
            # Apply full plot style first (fonts, ticks, grids, backgrounds, etc.)
            # This will also set the title via the local apply_plot_style function
            fig = apply_plot_style(fig, ps)
            
            # Then apply layout-specific settings (axis labels, size, limits)
            # Use update_xaxes/update_yaxes to preserve title_font settings
            fig.update_xaxes(title_text=x_label)
            fig.update_yaxes(title_text=y_label)
            
            layout_kwargs = dict(
                height=400,  # Default, will be overridden if custom size is enabled
                margin=dict(l=60, r=20, t=40, b=55),
            )
            layout_kwargs = apply_axis_limits(layout_kwargs, ps)
            layout_kwargs = apply_figure_size(layout_kwargs, ps)
            # Ensure legend is explicitly shown
            if ps.get("show_legend", True):
                layout_kwargs["showlegend"] = True
            fig.update_layout(**layout_kwargs)
            
            # Explicitly set backgrounds after all layout updates to override template
            fig.update_layout(
                plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"),
                paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF")
            )
            
            # Final check: explicitly ensure legend is visible
            if ps.get("show_legend", True) and len(fig.data) > 0:
                fig.update_layout(showlegend=True)
            
            # Determine tick format based on user preferences and normalization
            x_format_pref = ps.get("x_tick_format", "auto")
            x_decimals = ps.get("x_tick_decimals", 2)
            x_tick_format = get_tick_format(x_format_pref, x_decimals, normalize_x)
            
            y_format_pref = ps.get("y_tick_format", "auto")
            y_decimals = ps.get("y_tick_decimals", 2)
            y_tick_format = get_tick_format(y_format_pref, y_decimals, normalize_y)
            
            fig.update_xaxes(
                tickformat=x_tick_format,  # Format based on user preference and normalization
                separatethousands=False,  # Don't add thousand separators
            )
            fig.update_yaxes(
                tickformat=y_tick_format,  # Format based on user preference and normalization
                separatethousands=False,  # Don't add thousand separators
            )
            
            # Display plot - respect custom width if enabled
            use_container = not ps.get("enable_custom_size", False)
            col1, col2, col3 = st.columns([1, 10, 1])
            with col2:
                st.plotly_chart(fig, width='stretch' if use_container else 'content')
            capture_button(fig, title="Custom Multi-Trace Plot", source_page="Other Turbulence Stats")
            
            # Export panel
            export_panel(fig, data_dir, base_name="custom_multi_trace_plot")
    else:
        st.info("👆 Add traces above to create a multi-trace plot. Each trace can use different files and columns.")
    
    st.markdown("---")
    
    # =========================
    # Display Tables Section
    # =========================
    if table_data:
        # Display turbulence statistics tables
        turbulence_tables = {k: v for k, v in table_data.items() if v['type'] == 'turbulence_stats'}
        if turbulence_tables:
            st.header("Turbulence Statistics")
            
            if len(data_dirs) > 1:
                # Multiple directories: display each separately
                for key, table_info in turbulence_tables.items():
                    df_stats = table_info['df']
                    dir_name = table_info['dir_name']
                    
                    st.subheader(f"📁 {dir_name}")
                    
                    # Latest values table
                    st.markdown("**Latest Values:**")
                    latest = df_stats.iloc[-1]
                    latest_df = latest.to_frame().T
                    st.dataframe(latest_df, width='stretch')
                    capture_button(df=latest_df, title=f"Latest Values - {dir_name}", source_page="Other Turbulence Stats")
                    
                    # Full time series table
                    st.markdown("**Time Series Data:**")
                    st.dataframe(df_stats, width='stretch', height=300)
                    capture_button(df=df_stats, title=f"Time Series - {dir_name}", source_page="Other Turbulence Stats")
                    st.markdown("---")
            else:
                # Single directory: original behavior
                key = list(turbulence_tables.keys())[0]
                df_stats = turbulence_tables[key]['df']
                
                # Latest values table
                st.subheader("Latest Values")
                latest = df_stats.iloc[-1]
                latest_df = latest.to_frame().T
                st.dataframe(latest_df, width='stretch')
                capture_button(df=latest_df, title="Latest Values", source_page="Other Turbulence Stats")
                
                # Full time series table
                st.subheader("Time Series Data")
                st.dataframe(df_stats, width='stretch', height=400)
                capture_button(df=df_stats, title="Time Series Data", source_page="Other Turbulence Stats")
                
                st.markdown("---")

if __name__ == "__main__":
    main()
