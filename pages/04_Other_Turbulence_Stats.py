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
    * PNG/PDF/SVG/EPS/JPG/WEBP/TIFF + HTML
- Robust to missing columns/files
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.colors import hex_to_rgb
from pathlib import Path
import sys
import re
import json
import glob

# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.csv_reader import read_csv_data
from utils.file_detector import detect_simulation_files
from utils.theme_config import apply_theme_to_plot_style, inject_theme_css, template_selector
from utils.report_builder import capture_button

# ==========================================================
# JSON persistence (dataset-local)
# ==========================================================
def _legend_json_path(data_dir: Path) -> Path:
    return data_dir / "legend_names.json"

def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()

def _load_ui_metadata(data_dir: Path):
    path = _legend_json_path(data_dir)
    if not path.exists():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        st.session_state.energy_legend_names = meta.get("energy_legends", {})
        st.session_state.axis_labels_energy = meta.get("axis_labels_energy", {})
        st.session_state.plot_style = meta.get("plot_style", st.session_state.get("plot_style", {}))
    except Exception:
        st.toast("legend_names.json exists but could not be read. Using defaults.", icon="⚠️")

def _save_ui_metadata(data_dir: Path):
    path = _legend_json_path(data_dir)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    old.update({
        "energy_legends": st.session_state.get("energy_legend_names", {}),
        "axis_labels_energy": st.session_state.get("axis_labels_energy", {}),
        "plot_style": st.session_state.get("plot_style", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json (read-only folder?): {e}")


# ==========================================================
# Plot styling (shared keys with other pages)
# ==========================================================
def _default_plot_style():
    return {
        "font_family": "Arial",
        "font_size": 14,
        "title_size": 16,
        "legend_size": 12,
        "tick_font_size": 12,
        "axis_title_size": 14,

        "tick_len": 6,
        "tick_w": 1.2,
        "ticks_outside": True,

        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",

        "show_grid": True,
        "grid_on_x": True,
        "grid_on_y": True,
        "grid_w": 0.6,
        "grid_dash": "dot",
        "grid_color": "#B0B0B0",
        "grid_opacity": 0.6,

        "show_minor_grid": False,
        "minor_grid_w": 0.4,
        "minor_grid_dash": "dot",
        "minor_grid_color": "#D0D0D0",
        "minor_grid_opacity": 0.45,

        "line_width": 2.2,

        "palette": "Plotly",
        "custom_colors": ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                          "#8c564b", "#e377c2", "#7f7f7f"],
        "template": "plotly_white",

        "enable_per_sim_style": False,
        "per_sim_style_energy": {},  # {sim: {enabled,color,width,dash}}
    }

def _get_palette(ps):
    if ps["palette"] == "Custom":
        cols = ps.get("custom_colors", [])
        return cols if cols else pc.qualitative.Plotly

    mapping = {
        "Plotly": pc.qualitative.Plotly,
        "D3": pc.qualitative.D3,
        "G10": pc.qualitative.G10,
        "T10": pc.qualitative.T10,
        "Dark2": pc.qualitative.Dark2,
        "Set1": pc.qualitative.Set1,
        "Set2": pc.qualitative.Set2,
        "Pastel1": pc.qualitative.Pastel1,
        "Bold": pc.qualitative.Bold,
        "Prism": pc.qualitative.Prism,
    }
    return mapping.get(ps["palette"], pc.qualitative.Plotly)

def _axis_title_font(ps):
    return dict(family=ps["font_family"], size=ps["axis_title_size"])

def _tick_font(ps):
    return dict(family=ps["font_family"], size=ps["tick_font_size"])

def apply_plot_style(fig, ps):
    fig.update_layout(
        template=ps["template"],
        font=dict(family=ps["font_family"], size=ps["font_size"]),
        legend=dict(font=dict(size=ps["legend_size"])),
        title=dict(font=dict(size=ps["title_size"])),
        hovermode="x unified",
        plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"),
        paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF"),
    )

    tick_dir = "outside" if ps["ticks_outside"] else "inside"

    show_x_grid = ps["show_grid"] and ps.get("grid_on_x", True)
    show_y_grid = ps["show_grid"] and ps.get("grid_on_y", True)
    grid_rgba = f"rgba{hex_to_rgb(ps['grid_color']) + (ps['grid_opacity'],)}"

    show_minor = ps.get("show_minor_grid", False)
    minor_rgba = f"rgba{hex_to_rgb(ps['minor_grid_color']) + (ps['minor_grid_opacity'],)}"

    fig.update_xaxes(
        ticks=tick_dir,
        ticklen=ps["tick_len"],
        tickwidth=ps["tick_w"],
        tickfont=_tick_font(ps),
        title_font=_axis_title_font(ps),
        showgrid=show_x_grid,
        gridwidth=ps["grid_w"],
        griddash=ps["grid_dash"],
        gridcolor=grid_rgba,
        minor=dict(
            showgrid=show_minor,
            gridwidth=ps["minor_grid_w"],
            griddash=ps["minor_grid_dash"],
            gridcolor=minor_rgba,
        )
    )
    fig.update_yaxes(
        ticks=tick_dir,
        ticklen=ps["tick_len"],
        tickwidth=ps["tick_w"],
        tickfont=_tick_font(ps),
        title_font=_axis_title_font(ps),
        showgrid=show_y_grid,
        gridwidth=ps["grid_w"],
        griddash=ps["grid_dash"],
        gridcolor=grid_rgba,
        minor=dict(
            showgrid=show_minor,
            gridwidth=ps["minor_grid_w"],
            griddash=ps["minor_grid_dash"],
            gridcolor=minor_rgba,
        )
    )
    return fig

def _ensure_per_sim_defaults(ps, sim_groups):
    ps.setdefault("per_sim_style_energy", {})
    for k in sim_groups.keys():
        ps["per_sim_style_energy"].setdefault(k, {
            "enabled": False,
            "color": None,
            "width": None,
            "dash": "solid",
        })

def plot_style_sidebar(data_dir: Path, sim_groups):
    ps = dict(st.session_state.plot_style)
    _ensure_per_sim_defaults(ps, sim_groups)

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        ps["font_family"] = st.selectbox("Font family", fonts, index=fonts.index(ps.get("font_family", "Arial")))
        ps["font_size"] = st.slider("Base/global font size", 8, 26, int(ps.get("font_size", 14)))
        ps["title_size"] = st.slider("Plot title size", 10, 32, int(ps.get("title_size", 16)))
        ps["legend_size"] = st.slider("Legend font size", 8, 24, int(ps.get("legend_size", 12)))
        ps["tick_font_size"] = st.slider("Tick label font size", 6, 24, int(ps.get("tick_font_size", 12)))
        ps["axis_title_size"] = st.slider("Axis title font size", 8, 28, int(ps.get("axis_title_size", 14)))

        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker("Plot background (inside axes)", ps.get("plot_bgcolor", "#FFFFFF"))
        ps["paper_bgcolor"] = st.color_picker("Paper background (outside axes)", ps.get("paper_bgcolor", "#FFFFFF"))

        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider("Tick length", 2, 14, int(ps.get("tick_len", 6)))
        ps["tick_w"] = st.slider("Tick width", 0.5, 3.5, float(ps.get("tick_w", 1.2)))
        ps["ticks_outside"] = st.checkbox("Ticks outside", bool(ps.get("ticks_outside", True)))

        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox("Show major grid", bool(ps.get("show_grid", True)))
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            ps["grid_on_x"] = st.checkbox("Grid on X", bool(ps.get("grid_on_x", True)))
        with gcol2:
            ps["grid_on_y"] = st.checkbox("Grid on Y", bool(ps.get("grid_on_y", True)))
        ps["grid_w"] = st.slider("Major grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)))
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        ps["grid_dash"] = st.selectbox("Major grid type", grid_styles,
                                       index=grid_styles.index(ps.get("grid_dash", "dot")))
        ps["grid_color"] = st.color_picker("Major grid color", ps.get("grid_color", "#B0B0B0"))
        ps["grid_opacity"] = st.slider("Major grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)))

        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox("Show minor grid", bool(ps.get("show_minor_grid", False)))
        ps["minor_grid_w"] = st.slider("Minor grid width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)))
        ps["minor_grid_dash"] = st.selectbox("Minor grid type", grid_styles,
                                             index=grid_styles.index(ps.get("minor_grid_dash", "dot")),
                                             key="minor_grid_dash_energy")
        ps["minor_grid_color"] = st.color_picker("Minor grid color", ps.get("minor_grid_color", "#D0D0D0"))
        ps["minor_grid_opacity"] = st.slider("Minor grid opacity", 0.0, 1.0,
                                             float(ps.get("minor_grid_opacity", 0.45)))

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.2)))

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        ps["palette"] = st.selectbox("Palette", palettes,
                                     index=palettes.index(ps.get("palette", "Plotly")))
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c, key=f"cust_color_energy_{i}"))
            ps["custom_colors"] = new_cols

        st.markdown("---")
        st.markdown("**Theme**")
        template_selector(ps)

        st.markdown("---")
        st.markdown("**Per-simulation overrides (optional)**")
        ps["enable_per_sim_style"] = st.checkbox("Enable per-simulation overrides",
                                                 bool(ps.get("enable_per_sim_style", False)))

        if ps["enable_per_sim_style"]:
            dash_opts = ["solid", "dot", "dash", "dashdot", "longdash", "longdashdot"]
            with st.container(border=True):
                for sim_prefix in sorted(sim_groups.keys()):
                    s = ps["per_sim_style_energy"][sim_prefix]
                    st.markdown(f"`{sim_prefix}`")
                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
                        s["enabled"] = st.checkbox("Override", value=s.get("enabled", False),
                                                   key=f"energy_over_on_{sim_prefix}")
                    with c2:
                        s["color"] = st.color_picker("Color", value=s.get("color") or "#000000",
                                                     key=f"energy_over_color_{sim_prefix}",
                                                     disabled=not s["enabled"])
                    with c3:
                        s["width"] = st.slider("Width", 0.5, 8.0,
                                               float(s.get("width") or ps["line_width"]),
                                               key=f"energy_over_width_{sim_prefix}",
                                               disabled=not s["enabled"])
                    s["dash"] = st.selectbox("Dash", dash_opts,
                                             index=dash_opts.index(s.get("dash") or "solid"),
                                             key=f"energy_over_dash_{sim_prefix}",
                                             disabled=not s["enabled"])

        st.markdown("---")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save Plot Style"):
                st.session_state.plot_style = ps
                _save_ui_metadata(data_dir)
                st.success("Saved plot style.")
        with b2:
            if st.button("♻️ Reset Plot Style"):
                st.session_state.plot_style = _default_plot_style()
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")

    st.session_state.plot_style = ps

def _resolve_line_style(sim_prefix, idx, colors, ps):
    default_color = colors[idx % len(colors)]
    default_width = ps["line_width"]
    default_dash = "solid"

    if not ps.get("enable_per_sim_style", False):
        return default_color, default_width, default_dash

    s = ps.get("per_sim_style_energy", {}).get(sim_prefix, {})
    if not s.get("enabled", False):
        return default_color, default_width, default_dash

    color = s.get("color") or default_color
    width = float(s.get("width") or default_width)
    dash = s.get("dash") or default_dash
    return color, width, dash


# ==========================================================
# Export (research formats)
# ==========================================================
_EXPORT_FORMATS = {
    "PNG (raster)": "png",
    "PDF (vector)": "pdf",
    "SVG (vector)": "svg",
    "EPS (vector)": "eps",
    "JPG/JPEG (raster)": "jpg",
    "WEBP (raster)": "webp",
    "TIFF (raster)": "tiff",
    "HTML (interactive)": "html",
}

def export_panel(fig, out_dir: Path, base_name: str):
    with st.expander(f"📤 Export figure: {base_name}", expanded=False):
        fmts = st.multiselect(
            "Select export format(s)",
            list(_EXPORT_FORMATS.keys()),
            default=["PNG (raster)", "PDF (vector)", "SVG (vector)"],
            key=f"{base_name}_fmts"
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            scale = st.slider("Scale (DPI-like)", 1.0, 6.0, 3.0, 0.5, key=f"{base_name}_scale")
        with c2:
            width_px = st.number_input("Width px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_wpx")
        with c3:
            height_px = st.number_input("Height px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_hpx")

        if st.button("Export selected formats", key=f"{base_name}_doexport"):
            if not fmts:
                st.warning("Please select at least one format.")
                return

            errors = []
            for f_label in fmts:
                ext = _EXPORT_FORMATS[f_label]
                out = out_dir / f"{base_name}.{ext}"
                try:
                    if ext == "html":
                        fig.write_html(str(out))
                        continue

                    kwargs = {}
                    if width_px > 0:
                        kwargs["width"] = int(width_px)
                    if height_px > 0:
                        kwargs["height"] = int(height_px)

                    fig.write_image(str(out), scale=scale, **kwargs)
                except Exception as e:
                    errors.append((out.name, str(e)))

            if errors:
                st.error(
                    "Some exports failed. Ensure kaleido is installed:\n"
                    "pip install -U kaleido\n\n"
                    + "\n".join([f"- {n}: {msg}" for n, msg in errors])
                )
            else:
                st.success("All selected exports saved to dataset folder.")


# ==========================================================
# Page main
# ==========================================================
def main():
    inject_theme_css()
    
    st.title("📊 Other Turbulence Stats")

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
            project_root = Path(__file__).parent.parent
            for i, data_dir_path in enumerate(data_dirs, 1):
                data_dir_obj = Path(data_dir_path)
                try:
                    rel_path = data_dir_obj.relative_to(project_root)
                    st.markdown(f"**{i}.** `APP/{rel_path}`")
                except ValueError:
                    st.markdown(f"**{i}.** `{data_dir_path}`")
        st.markdown("---")
    
    # Apply theme to plot style on page load
    current_theme = st.session_state.get("theme", "Light Scientific")
    if 'plot_style' not in st.session_state:
        st.session_state.plot_style = _default_plot_style()
    
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

    # Collect all available dataframes
    all_dataframes = {}
    available_columns = {}
    
    # Load turbulence_stats CSV from all directories
    csv_files = files.get('csv', [])
    
    # Debug info (temporary - can help diagnose)
    if len(data_dirs) > 1:
        if csv_files:
            st.caption(f"🔍 Debug: Found {len(csv_files)} turbulence_stats CSV files")
        else:
            st.caption(f"⚠️ Debug: No turbulence_stats*.csv files found in {len(data_dirs)} directories")
    
    if csv_files:
        if len(data_dirs) > 1:
            # Multiple directories: load and display each separately
            st.header("Turbulence Statistics")
            
            for csv_file in csv_files:
                csv_path = Path(csv_file).resolve()  # Make absolute
                # Find which directory this file belongs to
                dir_name = None
                for data_dir_path in data_dirs:
                    data_dir_obj = Path(data_dir_path).resolve()  # Make absolute
                    try:
                        if csv_path.is_relative_to(data_dir_obj):
                            dir_name = Path(data_dir_path).name
                            break
                    except (ValueError, AttributeError):
                        # Try string comparison as fallback
                        csv_str = str(csv_path)
                        dir_str = str(data_dir_obj)
                        if csv_str.startswith(dir_str):
                            dir_name = Path(data_dir_path).name
                            break
                
                if not dir_name:
                    dir_name = csv_path.parent.name
                
                try:
                    df_stats = read_csv_data(str(csv_file))
                    key = f"turbulence_stats_{dir_name}"
                    all_dataframes[key] = df_stats
                    available_columns[key] = list(df_stats.columns)
                    
                    st.subheader(f"📁 {dir_name}")
                    
                    # Latest values table
                    st.markdown("**Latest Values:**")
                    latest = df_stats.iloc[-1]
                    latest_df = latest.to_frame().T
                    st.dataframe(latest_df, use_container_width=True)
                    capture_button(df=latest_df, title=f"Latest Values - {dir_name}", source_page="Other Turbulence Stats")
                    
                    # Full time series table
                    st.markdown("**Time Series Data:**")
                    st.dataframe(df_stats, use_container_width=True, height=300)
                    capture_button(df=df_stats, title=f"Time Series - {dir_name}", source_page="Other Turbulence Stats")
                    st.markdown("---")
                except Exception as e:
                    st.warning(f"Could not load {csv_path.name}: {e}")
                    continue
        else:
            # Single directory: original behavior
            df_stats = read_csv_data(str(files['csv'][0]))
            all_dataframes['turbulence_stats'] = df_stats
            available_columns['turbulence_stats'] = list(df_stats.columns)
            
            st.header("Turbulence Statistics")
            
            # Latest values table
            st.subheader("Latest Values")
            latest = df_stats.iloc[-1]
            latest_df = latest.to_frame().T
            st.dataframe(latest_df, use_container_width=True)
            capture_button(df=latest_df, title="Latest Values", source_page="Other Turbulence Stats")
            
            # Full time series table
            st.subheader("Time Series Data")
            st.dataframe(df_stats, use_container_width=True, height=400)
            capture_button(df=df_stats, title="Time Series Data", source_page="Other Turbulence Stats")
            
            st.markdown("---")

    # Load eps_real_validation CSV files from all directories
    eps_files = files.get("eps_validation", [])
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
                    try:
                        if eps_path.is_relative_to(data_dir_obj):
                            dir_name = Path(data_dir_path).name
                            break
                    except (ValueError, AttributeError):
                        # Try string comparison as fallback
                        eps_str = str(eps_path)
                        dir_str = str(data_dir_obj)
                        if eps_str.startswith(dir_str):
                            dir_name = Path(data_dir_path).name
                            break
                
                if not dir_name:
                    dir_name = eps_path.parent.name
                
                try:
                    df_val = pd.read_csv(str(eps_file))
                    key = f"eps_validation_{dir_name}"
                    all_dataframes[key] = df_val
                    available_columns[key] = list(df_val.columns)
                except Exception as e:
                    st.warning(f"Could not load {eps_path.name} from {dir_name}: {e}")
                    continue
        else:
            # Single directory: original behavior
            try:
                df_val = pd.read_csv(str(eps_files[0]))
                all_dataframes['eps_validation'] = df_val
                available_columns['eps_validation'] = list(df_val.columns)
            except Exception:
                pass

    # Custom Plotting Section
    st.header("📈 Custom Plotting")
    
    if not all_dataframes:
        st.info("No CSV files found. Please load data from the Overview page.")
        return
    
    # Column selection in main area (more visible)
    st.subheader("Select Columns to Plot")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Select data source
        data_source = st.selectbox(
            "📁 Data Source",
            options=list(all_dataframes.keys()),
            help="Select which CSV file to plot from",
            key="plot_data_source"
        )
    
    df_plot = all_dataframes[data_source]
    numeric_cols = [col for col in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[col])]
    
    if len(numeric_cols) < 2:
        st.warning(f"Not enough numeric columns in {data_source} for plotting.")
        return
    
    with col2:
        # X-axis column selection
        x_col = st.selectbox(
            "X-axis Column",
            options=numeric_cols,
            index=0 if 'iter' in numeric_cols else 0,
            help="Select column for X-axis (e.g., iter, time)",
            key="plot_x_col"
        )
    
    with col3:
        # Y-axis column selection
        y_col = st.selectbox(
            "Y-axis Column",
            options=[c for c in numeric_cols if c != x_col],
            index=0 if 'TKE' in [c for c in numeric_cols if c != x_col] else 0,
            help="Select column for Y-axis (e.g., TKE, u_rms)",
            key="plot_y_col"
        )
    
    st.markdown("---")
    
    # Sidebar: Plot options
    st.sidebar.subheader("Plot Options")
    use_abs = st.sidebar.checkbox("Use absolute value (Y-axis)", value=False)
    smooth_window = st.sidebar.slider("Moving average window (0=off)", 0, 500, 0, 10)
    
    # Normalization
    normalize_x = st.sidebar.checkbox("Normalize X-axis", value=False)
    x_norm = st.sidebar.number_input("X normalization constant", value=1000.0, min_value=1.0, step=100.0, disabled=not normalize_x)
    
    # Create plot
    x_data = df_plot[x_col].values
    y_data = df_plot[y_col].values
    
    # Convert to numeric and remove NaN
    x_data = pd.to_numeric(x_data, errors='coerce')
    y_data = pd.to_numeric(y_data, errors='coerce')
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    if len(x_data) == 0 or len(y_data) == 0:
        st.warning("No valid data points to plot.")
        return
    
    # Apply normalization
    if normalize_x:
        x_data = x_data / float(x_norm)
    
    # Apply absolute value
    if use_abs:
        y_data = np.abs(y_data)
    
    # Apply smoothing
    if smooth_window > 1 and len(y_data) > smooth_window:
        kernel = np.ones(int(smooth_window)) / int(smooth_window)
        y_smooth = np.convolve(y_data, kernel, mode="valid")
        x_smooth = x_data[int(smooth_window)//2: int(smooth_window)//2 + len(y_smooth)]
        x_plot, y_plot = x_smooth, y_smooth
    else:
        x_plot, y_plot = x_data, y_data
    
    # Create figure with theme styling
    ps = st.session_state.get("plot_style", _default_plot_style())
    colors = _get_palette(ps)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        mode="lines",
        name=f"{y_col} vs {x_col}",
        line=dict(width=ps.get("line_width", 2.2), color=colors[0] if colors else "#1f77b4"),
        hovertemplate=f"{x_col}=%{{x:.4g}}<br>{y_col}=%{{y:.4g}}<extra></extra>"
    ))
    
    x_label = f"{x_col}" + (f" / {x_norm}" if normalize_x else "")
    y_label = f"{y_col}" + (" (abs)" if use_abs else "")
    
    # Apply theme styling
    fig.update_layout(
        title=f"{y_col} vs {x_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        margin=dict(l=60, r=20, t=40, b=55),
        hovermode="x unified",
        template=ps.get("template", "plotly_white"),
        font=dict(family=ps.get("font_family", "Arial"), size=ps.get("font_size", 14)),
        plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"),
        paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF"),
    )
    
    # Apply grid styling from theme
    from plotly.colors import hex_to_rgb
    grid_color = ps.get("grid_color", "#B0B0B0")
    grid_opacity = ps.get("grid_opacity", 0.6)
    grid_rgba = f"rgba{hex_to_rgb(grid_color) + (grid_opacity,)}"
    
    fig.update_xaxes(
        showgrid=ps.get("show_grid", True) and ps.get("grid_on_x", True),
        gridcolor=grid_rgba,
        gridwidth=ps.get("grid_w", 0.6),
    )
    fig.update_yaxes(
        showgrid=ps.get("show_grid", True) and ps.get("grid_on_y", True),
        gridcolor=grid_rgba,
        gridwidth=ps.get("grid_w", 0.6),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    capture_button(fig, title=f"Custom Plot: {y_col} vs {x_col}", source_page="Other Turbulence Stats")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{np.mean(y_plot):.6f}")
    with col2:
        st.metric("Std Dev", f"{np.std(y_plot):.6f}")
    with col3:
        st.metric("Min", f"{np.min(y_plot):.6f}")
    with col4:
        st.metric("Max", f"{np.max(y_plot):.6f}")
    
    st.markdown("---")
    
    # Export panel for custom plot
    export_panel(fig, data_dir, base_name=f"custom_plot_{x_col}_vs_{y_col}")


if __name__ == "__main__":
    main()
