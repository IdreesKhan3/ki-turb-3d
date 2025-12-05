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
from utils.plot_style import resolve_line_style, render_per_sim_style_ui, render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size
st.set_page_config(page_icon="⚫")

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
        # Load custom plot legend names and axis labels
        if 'custom_plot_legend_names' not in st.session_state:
            st.session_state.custom_plot_legend_names = meta.get("custom_plot_legend_names", {})
        else:
            st.session_state.custom_plot_legend_names.update(meta.get("custom_plot_legend_names", {}))
        if 'custom_plot_axis_labels' not in st.session_state:
            st.session_state.custom_plot_axis_labels = meta.get("custom_plot_axis_labels", {'x': 'X', 'y': 'Y'})
        else:
            st.session_state.custom_plot_axis_labels.update(meta.get("custom_plot_axis_labels", {}))
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
        "custom_plot_legend_names": st.session_state.get("custom_plot_legend_names", {}),
        "custom_plot_axis_labels": st.session_state.get("custom_plot_axis_labels", {'x': 'X', 'y': 'Y'}),
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
        "per_sim_style_energy": {},  # {sim: {enabled,color,width,dash,marker,msize}}
        "marker_size": 6,
        
        # Axis limits
        "enable_x_limits": False,
        "x_min": None,
        "x_max": None,
        "enable_y_limits": False,
        "y_min": None,
        "y_max": None,
        
        # Figure size
        "enable_custom_size": False,
        "figure_width": 800,
        "figure_height": 500,
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
        render_axis_limits_ui(ps, key_prefix="energy")
        st.markdown("---")
        render_figure_size_ui(ps, key_prefix="energy")
        st.markdown("---")
        render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_energy", 
                                key_prefix="energy", include_marker=True)

        st.markdown("---")
        b1, b2 = st.columns(2)
        reset_pressed = False
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
                reset_pressed = True
                st.rerun()  # Rerun to update sidebar with default values
    
    # Auto-save plot style changes (applies immediately) - but not if reset was pressed
    if not reset_pressed:
        st.session_state.plot_style = ps



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
    
    # Load persistent UI metadata (legends, axis labels, plot style)
    _load_ui_metadata(data_dir)
    
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

    # Collect all available dataframes (load data first, but don't display tables yet)
    all_dataframes = {}
    available_columns = {}
    table_data = {}  # Store table data for display later
    
    # Load turbulence_stats CSV from all directories
    csv_files = files.get('csv', [])
    
    if csv_files:
        if len(data_dirs) > 1:
            # Multiple directories: load all
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
                    table_data[key] = {'df': df_stats, 'dir_name': dir_name, 'type': 'turbulence_stats'}
                except Exception as e:
                    st.warning(f"Could not load {csv_path.name}: {e}")
                    continue
        else:
            # Single directory: original behavior
            try:
                df_stats = read_csv_data(str(files['csv'][0]))
                all_dataframes['turbulence_stats'] = df_stats
                available_columns['turbulence_stats'] = list(df_stats.columns)
                table_data['turbulence_stats'] = {'df': df_stats, 'dir_name': None, 'type': 'turbulence_stats'}
            except Exception as e:
                st.warning(f"Could not load turbulence stats: {e}")

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

    # Custom Plotting Section (display first, before tables)
    st.header("📈 Custom Plotting")
    
    if not all_dataframes:
        st.info("No CSV files found. Please load data from the Overview page.")
        return
    
    # Create sim_groups structure for plot style sidebar (extract unique prefixes from dataframe keys)
    sim_groups = {}
    for key in all_dataframes.keys():
        # Extract simulation prefix from key (e.g., "turbulence_stats_768" -> "768")
        if '_' in key:
            prefix = key.split('_')[-1]  # Get last part after underscore
        else:
            prefix = key
        if prefix not in sim_groups:
            sim_groups[prefix] = []  # Empty list is fine, just need the keys
    
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
    
    # Legend & Axis Labels (persistent)
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Trace Legend Names")
        if st.session_state.custom_plot_traces:
            for idx, trace in enumerate(st.session_state.custom_plot_traces):
                trace_key = f"{trace['data_source']}_{trace['x_col']}_{trace['y_col']}"
                default_name = trace.get('label', f"{trace['data_source'].split('_')[-1]}: {trace['y_col']}")
                st.session_state.custom_plot_legend_names.setdefault(trace_key, default_name)
                st.session_state.custom_plot_legend_names[trace_key] = st.text_input(
                    f"Trace {idx+1} label",
                    value=st.session_state.custom_plot_legend_names[trace_key],
                    key=f"custom_legend_{trace_key}"
                )
        else:
            st.caption("Add traces to customize legend names")
        
        st.markdown("---")
        st.markdown("### Axis Labels")
        st.session_state.custom_plot_axis_labels['x'] = st.text_input(
            "X-axis label",
            value=st.session_state.custom_plot_axis_labels.get('x', 'X'),
            key="custom_axis_x"
        )
        st.session_state.custom_plot_axis_labels['y'] = st.text_input(
            "Y-axis label",
            value=st.session_state.custom_plot_axis_labels.get('y', 'Y'),
            key="custom_axis_y"
        )
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save labels/legends", key="save_custom_labels"):
                _save_ui_metadata(data_dir)
                st.success("Saved to legend_names.json")
        with b2:
            if st.button("♻️ Reset labels/legends", key="reset_custom_labels"):
                st.session_state.custom_plot_legend_names = {}
                st.session_state.custom_plot_axis_labels = {
                    'x': 'X',
                    'y': 'Y'
                }
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")
                st.rerun()
    
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
        ps = st.session_state.get("plot_style", _default_plot_style())
        colors = _get_palette(ps)
        fig = go.Figure()
        
        all_x_labels = set()
        all_y_labels = set()
        
        for idx, trace in enumerate(st.session_state.custom_plot_traces):
            data_source = trace['data_source']
            x_col = trace['x_col']
            y_col = trace['y_col']
            # Use custom legend name if available, otherwise use trace label
            trace_key = f"{data_source}_{x_col}_{y_col}"
            label = st.session_state.custom_plot_legend_names.get(
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
            
            # Build hover template showing original column names and transformations
            hover_x_label = x_col
            if normalize_x:
                hover_x_label = f"{x_col} (normalized)"
            
            hover_y_label = y_col
            if use_abs:
                hover_y_label = f"|{hover_y_label}|"
            if normalize_y:
                hover_y_label = f"{hover_y_label} / max"
            
            color = colors[idx % len(colors)] if colors else None
            
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
                fig.add_trace(go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=label,
                    line=dict(width=ps.get("line_width", 2.2), color=color),
                    hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label_smooth}=%{{y:.4g}}<extra></extra>"
                ))
            else:
                # No smoothing - just plot original data
                x_plot, y_plot = x_data, y_data
                fig.add_trace(go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=label,
                    line=dict(width=ps.get("line_width", 2.2), color=color),
                    hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label}=%{{y:.4g}}<extra></extra>"
                ))
            
            all_x_labels.add(x_col)
            all_y_labels.add(y_col)
        
        if len(fig.data) == 0:
            st.warning("No valid traces to plot. Please add traces with valid data.")
        else:
            # Set axis labels - use custom labels from session state if available
            custom_x_label = st.session_state.custom_plot_axis_labels.get('x', None)
            custom_y_label = st.session_state.custom_plot_axis_labels.get('y', None)
            
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
            
            # Apply theme styling using full apply_plot_style function
            layout_kwargs = dict(
                title="Custom Multi-Trace Plot",
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,  # Default, will be overridden if custom size is enabled
                margin=dict(l=60, r=20, t=40, b=55),
            )
            layout_kwargs = apply_axis_limits(layout_kwargs, ps)
            layout_kwargs = apply_figure_size(layout_kwargs, ps)
            fig.update_layout(**layout_kwargs)
            
            # Apply full plot style (fonts, ticks, grids, backgrounds, etc.)
            fig = apply_plot_style(fig, ps)
            
            # Override y-axis tick format to show original values without SI prefixes
            fig.update_yaxes(
                tickformat=".10g",  # Show original values without automatic SI unit prefixes (no μ, m, k, etc.)
                separatethousands=False,  # Don't add thousand separators
            )
            
            st.plotly_chart(fig, width='stretch')
            capture_button(fig, title="Custom Multi-Trace Plot", source_page="Other Turbulence Stats")
            
            # Export panel
            export_panel(fig, data_dir, base_name="custom_multi_trace_plot")
    else:
        st.info("👆 Add traces above to create a multi-trace plot. Each trace can use different files and columns.")
    
    st.markdown("---")
    
    # Display Tables Section (after plotting)
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
        
        # Display eps validation tables if needed
        eps_tables = {k: v for k, v in table_data.items() if v['type'] == 'eps_validation'}
        if eps_tables and len(data_dirs) == 1:
            # Only show eps validation table for single directory (multi-dir handled in plotting)
            st.header("Energy Balance Validation")
            key = list(eps_tables.keys())[0]
            df_val = eps_tables[key]['df']
            st.dataframe(df_val, width='stretch', height=300)
            capture_button(df=df_val, title="Energy Balance Validation", source_page="Other Turbulence Stats")


if __name__ == "__main__":
    main()
