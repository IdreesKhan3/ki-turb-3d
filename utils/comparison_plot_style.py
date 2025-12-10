"""
Plot style system for Comparison page
Helper module for plot styling and persistence
"""

import streamlit as st
import json
from pathlib import Path
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    _get_palette, _normalize_plot_name, ensure_per_sim_defaults,
    render_per_sim_style_ui, render_axis_limits_ui, render_figure_size_ui,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    render_plot_title_ui, plot_style_sidebar as shared_plot_style_sidebar
)
from utils.theme_config import apply_theme_to_plot_style
from utils.export_figs import export_panel as _export_panel

# Re-export for backward compatibility
export_panel = _export_panel


# ==========================================================
# JSON persistence
# ==========================================================
def _legend_json_path(data_dir: Path) -> Path:
    return data_dir / "legend_names.json"

def _load_ui_metadata(data_dir: Path):
    """Load plot styles from legend_names.json."""
    path = _legend_json_path(data_dir)
    if not path.exists():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        st.session_state.plot_styles = meta.get("plot_styles", {})
    except Exception:
        st.toast("legend_names.json exists but could not be read. Using defaults.", icon="⚠️")

def _save_ui_metadata(data_dir: Path):
    """Merge-save UI metadata without clobbering other pages."""
    path = _legend_json_path(data_dir)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    old.update({
        "plot_styles": st.session_state.get("plot_styles", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json (read-only folder?): {e}")


# ==========================================================
# Plot styling system
# ==========================================================
def _normalize_plot_name_local(plot_name: str) -> str:
    """Normalize plot name to a valid key format (handles special characters)."""
    cleaned = plot_name.replace("ₚ", "p").replace("ξ", "xi").replace("/", "_")
    return _normalize_plot_name(cleaned)

def apply_plot_style(fig, ps):
    """Apply plot style with margin handling."""
    fig = apply_plot_style_base(fig, ps)
    
    # Handle margins if specified
    if any(k in ps for k in ["margin_left", "margin_right", "margin_top", "margin_bottom"]):
        fig.update_layout(margin=dict(
            l=ps.get("margin_left", 50),
            r=ps.get("margin_right", 20),
            t=ps.get("margin_top", 30),
            b=ps.get("margin_bottom", 50)
        ))
    
    return fig

def get_plot_style(plot_name: str):
    """Get plot-specific style, merging defaults with plot-specific overrides."""
    default = default_plot_style()
    # Add page-specific defaults
    default.update({
        "line_width": 2.4,
        "margin_left": 50,
        "margin_right": 20,
        "margin_top": 30,
        "margin_bottom": 50,
        "per_sim_style_comparison": {},
    })
    
    # Set plot-specific defaults
    if plot_name == "R-Q Topological Space":
        default.update({
            "y_tick_format": "float",
            "y_tick_decimals": 1,
        })
    
    plot_styles = st.session_state.get("plot_styles", {})
    plot_style = plot_styles.get(plot_name, {})
    merged = default.copy()
    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value
    
    # Apply theme to ensure font_color is set correctly
    current_theme = st.session_state.get("theme", "Light Scientific")
    merged = apply_theme_to_plot_style(merged, current_theme)
    
    return merged

def plot_style_sidebar(data_dir: Path, file_list, plot_names: list):
    """Plot style configuration sidebar for Comparison page using shared module components."""
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure",
        plot_names,
        key="comparison_plot_selector"
    )
    
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    if selected_plot not in st.session_state.plot_styles:
        st.session_state.plot_styles[selected_plot] = {}
    
    ps = get_plot_style(selected_plot)
    plot_key = _normalize_plot_name_local(selected_plot)
    
    # Ensure per-file defaults (files are treated as sim_groups)
    file_groups = {f: f for f in file_list}
    ensure_per_sim_defaults(ps, file_groups, style_key="per_sim_style_comparison", include_marker=True)
    
    key_prefix = f"comparison_{plot_key}"
    
    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        font_idx = fonts.index(ps.get("font_family", "Arial")) if ps.get("font_family", "Arial") in fonts else 0
        ps["font_family"] = st.selectbox("Font family", fonts, index=font_idx, key=f"{key_prefix}_font_family")
        ps["font_size"] = st.slider("Base/global font size", 8, 26, int(ps.get("font_size", 14)), key=f"{key_prefix}_font_size")
        ps["title_size"] = st.slider("Plot title size", 10, 32, int(ps.get("title_size", 16)), key=f"{key_prefix}_title_size")
        ps["legend_size"] = st.slider("Legend font size", 8, 24, int(ps.get("legend_size", 12)), key=f"{key_prefix}_legend_size")
        ps["tick_font_size"] = st.slider("Tick label font size", 6, 24, int(ps.get("tick_font_size", 12)), key=f"{key_prefix}_tick_font_size")
        ps["axis_title_size"] = st.slider("Axis title font size", 8, 28, int(ps.get("axis_title_size", 14)), key=f"{key_prefix}_axis_title_size")
        
        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker("Plot background (inside axes)", ps.get("plot_bgcolor", "#FFFFFF"), key=f"{key_prefix}_plot_bgcolor")
        ps["paper_bgcolor"] = st.color_picker("Paper background (outside axes)", ps.get("paper_bgcolor", "#FFFFFF"), key=f"{key_prefix}_paper_bgcolor")
        
        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider("Tick length", 2, 14, int(ps.get("tick_len", 6)), key=f"{key_prefix}_tick_len")
        ps["tick_w"] = st.slider("Tick width", 0.5, 3.5, float(ps.get("tick_w", 1.2)), key=f"{key_prefix}_tick_w")
        ps["ticks_outside"] = st.checkbox("Ticks outside", bool(ps.get("ticks_outside", True)), key=f"{key_prefix}_ticks_outside")
        
        st.markdown("---")
        render_axis_scale_ui(ps, key_prefix=key_prefix)
        
        st.markdown("---")
        render_tick_format_ui(ps, key_prefix=key_prefix)
        
        st.markdown("---")
        render_axis_borders_ui(ps, key_prefix=key_prefix)
        
        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox("Show major grid", bool(ps.get("show_grid", True)), key=f"{key_prefix}_show_grid")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            ps["grid_on_x"] = st.checkbox("Grid on X", bool(ps.get("grid_on_x", True)), key=f"{key_prefix}_grid_on_x")
        with gcol2:
            ps["grid_on_y"] = st.checkbox("Grid on Y", bool(ps.get("grid_on_y", True)), key=f"{key_prefix}_grid_on_y")
        ps["grid_w"] = st.slider("Major grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)), key=f"{key_prefix}_grid_w")
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        grid_dash_idx = grid_styles.index(ps.get("grid_dash", "dot")) if ps.get("grid_dash", "dot") in grid_styles else 1
        ps["grid_dash"] = st.selectbox("Major grid type", grid_styles, index=grid_dash_idx, key=f"{key_prefix}_grid_dash")
        ps["grid_color"] = st.color_picker("Major grid color", ps.get("grid_color", "#B0B0B0"), key=f"{key_prefix}_grid_color")
        ps["grid_opacity"] = st.slider("Major grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)), key=f"{key_prefix}_grid_opacity")
        
        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox("Show minor grid", bool(ps.get("show_minor_grid", False)), key=f"{key_prefix}_show_minor_grid")
        ps["minor_grid_w"] = st.slider("Minor grid width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)), key=f"{key_prefix}_minor_grid_w")
        minor_grid_dash_idx = grid_styles.index(ps.get("minor_grid_dash", "dot")) if ps.get("minor_grid_dash", "dot") in grid_styles else 1
        ps["minor_grid_dash"] = st.selectbox("Minor grid type", grid_styles, index=minor_grid_dash_idx, key=f"{key_prefix}_minor_grid_dash")
        ps["minor_grid_color"] = st.color_picker("Minor grid color", ps.get("minor_grid_color", "#D0D0D0"), key=f"{key_prefix}_minor_grid_color")
        ps["minor_grid_opacity"] = st.slider("Minor grid opacity", 0.0, 1.0, float(ps.get("minor_grid_opacity", 0.45)), key=f"{key_prefix}_minor_grid_opacity")
        
        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.4)), key=f"{key_prefix}_line_width")
        ps["marker_size"] = st.slider("Global marker size", 0, 18, int(ps.get("marker_size", 6)), key=f"{key_prefix}_marker_size")
        
        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2", "Pastel1", "Bold", "Prism", "Custom"]
        palette_idx = palettes.index(ps.get("palette", "Plotly")) if ps.get("palette", "Plotly") in palettes else 0
        ps["palette"] = st.selectbox("Palette", palettes, index=palette_idx, key=f"{key_prefix}_palette")
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c, key=f"{key_prefix}_cust_color_{i}"))
            ps["custom_colors"] = new_cols
        
        st.markdown("---")
        st.markdown("**Theme**")
        old_template = ps.get("template", "plotly_white")
        templates = ["plotly_white", "simple_white", "plotly_dark"]
        ps["template"] = st.selectbox("Template", templates, index=templates.index(old_template) if old_template in templates else 0, key=f"{key_prefix}_template")
        if ps["template"] != old_template:
            if ps["template"] == "plotly_dark":
                ps["plot_bgcolor"] = "#1e1e1e"
                ps["paper_bgcolor"] = "#1e1e1e"
            else:
                ps["plot_bgcolor"] = "#FFFFFF"
                ps["paper_bgcolor"] = "#FFFFFF"
        
        st.markdown("---")
        render_plot_title_ui(ps, key_prefix=key_prefix)
        
        st.markdown("---")
        render_axis_limits_ui(ps, key_prefix=key_prefix)
        
        st.markdown("---")
        render_figure_size_ui(ps, key_prefix=key_prefix)
        
        st.markdown("---")
        st.markdown("**Frame/Margin Size**")
        col1, col2 = st.columns(2)
        with col1:
            ps["margin_left"] = st.number_input("Left margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_left", 50)), step=5, key=f"{key_prefix}_margin_left")
            ps["margin_top"] = st.number_input("Top margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_top", 30)), step=5, key=f"{key_prefix}_margin_top")
        with col2:
            ps["margin_right"] = st.number_input("Right margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_right", 20)), step=5, key=f"{key_prefix}_margin_right")
            ps["margin_bottom"] = st.number_input("Bottom margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_bottom", 50)), step=5, key=f"{key_prefix}_margin_bottom")
        
        st.markdown("---")
        render_per_sim_style_ui(ps, file_groups, style_key="per_sim_style_comparison", key_prefix=f"{key_prefix}_file", include_marker=True, show_enable_checkbox=True)
        
        st.markdown("---")
        b1, b2 = st.columns(2)
        reset_pressed = False
        with b1:
            if st.button("💾 Save Plot Style", key=f"{key_prefix}_save"):
                st.session_state.plot_styles[selected_plot] = ps
                _save_ui_metadata(data_dir)
                st.success(f"Saved style for '{selected_plot}'.")
        with b2:
            if st.button("♻️ Reset Plot Style", key=f"{key_prefix}_reset"):
                st.session_state.plot_styles[selected_plot] = {}
                
                # Clear widget state
                widget_keys = [
                    f"{key_prefix}_font_family", f"{key_prefix}_font_size", f"{key_prefix}_title_size",
                    f"{key_prefix}_legend_size", f"{key_prefix}_tick_font_size", f"{key_prefix}_axis_title_size",
                    f"{key_prefix}_plot_bgcolor", f"{key_prefix}_paper_bgcolor",
                    f"{key_prefix}_tick_len", f"{key_prefix}_tick_w", f"{key_prefix}_ticks_outside",
                    f"{key_prefix}_x_axis_type", f"{key_prefix}_y_axis_type",
                    f"{key_prefix}_x_tick_format", f"{key_prefix}_x_tick_decimals", f"{key_prefix}_y_tick_format", f"{key_prefix}_y_tick_decimals",
                    f"{key_prefix}_show_axis_lines", f"{key_prefix}_axis_line_width", f"{key_prefix}_axis_line_color", f"{key_prefix}_mirror_axes",
                    f"{key_prefix}_show_grid", f"{key_prefix}_grid_on_x", f"{key_prefix}_grid_on_y", f"{key_prefix}_grid_w", f"{key_prefix}_grid_dash",
                    f"{key_prefix}_grid_color", f"{key_prefix}_grid_opacity",
                    f"{key_prefix}_show_minor_grid", f"{key_prefix}_minor_grid_w", f"{key_prefix}_minor_grid_dash", f"{key_prefix}_minor_grid_color", f"{key_prefix}_minor_grid_opacity",
                    f"{key_prefix}_line_width", f"{key_prefix}_marker_size",
                    f"{key_prefix}_palette", f"{key_prefix}_template",
                    f"{key_prefix}_show_plot_title", f"{key_prefix}_plot_title",
                    f"{key_prefix}_enable_x_limits", f"{key_prefix}_x_min", f"{key_prefix}_x_max",
                    f"{key_prefix}_enable_y_limits", f"{key_prefix}_y_min", f"{key_prefix}_y_max",
                    f"{key_prefix}_enable_custom_size", f"{key_prefix}_figure_width", f"{key_prefix}_figure_height",
                    f"{key_prefix}_margin_left", f"{key_prefix}_margin_right", f"{key_prefix}_margin_top", f"{key_prefix}_margin_bottom",
                    f"{key_prefix}_file_enable_per_sim",
                ]
                for i in range(10):
                    widget_keys.append(f"{key_prefix}_cust_color_{i}")
                for filename in file_list:
                    for suffix in ["over_on", "over_color", "over_width", "over_dash", "over_marker", "over_msize"]:
                        widget_keys.append(f"{key_prefix}_file_{suffix}_{filename}")
                
                for k in widget_keys:
                    if k in st.session_state:
                        del st.session_state[k]
                
                _save_ui_metadata(data_dir)
                st.toast(f"Reset style for '{selected_plot}'.", icon="♻️")
                reset_pressed = True
                st.rerun()
    
    if not reset_pressed:
        ps_copy = ps.copy()
        for key, value in ps.items():
            if isinstance(value, dict):
                ps_copy[key] = value.copy()
        st.session_state.plot_styles[selected_plot] = ps_copy


# ==========================================================
# Export system - now uses shared export_figs module
# ==========================================================
# export_panel is imported and re-exported above for backward compatibility

