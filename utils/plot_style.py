"""
Centralized plot style utilities for per-simulation marker and line style management.

Provides:
- All Plotly-supported marker and line style options
- Unified resolve_line_style function
- Reusable UI component for per-simulation styling
- Comprehensive plot style sidebar with log scale, tick format, axis borders
- Plot style application functions
"""

import streamlit as st
import plotly.colors as pc
from plotly.colors import hex_to_rgb

# All Plotly-supported line dash styles
PLOTLY_LINE_STYLES = [
    "solid",
    "dot",
    "dash",
    "longdash",
    "dashdot",
    "longdashdot",
]

# All Plotly-supported marker styles
PLOTLY_MARKER_STYLES = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "triangle-ne",
    "triangle-se",
    "triangle-sw",
    "triangle-nw",
    "pentagon",
    "hexagon",
    "hexagon2",
    "octagon",
    "star",
    "hexagram",
    "star-triangle-up",
    "star-triangle-down",
    "star-square",
    "star-diamond",
    "diamond-tall",
    "diamond-wide",
    "hourglass",
    "bowtie",
    "circle-cross",
    "circle-x",
    "square-cross",
    "square-x",
    "diamond-cross",
    "diamond-x",
    "x-thin",
    "cross-thin",
    "asterisk",
    "hash",
    "y-up",
    "y-down",
    "y-left",
    "y-right",
    "line-ew",
    "line-ns",
    "line-ne",
    "line-nw",
]


def resolve_line_style(sim_prefix, idx, colors, ps, style_key="per_sim_style", 
                       include_marker=False, default_marker="circle"):
    """
    Resolve line style for a simulation from plot style configuration.
    
    Args:
        sim_prefix: Simulation prefix/identifier
        idx: Index for default color selection
        colors: List of default colors
        ps: Plot style dictionary
        style_key: Key in ps dict for per-simulation styles (e.g., "per_sim_style_energy")
        include_marker: Whether to return marker and marker size
        default_marker: Default marker style if not specified
    
    Returns:
        If include_marker=False: (color, width, dash)
        If include_marker=True: (color, width, dash, marker, msize, override_on)
    """
    default_color = colors[idx % len(colors)]
    default_width = ps.get("line_width", 2.0)
    default_dash = "solid"
    default_marker = default_marker
    default_msize = ps.get("marker_size", 6)

    if not ps.get("enable_per_sim_style", False):
        if include_marker:
            return default_color, default_width, default_dash, default_marker, default_msize, False
        return default_color, default_width, default_dash

    s = ps.get(style_key, {}).get(sim_prefix, {})
    if not s.get("enabled", False):
        if include_marker:
            return default_color, default_width, default_dash, default_marker, default_msize, False
        return default_color, default_width, default_dash

    color = s.get("color") or default_color
    width = float(s.get("width") or default_width)
    dash = s.get("dash") or default_dash
    
    if include_marker:
        marker = s.get("marker") or default_marker
        msize = int(s.get("msize") or default_msize)
        return color, width, dash, marker, msize, True
    
    return color, width, dash


def ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style", include_marker=False):
    """
    Ensure per-simulation style defaults exist in plot style dict.
    
    Args:
        ps: Plot style dictionary (modified in place)
        sim_groups: Dictionary of simulation groups (keys are sim_prefixes)
        style_key: Key in ps dict for per-simulation styles
        include_marker: Whether to include marker fields in defaults
    """
    ps.setdefault(style_key, {})
    for k in sim_groups.keys():
        default_style = {
            "enabled": False,
            "color": None,
            "width": None,
            "dash": "solid",
        }
        if include_marker:
            default_style["marker"] = "circle"
            default_style["msize"] = None
        ps[style_key].setdefault(k, default_style)


def render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style", 
                             key_prefix="", include_marker=False, show_enable_checkbox=True):
    """
    Render UI controls for per-simulation styling.
    
    Args:
        ps: Plot style dictionary
        sim_groups: Dictionary of simulation groups (keys are sim_prefixes)
        style_key: Key in ps dict for per-simulation styles
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
        include_marker: Whether to show marker controls
        show_enable_checkbox: Whether to show the "Enable per-simulation overrides" checkbox
    
    Returns:
        Updated ps dictionary (also modifies in place)
    """
    ensure_per_sim_defaults(ps, sim_groups, style_key, include_marker)
    
    if show_enable_checkbox:
        st.markdown("**Per-simulation line styles (optional)**")
        ps["enable_per_sim_style"] = st.checkbox(
            "Enable per-simulation overrides",
            bool(ps.get("enable_per_sim_style", False)),
            key=f"{key_prefix}_enable_per_sim"
        )

    if ps["enable_per_sim_style"]:
        with st.container(border=True):
            for sim_prefix in sorted(sim_groups.keys()):
                s = ps[style_key][sim_prefix]
                st.markdown(f"`{sim_prefix}`")
                
                if include_marker:
                    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
                    with c1:
                        s["enabled"] = st.checkbox(
                            "Override",
                            value=s.get("enabled", False),
                            key=f"{key_prefix}_over_on_{sim_prefix}"
                        )
                    with c2:
                        s["color"] = st.color_picker(
                            "Color",
                            value=s.get("color") or "#000000",
                            key=f"{key_prefix}_over_color_{sim_prefix}",
                            disabled=not s["enabled"]
                        )
                    with c3:
                        s["width"] = st.slider(
                            "Width",
                            0.5, 8.0,
                            float(s.get("width") or ps.get("line_width", 2.0)),
                            key=f"{key_prefix}_over_width_{sim_prefix}",
                            disabled=not s["enabled"]
                        )
                    with c4:
                        s["dash"] = st.selectbox(
                            "Dash",
                            PLOTLY_LINE_STYLES,
                            index=PLOTLY_LINE_STYLES.index(s.get("dash") or "solid"),
                            key=f"{key_prefix}_over_dash_{sim_prefix}",
                            disabled=not s["enabled"]
                        )
                    with c5:
                        s["marker"] = st.selectbox(
                            "Marker",
                            PLOTLY_MARKER_STYLES,
                            index=PLOTLY_MARKER_STYLES.index(s.get("marker") or "circle"),
                            key=f"{key_prefix}_over_marker_{sim_prefix}",
                            disabled=not s["enabled"]
                        )
                    s["msize"] = st.slider(
                        "Marker size",
                        0, 18,
                        int(s.get("msize") or ps.get("marker_size", 6)),
                        key=f"{key_prefix}_over_msize_{sim_prefix}",
                        disabled=not s["enabled"]
                    )
                else:
                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
                        s["enabled"] = st.checkbox(
                            "Override",
                            value=s.get("enabled", False),
                            key=f"{key_prefix}_over_on_{sim_prefix}"
                        )
                    with c2:
                        s["color"] = st.color_picker(
                            "Color",
                            value=s.get("color") or "#000000",
                            key=f"{key_prefix}_over_color_{sim_prefix}",
                            disabled=not s["enabled"]
                        )
                    with c3:
                        s["width"] = st.slider(
                            "Width",
                            0.5, 8.0,
                            float(s.get("width") or ps.get("line_width", 2.0)),
                            key=f"{key_prefix}_over_width_{sim_prefix}",
                            disabled=not s["enabled"]
                        )
                    s["dash"] = st.selectbox(
                        "Dash",
                        PLOTLY_LINE_STYLES,
                        index=PLOTLY_LINE_STYLES.index(s.get("dash") or "solid"),
                        key=f"{key_prefix}_over_dash_{sim_prefix}",
                        disabled=not s["enabled"]
                    )
    
    return ps


def render_axis_limits_ui(ps, key_prefix=""):
    """
    Render UI controls for axis limits.
    
    Args:
        ps: Plot style dictionary (modified in place)
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
    
    Returns:
        Updated ps dictionary (also modifies in place)
    """
    st.markdown("**Axis Limits**")
    ps["enable_x_limits"] = st.checkbox("Set X-axis limits", bool(ps.get("enable_x_limits", False)), 
                                         key=f"{key_prefix}_enable_x_limits")
    if ps["enable_x_limits"]:
        col1, col2 = st.columns(2)
        with col1:
            ps["x_min"] = st.number_input("X min", value=float(ps.get("x_min") or 0.0), 
                                          format="%.6g", key=f"{key_prefix}_x_min")
        with col2:
            ps["x_max"] = st.number_input("X max", value=float(ps.get("x_max") or 1.0), 
                                          format="%.6g", key=f"{key_prefix}_x_max")
    
    ps["enable_y_limits"] = st.checkbox("Set Y-axis limits", bool(ps.get("enable_y_limits", False)), 
                                         key=f"{key_prefix}_enable_y_limits")
    if ps["enable_y_limits"]:
        col1, col2 = st.columns(2)
        with col1:
            ps["y_min"] = st.number_input("Y min", value=float(ps.get("y_min") or 0.0), 
                                          format="%.6g", key=f"{key_prefix}_y_min")
        with col2:
            ps["y_max"] = st.number_input("Y max", value=float(ps.get("y_max") or 1.0), 
                                          format="%.6g", key=f"{key_prefix}_y_max")
    
    return ps


def apply_axis_limits(layout_kwargs, ps):
    """
    Apply axis limits to layout kwargs if enabled.
    
    For log scales, Plotly expects range values as base-10 logarithms, not raw values.
    This function automatically converts raw values to log10 when axis type is "log".
    
    Args:
        layout_kwargs: Dictionary of layout arguments for update_layout
        ps: Plot style dictionary
    
    Returns:
        Updated layout_kwargs dictionary
    """
    import numpy as np
    
    if ps.get("enable_x_limits") and ps.get("x_min") is not None and ps.get("x_max") is not None:
        x_min_raw = float(ps["x_min"])
        x_max_raw = float(ps["x_max"])
        x_axis_type = ps.get("x_axis_type", "linear")
        
        if x_axis_type == "log" and x_min_raw > 0 and x_max_raw > 0:
            # Convert raw numeric limits to base-10 log for Plotly's log-range
            x_range = [np.log10(x_min_raw), np.log10(x_max_raw)]
        else:
            x_range = [x_min_raw, x_max_raw]
        
        layout_kwargs["xaxis_range"] = x_range
    
    if ps.get("enable_y_limits") and ps.get("y_min") is not None and ps.get("y_max") is not None:
        y_min_raw = float(ps["y_min"])
        y_max_raw = float(ps["y_max"])
        y_axis_type = ps.get("y_axis_type", "linear")
        
        if y_axis_type == "log" and y_min_raw > 0 and y_max_raw > 0:
            # Convert raw numeric limits to base-10 log for Plotly's log-range
            y_range = [np.log10(y_min_raw), np.log10(y_max_raw)]
        else:
            y_range = [y_min_raw, y_max_raw]
        
        layout_kwargs["yaxis_range"] = y_range
    
    return layout_kwargs


def render_figure_size_ui(ps, key_prefix=""):
    """
    Render UI controls for figure size.
    
    Args:
        ps: Plot style dictionary (modified in place)
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
    
    Returns:
        Updated ps dictionary (also modifies in place)
    """
    st.markdown("**Figure Size**")
    ps["enable_custom_size"] = st.checkbox("Set custom figure size", bool(ps.get("enable_custom_size", False)), 
                                            key=f"{key_prefix}_enable_custom_size")
    if ps["enable_custom_size"]:
        col1, col2 = st.columns(2)
        with col1:
            ps["figure_width"] = st.number_input("Width (px)", min_value=200, max_value=2000, 
                                                  value=int(ps.get("figure_width") or 800), 
                                                  step=50, key=f"{key_prefix}_figure_width")
        with col2:
            ps["figure_height"] = st.number_input("Height (px)", min_value=200, max_value=2000, 
                                                   value=int(ps.get("figure_height") or 500), 
                                                   step=50, key=f"{key_prefix}_figure_height")
    
    return ps


def apply_figure_size(layout_kwargs, ps):
    """
    Apply figure size to layout kwargs if enabled.
    
    Args:
        layout_kwargs: Dictionary of layout arguments for update_layout
        ps: Plot style dictionary
    
    Returns:
        Updated layout_kwargs dictionary
    """
    if ps.get("enable_custom_size"):
        if ps.get("figure_width") is not None:
            layout_kwargs["width"] = ps["figure_width"]
        if ps.get("figure_height") is not None:
            layout_kwargs["height"] = ps["figure_height"]
    return layout_kwargs


# ==========================================================
# Helper functions
# ==========================================================

def _get_palette(ps):
    """
    Get color palette from plot style configuration.
    
    Args:
        ps: Plot style dictionary
    
    Returns:
        List of color hex codes
    """
    if ps.get("palette") == "Custom":
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
    return mapping.get(ps.get("palette", "Plotly"), pc.qualitative.Plotly)


def _axis_title_font(ps):
    """
    Get axis title font configuration.
    
    Args:
        ps: Plot style dictionary
    
    Returns:
        Font dictionary for axis titles
    """
    return dict(family=ps.get("font_family", "Arial"), size=ps.get("axis_title_size", 14))


def _tick_font(ps):
    """
    Get tick label font configuration.
    
    Args:
        ps: Plot style dictionary
    
    Returns:
        Font dictionary for tick labels
    """
    return dict(family=ps.get("font_family", "Arial"), size=ps.get("tick_font_size", 12))


def _normalize_plot_name(plot_name: str) -> str:
    """
    Normalize plot name to a valid key format.
    
    Args:
        plot_name: Plot name string
    
    Returns:
        Normalized plot name
    """
    return plot_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


# ==========================================================
# Default plot style
# ==========================================================

def default_plot_style():
    """
    Get default plot style configuration with all features.
    
    Returns:
        Dictionary with default plot style settings
    """
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
        
        # Axis scale type (linear or logarithmic)
        "x_axis_type": "linear",  # "linear" or "log"
        "y_axis_type": "linear",  # "linear" or "log"
        
        # Tick number format (integer, float, scientific, or normal)
        "x_tick_format": "auto",  # "auto", "integer", "float", "scientific", or "normal"
        "x_tick_decimals": 2,  # Number of decimal places when format is "float" or "scientific"
        "y_tick_format": "auto",  # "auto", "integer", "float", "scientific", or "normal"
        "y_tick_decimals": 2,  # Number of decimal places when format is "float" or "scientific"

        # Axis borders (spines) for scientific box appearance
        "show_axis_lines": True,
        "axis_line_width": 0.8,  # Thinner borders for cleaner look
        "axis_line_color": "#000000",
        "mirror_axes": True,  # Show borders on all sides (box)
        "tick_color": None,  # None means use axis_line_color

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
        "marker_size": 6,

        "show_legend": True,  # Show legend by default

        "palette": "Plotly",
        "custom_colors": ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                          "#8c564b", "#e377c2", "#7f7f7f"],
        "template": "plotly_white",

        "enable_per_sim_style": False,
        "per_sim_style": {},  # {sim: {enabled,color,width,dash,marker,msize}}
        
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


# ==========================================================
# UI rendering functions for new features
# ==========================================================

def render_axis_scale_ui(ps, key_prefix=""):
    """
    Render UI controls for axis scale type (linear/log).
    
    Args:
        ps: Plot style dictionary (modified in place)
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
    
    Returns:
        Updated ps dictionary (also modifies in place)
    """
    st.markdown("**Axis Scale Type**")
    axis_types = ["linear", "log"]
    x_axis_type_idx = axis_types.index(ps.get("x_axis_type", "linear"))
    y_axis_type_idx = axis_types.index(ps.get("y_axis_type", "linear"))
    ps["x_axis_type"] = st.selectbox(
        "X-axis scale", 
        axis_types, 
        index=x_axis_type_idx,
        help="Linear: standard scale. Log: logarithmic scale (useful for wide data ranges)",
        key=f"{key_prefix}_x_axis_type"
    )
    ps["y_axis_type"] = st.selectbox(
        "Y-axis scale", 
        axis_types, 
        index=y_axis_type_idx,
        help="Linear: standard scale. Log: logarithmic scale (useful for wide data ranges)",
        key=f"{key_prefix}_y_axis_type"
    )
    return ps


def render_tick_format_ui(ps, key_prefix=""):
    """
    Render UI controls for tick number format.
    
    Args:
        ps: Plot style dictionary (modified in place)
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
    
    Returns:
        Updated ps dictionary (also modifies in place)
    """
    st.markdown("**Tick Number Format**")
    tick_format_options = ["auto", "integer", "float", "scientific", "normal"]
    x_format_default = ps.get("x_tick_format", "auto")
    y_format_default = ps.get("y_tick_format", "auto")
    x_format_idx = tick_format_options.index(x_format_default) if x_format_default in tick_format_options else 0
    y_format_idx = tick_format_options.index(y_format_default) if y_format_default in tick_format_options else 0
    
    ps["x_tick_format"] = st.selectbox(
        "X-axis number format", 
        tick_format_options, 
        index=x_format_idx,
        help="Auto: let Plotly decide. Integer: no decimals. Float/Normal: show decimals. Scientific: exponential notation",
        key=f"{key_prefix}_x_tick_format"
    )
    if ps["x_tick_format"] in ["float", "scientific", "normal"]:
        ps["x_tick_decimals"] = st.slider(
            "X-axis decimal places", 
            0, 6, 
            int(ps.get("x_tick_decimals", 2)),
            help="Number of decimal places for X-axis",
            key=f"{key_prefix}_x_tick_decimals"
        )
    
    ps["y_tick_format"] = st.selectbox(
        "Y-axis number format", 
        tick_format_options, 
        index=y_format_idx,
        help="Auto: let Plotly decide. Integer: no decimals. Float/Normal: show decimals. Scientific: exponential notation",
        key=f"{key_prefix}_y_tick_format"
    )
    if ps["y_tick_format"] in ["float", "scientific", "normal"]:
        ps["y_tick_decimals"] = st.slider(
            "Y-axis decimal places", 
            0, 6, 
            int(ps.get("y_tick_decimals", 2)),
            help="Number of decimal places for Y-axis",
            key=f"{key_prefix}_y_tick_decimals"
        )
    
    return ps


def render_axis_borders_ui(ps, key_prefix=""):
    """
    Render UI controls for axis borders (box).
    
    Args:
        ps: Plot style dictionary (modified in place)
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
    
    Returns:
        Updated ps dictionary (also modifies in place)
    """
    st.markdown("**Axis Borders (Box)**")
    ps["show_axis_lines"] = st.checkbox(
        "Show axis borders", 
        bool(ps.get("show_axis_lines", True)), 
        help="Enable to show axis border lines (scientific paper style)",
        key=f"{key_prefix}_show_axis_lines"
    )
    ps["axis_line_width"] = st.slider(
        "Border line width", 
        0.5, 4.0, 
        float(ps.get("axis_line_width", 0.8)), 
        disabled=not ps.get("show_axis_lines", True),
        key=f"{key_prefix}_axis_line_width"
    )
    ps["axis_line_color"] = st.color_picker(
        "Border color", 
        ps.get("axis_line_color", "#000000"), 
        disabled=not ps.get("show_axis_lines", True),
        key=f"{key_prefix}_axis_line_color"
    )
    ps["mirror_axes"] = st.checkbox(
        "Box on all sides", 
        bool(ps.get("mirror_axes", True)), 
        help="Checked: 4 borders (box). Unchecked: 2 borders (bottom & left only)", 
        disabled=not ps.get("show_axis_lines", True),
        key=f"{key_prefix}_mirror_axes"
    )
    return ps


# ==========================================================
# Plot style application
# ==========================================================

def apply_plot_style(fig, ps):
    """
    Apply comprehensive plot style to a Plotly figure.
    
    This function applies all plot style settings including:
    - Fonts, colors, backgrounds
    - Grid (major and minor)
    - Axis scale types (linear/log)
    - Axis borders and ticks
    - Tick formatting
    
    Args:
        fig: Plotly figure object
        ps: Plot style dictionary
    
    Returns:
        Updated figure object
    """
    # Build legend configuration
    # Note: 'showlegend' is a top-level layout property, not in legend dict
    # The legend dict uses 'visible' property instead
    legend_config = dict(
        font=dict(size=ps.get("legend_size", 12)),
    )
    
    layout_update = dict(
        template=ps.get("template", "plotly_white"),
        font=dict(family=ps.get("font_family", "Arial"), size=ps.get("font_size", 14)),
        legend=legend_config,
        showlegend=ps.get("show_legend", True),  # Top-level showlegend property (not in legend dict)
        hovermode="x unified",
        plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"),
        paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF"),
    )
    
    if ps.get("title_size") is not None and ps.get("plot_title"):
        layout_update["title"] = dict(
            font=dict(
                family=ps.get("font_family", "Arial"),
                size=ps.get("title_size", 16)
            ),
            text=ps.get("plot_title")
        )
    
    fig.update_layout(**layout_update)

    tick_dir = "outside" if ps.get("ticks_outside", True) else "inside"

    show_x_grid = ps.get("show_grid", True) and ps.get("grid_on_x", True)
    show_y_grid = ps.get("show_grid", True) and ps.get("grid_on_y", True)
    grid_rgba = f"rgba{hex_to_rgb(ps.get('grid_color', '#B0B0B0')) + (ps.get('grid_opacity', 0.6),)}"

    show_minor = ps.get("show_minor_grid", False)
    minor_rgba = f"rgba{hex_to_rgb(ps.get('minor_grid_color', '#D0D0D0')) + (ps.get('minor_grid_opacity', 0.45),)}"

    # Axis borders (spines) for scientific box appearance
    show_axis_lines = ps.get("show_axis_lines", True)
    axis_line_width = ps.get("axis_line_width", 0.8)
    axis_line_color = ps.get("axis_line_color", "#000000")
    mirror_axes = ps.get("mirror_axes", True)
    
    # Ensure ticks are visible - set tick color to match axis line color
    tick_color = ps.get("tick_color") or axis_line_color
    
    # For mirror axes, use "ticks" to show ticks on all sides
    mirror_mode = "ticks" if mirror_axes else False
    
    # Get axis scale types (linear or log)
    x_axis_type = ps.get("x_axis_type", "linear")
    y_axis_type = ps.get("y_axis_type", "linear")
    
    # Check if scientific notation will be used (for controlling minor tick labels)
    x_format_pref = ps.get("x_tick_format", "auto")
    x_uses_scientific = (x_format_pref == "scientific") or (x_format_pref == "auto" and x_axis_type == "log")
    y_format_pref = ps.get("y_tick_format", "auto")
    y_uses_scientific = (y_format_pref == "scientific") or (y_format_pref == "auto" and y_axis_type == "log")
    
    # For log scales, Plotly automatically shows minor ticks (dense ticks between major ticks)
    # We enable them explicitly to ensure they're visible when needed
    show_minor_ticks_x = x_axis_type == "log"
    show_minor_ticks_y = y_axis_type == "log"
    
    # Build minor tick configuration for x-axis
    x_minor_dict = dict(
        showgrid=show_minor,
        gridwidth=ps.get("minor_grid_w", 0.4),
        griddash=ps.get("minor_grid_dash", "dot"),
        gridcolor=minor_rgba,
        ticks=tick_dir if show_minor_ticks_x else "",  # Enable minor ticks for log scales (Plotly handles density automatically)
        ticklen=ps.get("tick_len", 6) * 0.6,  # Minor ticks are shorter than major ticks
        tickcolor=tick_color,
    )
    # Note: Plotly's minor axis doesn't support showticklabels property
    # For log scales with scientific notation, Plotly should automatically only show labels on major ticks
    
    # Build axis update dictionaries - conditionally include linecolor
    # Plotly doesn't accept "transparent" as a color, so we only set linecolor when showing lines
    xaxis_update = dict(
        type=x_axis_type,  # "linear" or "log" - controls axis scale type
        ticks=tick_dir,  # "outside" or "inside" - controls tick visibility
        ticklen=ps.get("tick_len", 6),
        tickwidth=ps.get("tick_w", 1.2),
        tickcolor=tick_color,  # Make ticks visible with explicit color
        tickfont=_tick_font(ps),
        title_font=_axis_title_font(ps),
        showticklabels=True,  # Explicitly show tick labels
        showgrid=show_x_grid,
        gridwidth=ps.get("grid_w", 0.6),
        griddash=ps.get("grid_dash", "dot"),
        gridcolor=grid_rgba,
        showline=show_axis_lines,
        linewidth=axis_line_width if show_axis_lines else 0,
        mirror=mirror_mode,  # "ticks" shows ticks on all sides when True, False shows only bottom/left
        minor=x_minor_dict,
    )
    # Only set linecolor when showing axis lines (avoids "transparent" error)
    if show_axis_lines:
        xaxis_update["linecolor"] = axis_line_color
    
    # Build minor tick configuration for y-axis
    y_minor_dict = dict(
        showgrid=show_minor,
        gridwidth=ps.get("minor_grid_w", 0.4),
        griddash=ps.get("minor_grid_dash", "dot"),
        gridcolor=minor_rgba,
        ticks=tick_dir if show_minor_ticks_y else "",  # Enable minor ticks for log scales (Plotly handles density automatically)
        ticklen=ps.get("tick_len", 6) * 0.6,  # Minor ticks are shorter than major ticks
        tickcolor=tick_color,
    )
    # Note: Plotly's minor axis doesn't support showticklabels property
    # For log scales with scientific notation, Plotly should automatically only show labels on major ticks
    
    yaxis_update = dict(
        type=y_axis_type,  # "linear" or "log" - controls axis scale type
        ticks=tick_dir,  # "outside" or "inside" - controls tick visibility
        ticklen=ps.get("tick_len", 6),
        tickwidth=ps.get("tick_w", 1.2),
        tickcolor=tick_color,  # Make ticks visible with explicit color
        tickfont=_tick_font(ps),
        title_font=_axis_title_font(ps),
        showticklabels=True,  # Explicitly show tick labels
        showgrid=show_y_grid,
        gridwidth=ps.get("grid_w", 0.6),
        griddash=ps.get("grid_dash", "dot"),
        gridcolor=grid_rgba,
        showline=show_axis_lines,
        linewidth=axis_line_width if show_axis_lines else 0,
        mirror=mirror_mode,  # "ticks" shows ticks on all sides when True, False shows only bottom/left
        minor=y_minor_dict,
    )
    # Only set linecolor when showing axis lines (avoids "transparent" error)
    if show_axis_lines:
        yaxis_update["linecolor"] = axis_line_color
    
    # Apply updates
    fig.update_xaxes(**xaxis_update)
    fig.update_yaxes(**yaxis_update)
    
    # Apply tick format if specified
    x_format_pref = ps.get("x_tick_format", "auto")
    x_axis_type = ps.get("x_axis_type", "linear")
    
    if x_format_pref == "integer":
        fig.update_xaxes(tickformat=".0f")
    elif x_format_pref == "float":
        decimals = ps.get("x_tick_decimals", 2)
        fig.update_xaxes(tickformat=f".{decimals}f")
    elif x_format_pref == "scientific":
        decimals = ps.get("x_tick_decimals", 2)
        fig.update_xaxes(tickformat=f".{decimals}e")
        # For log scales with scientific notation, ensure only major ticks show labels
        if x_axis_type == "log":
            fig.update_xaxes(dtick=1)  # Only show labels at powers of 10 (major ticks)
    elif x_format_pref == "normal":
        decimals = ps.get("x_tick_decimals", 2)
        fig.update_xaxes(tickformat=f".{decimals}f")
    elif x_format_pref == "auto":
        # For "auto" on log scale, use scientific notation to avoid SI prefixes (like μ)
        # For linear scale, let Plotly decide
        if x_axis_type == "log":
            # Use scientific notation for log scale to prevent SI prefixes
            decimals = ps.get("x_tick_decimals", 2)
            fig.update_xaxes(tickformat=f".{decimals}e", dtick=1)  # Only show labels at powers of 10 (major ticks)
        # For linear scale with auto, let Plotly use default (no explicit format)
    
    y_format_pref = ps.get("y_tick_format", "auto")
    y_axis_type = ps.get("y_axis_type", "linear")
    
    if y_format_pref == "integer":
        fig.update_yaxes(tickformat=".0f")
    elif y_format_pref == "float":
        decimals = ps.get("y_tick_decimals", 2)
        fig.update_yaxes(tickformat=f".{decimals}f")
    elif y_format_pref == "scientific":
        decimals = ps.get("y_tick_decimals", 2)
        fig.update_yaxes(tickformat=f".{decimals}e")
        # For log scales with scientific notation, ensure only major ticks show labels
        if y_axis_type == "log":
            fig.update_yaxes(dtick=1)  # Only show labels at powers of 10 (major ticks)
    elif y_format_pref == "normal":
        decimals = ps.get("y_tick_decimals", 2)
        fig.update_yaxes(tickformat=f".{decimals}f")
    elif y_format_pref == "auto":
        # For "auto" on log scale, use scientific notation to avoid SI prefixes (like μ)
        # For linear scale, let Plotly decide
        if y_axis_type == "log":
            # Use scientific notation for log scale to prevent SI prefixes
            decimals = ps.get("y_tick_decimals", 2)
            fig.update_yaxes(tickformat=f".{decimals}e", dtick=1)  # Only show labels at powers of 10 (major ticks)
        # For linear scale with auto, let Plotly use default (no explicit format)
    
    return fig


def get_tick_format(axis_format, decimals, is_normalized=False):
    """
    Get tick format string based on user preference and normalization state.
    
    Args:
        axis_format: "auto", "integer", or "float"
        decimals: Number of decimal places (for float format)
        is_normalized: Whether the data is normalized (affects auto format)
    
    Returns:
        Format string for Plotly tickformat
    """
    if axis_format == "integer":
        return ".0f"  # Integer format (no decimals)
    elif axis_format == "float":
        return f".{decimals}f"  # Float with specified decimal places
    else:  # "auto"
        # Auto: use fewer decimals for normalized, more for non-normalized
        return ".3f" if is_normalized else ".4g"


# ==========================================================
# Comprehensive plot style sidebar
# ==========================================================

# ==========================================================
# Legend and Axis Labels UI
# ==========================================================

def render_legend_axis_labels_ui(data_dir=None, traces=None, 
                                  legend_names_key="custom_plot_legend_names",
                                  axis_labels_key="custom_plot_axis_labels",
                                  trace_key_func=None,
                                  save_callback=None, reset_callback=None,
                                  key_prefix=""):
    """
    Render UI for legend names and axis labels (persistent).
    
    This function provides a reusable sidebar UI for managing:
    - Trace legend names (customizable per trace)
    - Axis labels (X and Y)
    - Save/reset functionality
    
    Args:
        data_dir: Optional Path for saving metadata
        traces: List of trace dictionaries (each with keys like 'data_source', 'x_col', 'y_col', 'label')
        legend_names_key: Key in session_state for legend names dict
        axis_labels_key: Key in session_state for axis labels dict
        trace_key_func: Optional function(trace) -> str to generate trace key. 
                       Default: f"{trace['data_source']}_{trace['x_col']}_{trace['y_col']}"
        save_callback: Optional function(data_dir) to call on save
        reset_callback: Optional function() to call on reset
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
    
    Returns:
        Updated legend_names and axis_labels dictionaries (also modifies session_state)
    """
    # Initialize session state if needed
    if legend_names_key not in st.session_state:
        st.session_state[legend_names_key] = {}
    if axis_labels_key not in st.session_state:
        st.session_state[axis_labels_key] = {'x': 'X', 'y': 'Y'}
    
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Trace Legend Names")
        if traces:
            for idx, trace in enumerate(traces):
                # Generate trace key
                if trace_key_func:
                    trace_key = trace_key_func(trace)
                else:
                    # Default: use data_source, x_col, y_col
                    trace_key = f"{trace.get('data_source', '')}_{trace.get('x_col', '')}_{trace.get('y_col', '')}"
                
                # Get default name
                default_name = trace.get('label', 
                    f"{trace.get('data_source', '').split('_')[-1]}: {trace.get('y_col', '')}")
                
                # Ensure key exists in dict
                st.session_state[legend_names_key].setdefault(trace_key, default_name)
                
                # Render text input
                st.session_state[legend_names_key][trace_key] = st.text_input(
                    f"Trace {idx+1} label",
                    value=st.session_state[legend_names_key][trace_key],
                    key=f"{key_prefix}_legend_{idx}_{trace_key}"
                )
        else:
            st.caption("Add traces to customize legend names")
        
        st.markdown("---")
        st.markdown("### Axis Labels")
        st.session_state[axis_labels_key]['x'] = st.text_input(
            "X-axis label",
            value=st.session_state[axis_labels_key].get('x', 'X'),
            key=f"{key_prefix}_axis_x"
        )
        st.session_state[axis_labels_key]['y'] = st.text_input(
            "Y-axis label",
            value=st.session_state[axis_labels_key].get('y', 'Y'),
            key=f"{key_prefix}_axis_y"
        )
        
        # Save/Reset buttons if data_dir provided
        if data_dir is not None:
            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Save labels/legends", key=f"{key_prefix}_save_labels"):
                    if save_callback:
                        save_callback(data_dir)
                    else:
                        # Default save behavior - try to save to legend_names.json
                        try:
                            from pathlib import Path
                            import json
                            json_path = data_dir / "legend_names.json"
                            old = {}
                            if json_path.exists():
                                try:
                                    old = json.loads(json_path.read_text(encoding="utf-8"))
                                except Exception:
                                    old = {}
                            old[legend_names_key] = st.session_state[legend_names_key]
                            old[axis_labels_key] = st.session_state[axis_labels_key]
                            json_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
                        except Exception as e:
                            st.warning(f"Could not save: {e}")
                    st.success("Saved to legend_names.json")
            with b2:
                if st.button("♻️ Reset labels/legends", key=f"{key_prefix}_reset_labels"):
                    # 1) Reset our own dicts
                    st.session_state[legend_names_key] = {}
                    st.session_state[axis_labels_key] = {'x': 'X', 'y': 'Y'}
                    
                    # 2) Reset widget state for axis text inputs
                    for k in [f"{key_prefix}_axis_x", f"{key_prefix}_axis_y"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    
                    # 3) Reset widget state for legend text inputs
                    if traces:
                        for idx, trace in enumerate(traces):
                            # Same key logic used above
                            if trace_key_func:
                                trace_key = trace_key_func(trace)
                            else:
                                trace_key = f"{trace.get('data_source', '')}_{trace.get('x_col', '')}_{trace.get('y_col', '')}"
                            widget_key = f"{key_prefix}_legend_{idx}_{trace_key}"
                            if widget_key in st.session_state:
                                del st.session_state[widget_key]
                    
                    if reset_callback:
                        reset_callback()
                    else:
                        # Default reset behavior - try to save to legend_names.json
                        try:
                            from pathlib import Path
                            import json
                            json_path = data_dir / "legend_names.json"
                            old = {}
                            if json_path.exists():
                                try:
                                    old = json.loads(json_path.read_text(encoding="utf-8"))
                                except Exception:
                                    old = {}
                            old[legend_names_key] = st.session_state[legend_names_key]
                            old[axis_labels_key] = st.session_state[axis_labels_key]
                            json_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
                        except Exception:
                            pass  # Silent fail on reset save
                    st.toast("Reset + saved.", icon="♻️")
                    st.rerun()
    
    return st.session_state[legend_names_key], st.session_state[axis_labels_key]


def plot_style_sidebar(data_dir=None, sim_groups=None, style_key="per_sim_style", 
                       key_prefix="", include_marker=False, 
                       save_callback=None, reset_callback=None,
                       theme_selector=None, plot_name=None):
    """
    Render comprehensive plot style sidebar with all features.
    
    This is the main function for rendering the "Plot Style (persistent)" sidebar.
    It includes all features from the Other Turbulence Stats page:
    - Fonts, backgrounds, ticks
    - Axis scale type (linear/log)
    - Tick number format
    - Axis borders (box)
    - Grid (major and minor)
    - Colors and palette
    - Theme selector (optional)
    - Axis limits
    - Figure size
    - Per-simulation style overrides
    
    Args:
        data_dir: Optional Path for saving metadata (if None, no save/reset buttons)
        sim_groups: Optional dict of simulation groups for per-sim styling
        style_key: Key in plot style dict for per-simulation styles
        key_prefix: Prefix for Streamlit keys (to avoid conflicts)
        include_marker: Whether to show marker controls in per-sim styling
        save_callback: Optional function(data_dir) to call on save
        reset_callback: Optional function() to call on reset
        theme_selector: Optional function(ps) to render theme selector
        plot_name: Optional plot name to display in header
    
    Returns:
        Updated plot style dictionary (also modifies st.session_state.plot_style)
    """
    # Get current plot style from session state
    if "plot_style" not in st.session_state:
        st.session_state.plot_style = default_plot_style()
    
    ps = dict(st.session_state.plot_style)
    
    # Ensure per-sim defaults if sim_groups provided
    if sim_groups:
        ensure_per_sim_defaults(ps, sim_groups, style_key, include_marker)
    
    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        if plot_name:
            st.markdown(f"**Configuring: {plot_name}**")
        
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        font_idx = fonts.index(ps.get("font_family", "Arial")) if ps.get("font_family", "Arial") in fonts else 0
        ps["font_family"] = st.selectbox(
            "Font family", 
            fonts, 
            index=font_idx,
            key=f"{key_prefix}_font_family"
        )
        ps["font_size"] = st.slider(
            "Base/global font size", 
            8, 26, 
            int(ps.get("font_size", 14)),
            key=f"{key_prefix}_font_size"
        )
        ps["title_size"] = st.slider(
            "Plot title size", 
            10, 32, 
            int(ps.get("title_size", 16)),
            key=f"{key_prefix}_title_size"
        )
        ps["legend_size"] = st.slider(
            "Legend font size", 
            8, 24, 
            int(ps.get("legend_size", 12)),
            key=f"{key_prefix}_legend_size"
        )
        ps["show_legend"] = st.checkbox(
            "Show legend", 
            bool(ps.get("show_legend", True)),
            help="Display legend on the plot",
            key=f"{key_prefix}_show_legend"
        )
        ps["tick_font_size"] = st.slider(
            "Tick label font size", 
            6, 24, 
            int(ps.get("tick_font_size", 12)),
            key=f"{key_prefix}_tick_font_size"
        )
        ps["axis_title_size"] = st.slider(
            "Axis title font size", 
            8, 28, 
            int(ps.get("axis_title_size", 14)),
            key=f"{key_prefix}_axis_title_size"
        )

        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker(
            "Plot background (inside axes)", 
            ps.get("plot_bgcolor", "#FFFFFF"),
            key=f"{key_prefix}_plot_bgcolor"
        )
        ps["paper_bgcolor"] = st.color_picker(
            "Paper background (outside axes)", 
            ps.get("paper_bgcolor", "#FFFFFF"),
            key=f"{key_prefix}_paper_bgcolor"
        )

        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider(
            "Tick length", 
            2, 14, 
            int(ps.get("tick_len", 6)),
            key=f"{key_prefix}_tick_len"
        )
        ps["tick_w"] = st.slider(
            "Tick width", 
            0.5, 3.5, 
            float(ps.get("tick_w", 1.2)),
            key=f"{key_prefix}_tick_w"
        )
        ps["ticks_outside"] = st.checkbox(
            "Ticks outside", 
            bool(ps.get("ticks_outside", True)),
            key=f"{key_prefix}_ticks_outside"
        )

        st.markdown("---")
        render_axis_scale_ui(ps, key_prefix=key_prefix)

        st.markdown("---")
        render_tick_format_ui(ps, key_prefix=key_prefix)

        st.markdown("---")
        render_axis_borders_ui(ps, key_prefix=key_prefix)

        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox(
            "Show major grid", 
            bool(ps.get("show_grid", True)),
            key=f"{key_prefix}_show_grid"
        )
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            ps["grid_on_x"] = st.checkbox(
                "Grid on X", 
                bool(ps.get("grid_on_x", True)),
                key=f"{key_prefix}_grid_on_x"
            )
        with gcol2:
            ps["grid_on_y"] = st.checkbox(
                "Grid on Y", 
                bool(ps.get("grid_on_y", True)),
                key=f"{key_prefix}_grid_on_y"
            )
        ps["grid_w"] = st.slider(
            "Major grid width", 
            0.2, 2.5, 
            float(ps.get("grid_w", 0.6)),
            key=f"{key_prefix}_grid_w"
        )
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        grid_dash_idx = grid_styles.index(ps.get("grid_dash", "dot")) if ps.get("grid_dash", "dot") in grid_styles else 1
        ps["grid_dash"] = st.selectbox(
            "Major grid type", 
            grid_styles,
            index=grid_dash_idx,
            key=f"{key_prefix}_grid_dash"
        )
        ps["grid_color"] = st.color_picker(
            "Major grid color", 
            ps.get("grid_color", "#B0B0B0"),
            key=f"{key_prefix}_grid_color"
        )
        ps["grid_opacity"] = st.slider(
            "Major grid opacity", 
            0.0, 1.0, 
            float(ps.get("grid_opacity", 0.6)),
            key=f"{key_prefix}_grid_opacity"
        )

        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox(
            "Show minor grid", 
            bool(ps.get("show_minor_grid", False)),
            key=f"{key_prefix}_show_minor_grid"
        )
        ps["minor_grid_w"] = st.slider(
            "Minor grid width", 
            0.1, 2.0, 
            float(ps.get("minor_grid_w", 0.4)),
            key=f"{key_prefix}_minor_grid_w"
        )
        minor_grid_dash_idx = grid_styles.index(ps.get("minor_grid_dash", "dot")) if ps.get("minor_grid_dash", "dot") in grid_styles else 1
        ps["minor_grid_dash"] = st.selectbox(
            "Minor grid type", 
            grid_styles,
            index=minor_grid_dash_idx,
            key=f"{key_prefix}_minor_grid_dash"
        )
        ps["minor_grid_color"] = st.color_picker(
            "Minor grid color", 
            ps.get("minor_grid_color", "#D0D0D0"),
            key=f"{key_prefix}_minor_grid_color"
        )
        ps["minor_grid_opacity"] = st.slider(
            "Minor grid opacity", 
            0.0, 1.0, 
            float(ps.get("minor_grid_opacity", 0.45)),
            key=f"{key_prefix}_minor_grid_opacity"
        )

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider(
            "Global line width", 
            0.5, 7.0, 
            float(ps.get("line_width", 2.2)),
            key=f"{key_prefix}_line_width"
        )
        ps["marker_size"] = st.slider(
            "Global marker size", 
            0, 18, 
            int(ps.get("marker_size", 6)),
            key=f"{key_prefix}_marker_size"
        )

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        palette_idx = palettes.index(ps.get("palette", "Plotly")) if ps.get("palette", "Plotly") in palettes else 0
        ps["palette"] = st.selectbox(
            "Palette", 
            palettes,
            index=palette_idx,
            key=f"{key_prefix}_palette"
        )
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(
                    f"Color {i+1}", 
                    c, 
                    key=f"{key_prefix}_cust_color_{i}"
                ))
            ps["custom_colors"] = new_cols

        if theme_selector:
            st.markdown("---")
            st.markdown("**Theme**")
            theme_selector(ps)

        st.markdown("---")
        render_axis_limits_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_figure_size_ui(ps, key_prefix=key_prefix)
        
        if sim_groups:
            st.markdown("---")
            render_per_sim_style_ui(
                ps, 
                sim_groups, 
                style_key=style_key, 
                key_prefix=key_prefix, 
                include_marker=include_marker
            )

        # Save/Reset buttons if data_dir provided
        if data_dir is not None:
            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Save Plot Style", key=f"{key_prefix}_save_style"):
                    st.session_state.plot_style = ps
                    if save_callback:
                        save_callback(data_dir)
                    else:
                        # Default save behavior - try to save to legend_names.json
                        try:
                            from pathlib import Path
                            import json
                            json_path = data_dir / "legend_names.json"
                            old = {}
                            if json_path.exists():
                                try:
                                    old = json.loads(json_path.read_text(encoding="utf-8"))
                                except Exception:
                                    old = {}
                            old["plot_style"] = ps
                            json_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
                        except Exception as e:
                            st.warning(f"Could not save: {e}")
                    st.success("Saved plot style.")
            with b2:
                if st.button("♻️ Reset Plot Style", key=f"{key_prefix}_reset_style"):
                    # 1) Reset the underlying style dict
                    st.session_state.plot_style = default_plot_style()

                    # 2) Clear widget state so widgets re-read from defaults next run
                    widget_keys = [
                        # Fonts / legend
                        f"{key_prefix}_font_family",
                        f"{key_prefix}_font_size",
                        f"{key_prefix}_title_size",
                        f"{key_prefix}_legend_size",
                        f"{key_prefix}_show_legend",
                        f"{key_prefix}_tick_font_size",
                        f"{key_prefix}_axis_title_size",
                        # Backgrounds
                        f"{key_prefix}_plot_bgcolor",
                        f"{key_prefix}_paper_bgcolor",
                        # Ticks
                        f"{key_prefix}_tick_len",
                        f"{key_prefix}_tick_w",
                        f"{key_prefix}_ticks_outside",
                        # Axis scale
                        f"{key_prefix}_x_axis_type",
                        f"{key_prefix}_y_axis_type",
                        # Tick formats
                        f"{key_prefix}_x_tick_format",
                        f"{key_prefix}_x_tick_decimals",
                        f"{key_prefix}_y_tick_format",
                        f"{key_prefix}_y_tick_decimals",
                        # Axis borders
                        f"{key_prefix}_show_axis_lines",
                        f"{key_prefix}_axis_line_width",
                        f"{key_prefix}_axis_line_color",
                        f"{key_prefix}_mirror_axes",
                        # Major grid
                        f"{key_prefix}_show_grid",
                        f"{key_prefix}_grid_on_x",
                        f"{key_prefix}_grid_on_y",
                        f"{key_prefix}_grid_w",
                        f"{key_prefix}_grid_dash",
                        f"{key_prefix}_grid_color",
                        f"{key_prefix}_grid_opacity",
                        # Minor grid
                        f"{key_prefix}_show_minor_grid",
                        f"{key_prefix}_minor_grid_w",
                        f"{key_prefix}_minor_grid_dash",
                        f"{key_prefix}_minor_grid_color",
                        f"{key_prefix}_minor_grid_opacity",
                        # Curves
                        f"{key_prefix}_line_width",
                        f"{key_prefix}_marker_size",
                        # Palette
                        f"{key_prefix}_palette",
                        # Axis limits
                        f"{key_prefix}_enable_x_limits",
                        f"{key_prefix}_x_min",
                        f"{key_prefix}_x_max",
                        f"{key_prefix}_enable_y_limits",
                        f"{key_prefix}_y_min",
                        f"{key_prefix}_y_max",
                        # Figure size
                        f"{key_prefix}_enable_custom_size",
                        f"{key_prefix}_figure_width",
                        f"{key_prefix}_figure_height",
                        # Per-sim global toggle
                        f"{key_prefix}_enable_per_sim",
                    ]

                    # Custom color inputs (if you ever used them)
                    for i in range(10):
                        widget_keys.append(f"{key_prefix}_cust_color_{i}")

                    # Per-simulation style widgets (if sim_groups provided)
                    if sim_groups:
                        for sim_prefix in sim_groups.keys():
                            for suffix in [
                                "over_on",
                                "over_color",
                                "over_width",
                                "over_dash",
                                "over_marker",
                                "over_msize",
                            ]:
                                widget_keys.append(f"{key_prefix}_{suffix}_{sim_prefix}")

                    for k in widget_keys:
                        if k in st.session_state:
                            del st.session_state[k]

                    # Optional: let external reset callback do extra cleanup
                    if reset_callback:
                        reset_callback()
                    else:
                        # Default: save default style to legend_names.json
                        try:
                            from pathlib import Path
                            import json
                            json_path = data_dir / "legend_names.json"
                            old = {}
                            if json_path.exists():
                                try:
                                    old = json.loads(json_path.read_text(encoding="utf-8"))
                                except Exception:
                                    old = {}
                            old["plot_style"] = st.session_state.plot_style
                            json_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
                        except Exception:
                            pass  # silent fail

                    st.toast("Reset + saved.", icon="♻️")
                    st.rerun()
    
    st.session_state.plot_style = ps
    return ps

