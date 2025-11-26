"""
Centralized plot style utilities for per-simulation marker and line style management.

Provides:
- All Plotly-supported marker and line style options
- Unified resolve_line_style function
- Reusable UI component for per-simulation styling
"""

import streamlit as st

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
    
    Args:
        layout_kwargs: Dictionary of layout arguments for update_layout
        ps: Plot style dictionary
    
    Returns:
        Updated layout_kwargs dictionary
    """
    if ps.get("enable_x_limits") and ps.get("x_min") is not None and ps.get("x_max") is not None:
        layout_kwargs["xaxis_range"] = [ps["x_min"], ps["x_max"]]
    if ps.get("enable_y_limits") and ps.get("y_min") is not None and ps.get("y_max") is not None:
        layout_kwargs["yaxis_range"] = [ps["y_min"], ps["y_max"]]
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

