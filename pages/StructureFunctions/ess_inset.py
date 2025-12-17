import plotly.graph_objects as go
import numpy as np
from utils.plot_style import resolve_line_style, _default_labelify, _axis_title_font, _tick_font, hex_to_rgb

# --- Theory Constants & Functions ---
TABLE_P = [2, 3, 4, 5, 6]
EXP_ZETA = [0.71, 1.00, 1.28, 1.53, 1.78]

def zeta_p_she_leveque(p):
    """Computes She-Leveque scaling exponent."""
    return p/9 + 2*(1 - (2/3)**(p/3))

def add_ess_inset(
    fig: go.Figure,
    xi_all: dict,
    anom_all: dict,
    xi_err_all: dict,
    sim_groups: dict,
    legend_names: dict,
    colors_palette: list,
    plot_style: dict,
    show_sl_theory: bool = True,
    show_exp_anom: bool = True
):
    """
    Adds the Anomaly (xi_p - p/3) inset to the ESS figure.
    """
    # 1. Determine ranges and p values
    inset_ps = []
    inset_y_min = None
    inset_y_max = None
    
    if xi_all:
        all_ps = []
        for k in xi_all.keys():
            if xi_all[k]:
                all_ps.extend(xi_all[k].keys())
        if all_ps:
            inset_ps = sorted(set(all_ps))
            
            # Get y range for inset
            all_anom_vals = []
            for k in anom_all.keys():
                if anom_all[k]:
                    all_anom_vals.extend(anom_all[k].values())
            
            if all_anom_vals:
                y_range = max(all_anom_vals) - min(all_anom_vals)
                # Add 15% padding
                inset_y_min = min(all_anom_vals) - 0.15 * y_range if y_range > 0 else -0.2
                inset_y_max = max(all_anom_vals) + 0.15 * y_range if y_range > 0 else 0.2
            else:
                inset_y_min, inset_y_max = -0.2, 0.2

    # Return if no data
    if not inset_ps or inset_y_min is None:
        return fig

    # 2. Extract style settings from plot_style (reuse existing utilities)
    # Scale down font sizes for inset (typically 70% of main plot)
    inset_scale = 0.7
    base_tick_font = _tick_font(plot_style)
    base_axis_font = _axis_title_font(plot_style)
    
    inset_tick_font = dict(
        family=base_tick_font.get("family", "Arial"),
        size=int(base_tick_font.get("size", 12) * inset_scale)
    )
    inset_axis_font = dict(
        family=base_axis_font.get("family", "Arial"),
        size=int(base_axis_font.get("size", 14) * inset_scale)
    )
    
    # Grid settings from plot_style
    show_grid = plot_style.get("show_grid", True)
    grid_on_x = plot_style.get("grid_on_x", True) and show_grid
    grid_on_y = plot_style.get("grid_on_y", True) and show_grid
    grid_color = plot_style.get("grid_color", "#B0B0B0")
    grid_opacity = plot_style.get("grid_opacity", 0.6)
    grid_width = plot_style.get("grid_w", 0.6)
    grid_dash = plot_style.get("grid_dash", "dot")
    
    # Convert grid color to rgba
    rgb = hex_to_rgb(grid_color)
    grid_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{grid_opacity})"
    
    # Axis line settings
    show_axis_lines = plot_style.get("show_axis_lines", True)
    axis_line_color = plot_style.get("axis_line_color", "#000000")
    axis_line_width = plot_style.get("axis_line_width", 0.8) * inset_scale
    mirror_axes = plot_style.get("mirror_axes", True)
    mirror_mode = "ticks" if mirror_axes else False
    
    # Tick settings
    tick_len = int(plot_style.get("tick_len", 6) * inset_scale)
    tick_w = plot_style.get("tick_w", 1.2) * inset_scale
    ticks_outside = plot_style.get("ticks_outside", True)
    tick_dir = "outside" if ticks_outside else "inside"
    tick_color = plot_style.get("tick_color") or axis_line_color
    
    # 3. Update Layout with Inset Axes (using plot_style settings)
    fig.update_layout(
        xaxis2=dict(
            domain=[0.77, 0.95],
            anchor="y2",
            title=dict(text="p", font=inset_axis_font),
            tickfont=inset_tick_font,
            type="linear",
            showgrid=grid_on_x,
            gridcolor=grid_rgba,
            gridwidth=grid_width,
            griddash=grid_dash,
            showline=show_axis_lines,
            linecolor=axis_line_color if show_axis_lines else "transparent",
            linewidth=axis_line_width if show_axis_lines else 0,
            mirror=mirror_mode,
            tickmode="array",
            tickvals=inset_ps,
            ticklen=tick_len,
            tickwidth=tick_w,
            tickcolor=tick_color,
            ticks=tick_dir,
            range=[min(inset_ps)-0.5, max(inset_ps)+0.5],
            visible=True
        ),
        yaxis2=dict(
            domain=[0.3, 0.52],
            anchor="x2",
            title=dict(text="ξp - p/3", font=inset_axis_font),
            tickfont=inset_tick_font,
            type="linear",
            showgrid=grid_on_y,
            gridcolor=grid_rgba,
            gridwidth=grid_width,
            griddash=grid_dash,
            showline=show_axis_lines,
            linecolor=axis_line_color if show_axis_lines else "transparent",
            linewidth=axis_line_width if show_axis_lines else 0,
            mirror=mirror_mode,
            ticklen=tick_len,
            tickwidth=tick_w,
            tickcolor=tick_color,
            ticks=tick_dir,
            range=[inset_y_min, inset_y_max],
            visible=True
        )
    )

    # 3. Add Simulation Traces
    # We iterate exactly as the main loop to ensure colors match
    for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
        # Skip if this sim has no ESS data calculated
        if sim_prefix not in xi_all or not xi_all[sim_prefix]:
            continue

        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix, idx, colors_palette, plot_style,
            style_key="per_sim_style_structure",
            include_marker=True,
            default_marker="circle"
        )

        ps_show = sorted(xi_all[sim_prefix].keys())
        yvals = [anom_all[sim_prefix][p] for p in ps_show]
        yerr = [xi_err_all[sim_prefix].get(p, 0.0) for p in ps_show]
        
        name_label = legend_names.get(sim_prefix, _default_labelify(sim_prefix))

        # Scale down line width and marker size for inset (70% of main plot)
        inset_lw = lw * 0.7
        inset_ms = max(2, int(msize * 0.5))
        
        fig.add_trace(go.Scatter(
            x=ps_show, y=yvals,
            mode="lines+markers",
            name=f"{name_label} (inset)",
            line=dict(color=color, width=inset_lw, dash=dash),
            marker=dict(symbol=marker, size=inset_ms),
            error_y=dict(type="data", array=yerr, visible=True, thickness=1, width=3),
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
            hoverinfo="skip"
        ))

    # 4. Add She-Leveque Theory
    if show_sl_theory:
        theory_anom = [zeta_p_she_leveque(p) - p/3 for p in inset_ps]
        sl_color = plot_style.get("she_leveque_color", "#000000")
        inset_lw = plot_style.get("line_width", 2.2) * 0.7
        inset_ms = max(2, int(plot_style.get("marker_size", 6) * 0.5))
        fig.add_trace(go.Scatter(
            x=inset_ps, y=theory_anom,
            mode="lines+markers",
            name="She–Leveque 1994 (inset)",
            line=dict(color=sl_color, dash="dash", width=inset_lw),
            marker=dict(symbol="diamond", size=inset_ms),
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
            hoverinfo="skip"
        ))

    # 5. Add Experimental B93
    if show_exp_anom:
        exp_anom = [EXP_ZETA[i] - TABLE_P[i]/3 for i in range(len(TABLE_P))]
        exp_color = plot_style.get("experimental_b93_color", "#00BFC4")
        inset_lw = plot_style.get("line_width", 2.2) * 0.7
        inset_ms = max(2, int(plot_style.get("marker_size", 6) * 0.5))
        fig.add_trace(go.Scatter(
            x=TABLE_P, y=exp_anom,
            mode="lines+markers",
            name="Experiment B93 (inset)",
            line=dict(color=exp_color, width=inset_lw),
            marker=dict(symbol="x", size=inset_ms),
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
            hoverinfo="skip"
        ))

    # 6. Add Background and Zero-line
    # Use plot background color from plot_style
    inset_bgcolor = plot_style.get("plot_bgcolor", "#FFFFFF")
    inset_bg_opacity = 0.9  # Keep background slightly transparent for overlay effect
    
    fig.add_shape(
        type="rect",
        x0=min(inset_ps)-0.5, x1=max(inset_ps)+0.5,
        y0=inset_y_min, y1=inset_y_max,
        fillcolor=inset_bgcolor,
        opacity=inset_bg_opacity,
        layer="below",
        line=dict(width=0),
        xref="x2", yref="y2"
    )

    # Zero line - use zero_line_color from plot_style if available, otherwise fall back to axis_line_color
    zero_line_color = plot_style.get("zero_line_color", axis_line_color)
    zero_line_width = plot_style.get("zero_line_width", axis_line_width)
    # Ensure zero line is at least slightly thicker than grid to be visible
    if zero_line_width <= grid_width:
        zero_line_width = max(zero_line_width, grid_width * 1.5)
    
    # Draw zero line above background but ensure it's visible
    # Use "above" layer so it appears on top of axis lines and grid
    # Use a distinct dash style to differentiate from grid (dashdot instead of dash)
    fig.add_shape(
        type="line",
        x0=min(inset_ps)-0.5, x1=max(inset_ps)+0.5,
        y0=0, y1=0,
        line=dict(dash="dashdot", color=zero_line_color, width=zero_line_width),
        xref="x2", yref="y2",
        layer="above"  # Draw above axis lines and grid so it's clearly visible
    )

    return fig