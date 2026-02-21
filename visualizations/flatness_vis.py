import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def create_flatness_figure(
    sim_items: List[Tuple[str, Dict[str, Any]]],
    style_config: Dict[str, Any],
    show_std_band: bool = True,
    show_error_bars: bool = False,
    axis_labels: Optional[Dict[str, str]] = None,
    simulation_legend_names: Optional[Dict[str, str]] = None,
    apply_style: bool = True,
) -> Optional[go.Figure]:
    """Creates a Plotly figure for flatness factor F(r) vs r."""

    if not sim_items:
        return None

    fig = go.Figure()

    axis_labels = axis_labels or {"x": "r", "y": "F(r)"}
    simulation_legend_names = simulation_legend_names or {}

    for i, (sim_prefix, data) in enumerate(sim_items):
        r = np.array(data["r"])
        F_mean = np.array(data["F_mean"])
        F_std = np.array(data["F_std"])

        if r.size == 0 or F_mean.size == 0:
            continue

        sim_display_name = simulation_legend_names.get(sim_prefix, sim_prefix)

        # Mean curve
        fig.add_trace(
            go.Scatter(
                x=r,
                y=F_mean,
                mode="lines",
                name=sim_display_name,
                line=dict(width=style_config.get("line_width", 2.2), color=style_config.get("custom_colors", [])[i] if style_config.get("palette") == "Custom" and len(style_config.get("custom_colors", [])) > i else None),
                hovertemplate=f"<b>{{name}}</b><br>r: %{{x:.2e}}<br>F(r): %{{y:.4f}}<extra></extra>",
            )
        )

        # Uncertainty band
        if show_std_band and F_std.size > 0:
            F_upper = F_mean + F_std
            F_lower = F_mean - F_std
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([r, r[::-1]]),
                    y=np.concatenate([F_upper, F_lower[::-1]]),
                    fill="toself",
                    fillcolor=style_config.get("custom_colors", [])[i] if style_config.get("palette") == "Custom" and len(style_config.get("custom_colors", [])) > i else None,
                    opacity=0.2,
                    line=dict(width=0),
                    name=f"{sim_display_name} Std Dev",
                    showlegend=False,
                    hovertemplate=f"<b>{{name}}</b><br>r: %{{x:.2e}}<br>F(r) Upper: %{{y:.4f}}<extra></extra>",
                )
            )

        # Error bars
        if show_error_bars and F_std.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=r,
                    y=F_mean,
                    mode="markers",
                    name=f"{sim_display_name} Error Bars",
                    error_y=dict(type="data", array=F_std, visible=True),
                    marker=dict(size=style_config.get("marker_size", 6), color=style_config.get("custom_colors", [])[i] if style_config.get("palette") == "Custom" and len(style_config.get("custom_colors", [])) > i else None),
                    showlegend=False,
                    hovertemplate=f"<b>{{name}}</b><br>r: %{{x:.2e}}<br>F(r): %{{y:.4f}}<br>Std: %{{customdata[0]:.4f}}<extra></extra>",
                    customdata=F_std[:, np.newaxis],
                )
            )

    if apply_style:
        # Apply general style settings
        fig.update_layout(
            title_text=style_config.get("plot_title", "Flatness Factor F(r)") if style_config.get("show_plot_title", True) else None,
            title_font_size=style_config.get("title_size", 18),
            font_family=style_config.get("font_family", "Arial"),
            font_size=style_config.get("font_size", 12),
            font_color=style_config.get("font_color", "#333"),
            plot_bgcolor=style_config.get("plot_bgcolor", "white"),
            paper_bgcolor=style_config.get("paper_bgcolor", "white"),
            showlegend=style_config.get("show_legend", True),
            legend_font_size=style_config.get("legend_size", 12),
            margin=dict(
                l=style_config.get("margin_left", 60),
                r=style_config.get("margin_right", 20),
                t=style_config.get("margin_top", 40),
                b=style_config.get("margin_bottom", 50),
            ),
            width=style_config.get("figure_width", 800) if style_config.get("enable_custom_size", False) else None,
            height=style_config.get("figure_height", 600) if style_config.get("enable_custom_size", False) else None,
        )

        # Apply axis settings
        fig.update_xaxes(
            title_text=axis_labels.get("x", "r"),
            type=style_config.get("x_axis_type", "log"),
            tickformat=style_config.get("x_tick_format"),
            showgrid=style_config.get("grid_on_x", True) and style_config.get("show_grid", True),
            gridwidth=style_config.get("grid_w", 1),
            gridcolor=style_config.get("grid_color", "#eee"),
            griddash=style_config.get("grid_dash", "solid"),
            showline=style_config.get("show_axis_lines", True),
            linewidth=style_config.get("axis_line_width", 1),
            linecolor=style_config.get("axis_line_color", "#333"),
            mirror=style_config.get("mirror_axes", False),
            ticks=style_config.get("ticks_outside", True) and "outside" or "",
            ticklen=style_config.get("tick_len", 5),
            tickwidth=style_config.get("tick_w", 1),
            tickcolor=style_config.get("tick_color", "#333"),
            range=[np.log10(style_config["x_min"]), np.log10(style_config["x_max"])] if style_config.get("enable_x_limits", False) else None,
            title_font_size=style_config.get("axis_title_size", 14),
            tickfont_size=style_config.get("tick_font_size", 10),
        )
        fig.update_yaxes(
            title_text=axis_labels.get("y", "F(r)"),
            type=style_config.get("y_axis_type", "linear"),
            tickformat=style_config.get("y_tick_format"),
            showgrid=style_config.get("grid_on_y", True) and style_config.get("show_grid", True),
            gridwidth=style_config.get("grid_w", 1),
            gridcolor=style_config.get("grid_color", "#eee"),
            griddash=style_config.get("grid_dash", "solid"),
            showline=style_config.get("show_axis_lines", True),
            linewidth=style_config.get("axis_line_width", 1),
            linecolor=style_config.get("axis_line_color", "#333"),
            mirror=style_config.get("mirror_axes", False),
            ticks=style_config.get("ticks_outside", True) and "outside" or "",
            ticklen=style_config.get("tick_len", 5),
            tickwidth=style_config.get("tick_w", 1),
            tickcolor=style_config.get("tick_color", "#333"),
            range=[style_config["y_min"], style_config["y_max"]] if style_config.get("enable_y_limits", False) else None,
            title_font_size=style_config.get("axis_title_size", 14),
            tickfont_size=style_config.get("tick_font_size", 10),
        )

        # Minor grid
        if style_config.get("show_minor_grid", False):
            fig.update_xaxes(
                minor_showgrid=True,
                minor_gridwidth=style_config.get("minor_grid_w", 0.5),
                minor_gridcolor=style_config.get("minor_grid_color", "#ddd"),
                minor_griddash=style_config.get("minor_grid_dash", "dot"),
            )
            fig.update_yaxes(
                minor_showgrid=True,
                minor_gridwidth=style_config.get("minor_grid_w", 0.5),
                minor_gridcolor=style_config.get("minor_grid_color", "#ddd"),
                minor_griddash=style_config.get("minor_grid_dash", "dot"),
            )

        # Apply template if specified
        if style_config.get("template"): # e.g., 'plotly_dark'
            fig.update_layout(template=style_config["template"])

    return fig
