"""
Flatness — Flatness factors page package.
"""

from .file_loading import (
    init_session_state,
    load_flatness_groups,
    render_legend_and_axis_labels,
)
from .plot_style import get_plot_style, apply_plot_style, plot_style_sidebar
from .views import render_main_plot, render_theory_section
from .data_helpers import (
    read_flatness_cached,
    compute_time_avg_flatness,
    format_legend_name,
    color_to_rgb_tuple,
)

__all__ = [
    "init_session_state",
    "load_flatness_groups",
    "render_legend_and_axis_labels",
    "get_plot_style",
    "apply_plot_style",
    "plot_style_sidebar",
    "render_main_plot",
    "render_theory_section",
    "read_flatness_cached",
    "compute_time_avg_flatness",
    "format_legend_name",
    "color_to_rgb_tuple",
]
