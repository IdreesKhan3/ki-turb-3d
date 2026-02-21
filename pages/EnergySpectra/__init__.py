"""
Energy Spectra page modules — plot style, file loading, time-averaged, time evolution.
"""

from .plot_style import get_plot_style, apply_plot_style, plot_style_sidebar
from .file_loading import init_session_state, load_files_and_groups, render_legend_and_axis_labels
from .time_averaged import render_time_averaged
from .time_evolution import render_time_evolution

__all__ = [
    "get_plot_style",
    "apply_plot_style",
    "plot_style_sidebar",
    "init_session_state",
    "load_files_and_groups",
    "render_legend_and_axis_labels",
    "render_time_averaged",
    "render_time_evolution",
]
