"""
Structure Functions page utilities.
"""

from .ess_inset import add_ess_inset
from .file_loading import (
    init_session_state,
    load_structure_groups,
    render_legend_and_axis_labels,
)
from .plot_style import get_plot_style, apply_plot_style, plot_style_sidebar
from .views import render_sp_tab, render_ess_tab, render_table_tab, render_theory_section
from .data_helpers import (
    read_structure_bin_cached,
    read_structure_txt_cached,
    compute_time_avg_structure,
    color_to_rgb_tuple,
)

__all__ = [
    "add_ess_inset",
    "init_session_state",
    "load_structure_groups",
    "render_legend_and_axis_labels",
    "get_plot_style",
    "apply_plot_style",
    "plot_style_sidebar",
    "render_sp_tab",
    "render_ess_tab",
    "render_table_tab",
    "render_theory_section",
    "read_structure_bin_cached",
    "read_structure_txt_cached",
    "compute_time_avg_structure",
    "color_to_rgb_tuple",
]
