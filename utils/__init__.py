"""
Utility functions module
"""

from .file_detector import detect_simulation_files
from .data_processor import process_time_average
from .export import export_plot
from .plot_style import (
    PLOTLY_LINE_STYLES,
    PLOTLY_MARKER_STYLES,
    resolve_line_style,
    ensure_per_sim_defaults,
    render_per_sim_style_ui,
    render_axis_limits_ui,
    apply_axis_limits,
    render_figure_size_ui,
    apply_figure_size,
)

__all__ = [
    'detect_simulation_files',
    'process_time_average',
    'export_plot',
    'PLOTLY_LINE_STYLES',
    'PLOTLY_MARKER_STYLES',
    'resolve_line_style',
    'ensure_per_sim_defaults',
    'render_per_sim_style_ui',
    'render_axis_limits_ui',
    'apply_axis_limits',
    'render_figure_size_ui',
    'apply_figure_size',
]

