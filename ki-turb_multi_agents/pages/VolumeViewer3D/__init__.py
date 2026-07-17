"""
3D Volume Viewer — Interactive 3D volume visualization with ParaView-like features.
"""

from .file_loading import load_volume_data
from .plot_style import get_plot_style_3d, render_plot_style_sidebar
from .views import render_main_view, render_theory_section
from .data_helpers import (
    load_velocity_file,
    cached_read_vti,
    cached_read_hdf5,
    safe_minmax,
    downsample3d,
    make_grid,
    apply_clip,
    colormap_options,
)

__all__ = [
    "load_volume_data",
    "get_plot_style_3d",
    "render_plot_style_sidebar",
    "render_main_view",
    "render_theory_section",
    "load_velocity_file",
    "cached_read_vti",
    "cached_read_hdf5",
    "safe_minmax",
    "downsample3d",
    "make_grid",
    "apply_clip",
    "colormap_options",
]
