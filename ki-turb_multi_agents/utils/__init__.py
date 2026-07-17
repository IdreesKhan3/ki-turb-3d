"""Core-safe utility exports; Streamlit-dependent plot helpers are loaded lazily."""
from __future__ import annotations
from typing import Any

__all__ = [
    "detect_simulation_files", "PLOTLY_LINE_STYLES", "PLOTLY_MARKER_STYLES",
    "resolve_line_style", "ensure_per_sim_defaults", "render_per_sim_style_ui",
    "render_axis_limits_ui", "apply_axis_limits", "render_figure_size_ui",
    "apply_figure_size",
]

def __getattr__(name: str) -> Any:
    if name == "detect_simulation_files":
        from .file_detector import detect_simulation_files
        return detect_simulation_files
    if name in set(__all__) - {"detect_simulation_files"}:
        from . import plot_style
        return getattr(plot_style, name)
    raise AttributeError(name)
