"""
Flatness — Cached readers, time averaging, format helpers.
"""

import re
import streamlit as st
import numpy as np
from plotly.colors import hex_to_rgb

from data_readers.text_reader import read_flatness_file
from core_physics import compute_flatness_time_avg


@st.cache_data(show_spinner=False)
def read_flatness_cached(fname: str):
    """Read flatness file with caching."""
    r, F = read_flatness_file(fname)
    return np.asarray(r, float), np.asarray(F, float)


@st.cache_data(show_spinner=False)
def compute_time_avg_flatness(files: tuple, num_errorbars: int = 20):
    """Compute time-averaged flatness from file list."""
    data_list = [read_flatness_cached(str(f)) for f in files]
    return compute_flatness_time_avg(data_list, num_errorbars)


def format_legend_name(prefix: str) -> str:
    """Format simulation prefix for legend display."""
    name = prefix.replace("flatness_", "").replace("data", "").strip("_")
    name = name.replace("_", " ").title()
    return name if name else prefix


def color_to_rgb_tuple(color) -> tuple:
    """Convert color to RGB tuple, handling hex and rgb() string formats."""
    if isinstance(color, str) and color.startswith("rgb("):
        match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    try:
        return hex_to_rgb(color)
    except (ValueError, TypeError):
        return (0, 0, 0)
