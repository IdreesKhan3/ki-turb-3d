"""
Flatness — Cached readers, time averaging, format helpers.
"""

import streamlit as st
import numpy as np

from utils.plot_style import color_to_rgb
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
    """Convert color to RGB tuple. Delegates to utils.plot_style.color_to_rgb."""
    return color_to_rgb(color)
