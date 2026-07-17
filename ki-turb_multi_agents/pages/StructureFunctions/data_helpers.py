"""
Structure Functions — Cached readers, time averaging, format helpers.
"""

import streamlit as st

from utils.plot_style import color_to_rgb
from data_readers.binary_reader import read_structure_function_file
from core_physics import compute_structure_time_avg

try:
    from data_readers.text_reader import read_structure_function_txt
except Exception:
    read_structure_function_txt = None


@st.cache_data(show_spinner=False)
def read_structure_bin_cached(fname: str):
    """Read binary structure function file with caching."""
    return read_structure_function_file(fname)


@st.cache_data(show_spinner=False)
def read_structure_txt_cached(fname: str):
    """Read text structure function file with caching."""
    if read_structure_function_txt is None:
        raise RuntimeError("Text reader not available.")
    return read_structure_function_txt(fname)


@st.cache_data(show_spinner=False)
def compute_time_avg_structure(files: tuple, kind: str):
    """
    Compute time-averaged structure functions.
    kind: "bin" or "txt"
    Returns (r_mean, Sp_mean_dict, Sp_std_dict, u_rms_mean, ps) or (None,...) on failure.
    """
    data_list = []
    for f in files:
        try:
            data = read_structure_bin_cached(str(f)) if kind == "bin" else read_structure_txt_cached(str(f))
            data_list.append(data)
        except Exception:
            continue
    result = compute_structure_time_avg(data_list)
    r_mean, Sp_mean_dict, Sp_std_dict, u_rms_mean, ps = result
    if r_mean is None:
        return None, None, None, None, None
    return r_mean, Sp_mean_dict, Sp_std_dict, u_rms_mean, ps


def color_to_rgb_tuple(color) -> tuple:
    """Convert color to RGB tuple. Delegates to utils.plot_style.color_to_rgb."""
    return color_to_rgb(color)
