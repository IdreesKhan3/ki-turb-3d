"""
Structure Functions — Cached readers, time averaging, format helpers.
"""

import re
import streamlit as st
from pathlib import Path
from plotly.colors import hex_to_rgb

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


def extract_iter(fname: str) -> int | None:
    """Extract iteration number from filename stem."""
    stem = Path(fname).stem
    nums = re.findall(r"(\d+)", stem)
    return int(nums[-1]) if nums else None


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
    """Convert color to RGB tuple, handling hex and rgb() string formats."""
    if isinstance(color, str) and color.startswith("rgb("):
        match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    try:
        return hex_to_rgb(color)
    except (ValueError, TypeError):
        return (0, 0, 0)
