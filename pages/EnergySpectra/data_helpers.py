"""
Energy Spectra — Cached data readers and compute helpers.
"""

import re
import streamlit as st
import numpy as np
from pathlib import Path

from data_readers.spectrum_reader import read_spectrum_file
from data_readers.norm_spectrum_reader import read_norm_spectrum_file
from core_physics import compute_spectrum_time_avg, compute_spectrum_time_avg_norm


@st.cache_data(show_spinner=False)
def read_spectrum_cached(fname: str):
    k, E = read_spectrum_file(fname)
    return np.asarray(k, float), np.asarray(E, float)


@st.cache_data(show_spinner=False)
def read_norm_cached(fname: str):
    keta, Enorm, Epope = read_norm_spectrum_file(fname)
    return (
        np.asarray(keta, float),
        np.asarray(Enorm, float),
        np.asarray(Epope, float),
    )


@st.cache_data(show_spinner=False)
def compute_time_avg(files: tuple):
    data_list = [read_spectrum_cached(str(f)) for f in files]
    return compute_spectrum_time_avg(data_list)


@st.cache_data(show_spinner=False)
def compute_time_avg_norm(files: tuple):
    data_list = [read_norm_cached(str(f)) for f in files]
    return compute_spectrum_time_avg_norm(data_list)


def extract_iter(fname: str):
    """Extract last number from filename stem."""
    stem = Path(fname).stem
    nums = re.findall(r"(\d+)", stem)
    return int(nums[-1]) if nums else None
