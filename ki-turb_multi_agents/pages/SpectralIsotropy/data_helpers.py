"""
Spectral Isotropy — Cached readers and averaging.
"""

import streamlit as st
from pathlib import Path

from core_physics import read_isotropy_coeff_file, avg_isotropy_coeff


@st.cache_data(show_spinner=False)
def read_isotropy_coeff_cached(fname: str):
    return read_isotropy_coeff_file(Path(fname))


def avg_isotropy_coeff_from_files(files):
    """Time-average isotropy coefficient over files."""
    data_list = []
    for f in files:
        d = read_isotropy_coeff_cached(str(f))
        if d.size > 0:
            data_list.append(d)
    return avg_isotropy_coeff(data_list)
