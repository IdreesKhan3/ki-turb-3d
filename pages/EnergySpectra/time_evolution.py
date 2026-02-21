"""
Energy Spectra — Time evolution rendering.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List

from utils.report_builder import capture_button
from utils.export_figs import export_panel
from visualizations.spectra_vis import create_time_evolution_figure

from .plot_style import get_plot_style, apply_plot_style
from .data_helpers import read_spectrum_cached, extract_iter
from .file_loading import _default_labelify


def render_time_evolution(
    data_dir: Path,
    sim_groups: Dict[str, List[str]],
) -> bool:
    """
    Render time evolution of energy spectra.
    Returns True if rendered, False if early exit.
    """
    st.header("Time Evolution of Energy Spectra")

    if not sim_groups:
        st.error("Time evolution requires spectrum*.dat files.")
        return False

    sim_names = sorted(sim_groups.keys())
    sim_display_names = [_default_labelify(n) for n in sim_names]
    sim_sel_display = st.sidebar.selectbox(
        "Simulation group", sim_display_names, index=0, key="energy_evol_sim"
    )
    sim_sel = sim_names[sim_display_names.index(sim_sel_display)]

    files = sim_groups[sim_sel]
    iters = [extract_iter(f) for f in files]
    if all(i is None for i in iters):
        st.error("Could not extract iteration numbers from filenames.")
        return False

    every_n = st.sidebar.slider(
        "Show every Nth iteration curve", 1, min(50, len(files)), 5, key="energy_evol_every_n"
    )
    thin_idx = list(range(0, len(files), every_n))
    thin_files = [files[i] for i in thin_idx]
    thin_iters = [iters[i] for i in thin_idx]

    sel_pos = st.sidebar.slider(
        "Highlight curve (thinned index)", 0, len(thin_files) - 1, len(thin_files) - 1,
        key="energy_evol_highlight"
    )
    highlight_file = thin_files[sel_pos]
    highlight_iter = thin_iters[sel_pos]

    ps_evol = get_plot_style("Time Evolution")

    thin_curves = []
    for f, it in zip(thin_files, thin_iters):
        try:
            k, E = read_spectrum_cached(str(f))
            thin_curves.append({"x": k, "y": E})
        except Exception:
            continue

    highlight_curve = None
    try:
        kH, EH = read_spectrum_cached(str(highlight_file))
        highlight_curve = {"x": kH, "y": EH, "label": f"Highlighted iter {highlight_iter}"}
    except Exception as e:
        st.warning(f"Highlight read failed: {e}")

    figE = create_time_evolution_figure(
        thin_curves,
        highlight_curve,
        ps_evol,
        axis_labels=st.session_state.axis_labels_raw,
        apply_style=False,
    )
    figE = apply_plot_style(figE, ps_evol)

    st.plotly_chart(figE, width="content")
    capture_button(
        figE, title=f"Energy Spectra Time Evolution - {sim_sel}", source_page="Energy Spectra"
    )

    st.subheader("Export time evolution figure")
    export_panel(figE, data_dir, f"energy_spectra_time_evolution_{sim_sel}")

    return True
