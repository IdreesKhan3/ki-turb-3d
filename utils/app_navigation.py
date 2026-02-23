"""
App navigation config and UI for in-page browsing.

Page list and iframe renderer used by Autonomous Lab (and potentially other pages).
Not specific to AutonomousLab — app-wide navigation config.
"""

import streamlit as st
from typing import List, Tuple

# Streamlit page slug (URL path) -> display name
# Slug must match pages/NN_Name.py filename (without .py)
APP_NAVIGATION_PAGES: List[Tuple[str, str]] = [
    ("01_Overview", "Overview"),
    ("02_Theory_Equations", "Theory Equations"),
    ("03_Multi_Method_Support", "Multi Method Support"),
    ("04_Real_Isotropy", "Real Isotropy"),
    ("05_Spectral_Isotropy", "Spectral Isotropy"),
    ("06_Energy_Spectra", "Energy Spectra"),
    ("07_Flatness", "Flatness"),
    ("08_Structure_Functions", "Structure Functions"),
    ("09_PDFs", "PDFs"),
    ("10_Other_Turbulence_Stats", "Other Turbulence Stats"),
    ("11_3D_Volume_Viewer", "3D Volume Viewer"),
    ("12_Report_Generator", "Report Generator"),
    ("13_Citation", "Citation"),
]


def render_app_navigation_iframe(key_prefix: str = "lab", in_sidebar: bool = False):
    """Render expander with iframe to browse app pages without leaving current page."""
    iframe_height = 380 if in_sidebar else 700
    with st.expander("Browse App Pages", expanded=False):
        st.caption("Select a page to view. Session state is shared.")
        options = [name for _, name in APP_NAVIGATION_PAGES]
        slugs = {name: slug for slug, name in APP_NAVIGATION_PAGES}
        selected = st.selectbox(
            "Page",
            options=options,
            index=0,
            key=f"{key_prefix}_browse_page_select",
            label_visibility="collapsed",
        )
        slug = slugs.get(selected, APP_NAVIGATION_PAGES[0][0])
        iframe_src = f"/{slug}"
        iframe_html = f"""
        <iframe
            src="{iframe_src}"
            style="width:100%; height:{iframe_height}px; border:1px solid #ddd; border-radius:4px;"
            title="Browse: {selected}"
        ></iframe>
        """
        st.components.v1.html(iframe_html, height=iframe_height + 20, scrolling=False)
