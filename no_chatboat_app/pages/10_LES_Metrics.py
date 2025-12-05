"""
LES Metrics Page
C_S statistics and effective viscosity
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
st.set_page_config(page_icon="⚫")

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    st.title("🔬 LES Metrics")
    st.info("LES Metrics page - Implementation in progress")

if __name__ == "__main__":
    main()

