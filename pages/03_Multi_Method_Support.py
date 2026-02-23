"""
Multi-Method Support Page
Tool scope and extension guidelines for HIT turbulence analysis
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
st.set_page_config(page_icon="⚫")

def main():
    inject_theme_css()
    st.title("Multi-Method Support")
    
    # Get theme colors for consistent styling
    current_theme = st.session_state.get("theme", "Light Scientific")
    is_dark = "Dark" in current_theme
    text_color = "#d4d4d4" if is_dark else "#1f1f1f"
    secondary_text = "#b0b0b0" if is_dark else "#666666"
    card_bg = "#2a2a2a" if is_dark else "#f8f9fa"
    border_color = "#444444" if is_dark else "#e0e0e0"
    accent_color = "#4a9eff" if is_dark else "#0066cc"
    
    # Placeholder message
    st.info("This page is intentionally left as a **placeholder** for users who wish to extend the tool with their own analysis scripts.")
    
    st.markdown(f"""
    <div style='background: {card_bg}; padding: 1rem; border-radius: 6px; border-left: 4px solid {accent_color}; margin-bottom: 1rem;'>
        <h3 style='margin: 0 0 0.5rem 0; color: {text_color}; font-size: 1.1rem;'>Extension Point</h3>
        <p style='margin: 0; color: {text_color}; line-height: 1.6; font-size: 0.95rem;'>
            Core analyses—energy spectra, structure functions, flatness, isotropy, PDFs, and related statistics—are 
            <strong>precomputed and visualized</strong> in this tool. If you want to add further analysis, you can 
            <strong>wire your own scripts</strong> to compute and visualize additional quantities from velocity fields 
            (e.g. <code>*.vti</code>, <code>*.h5</code>, <code>*.hdf5</code>), following the same pattern as the built-in pages.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

