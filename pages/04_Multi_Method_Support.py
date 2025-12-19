"""
Multi-Method Support Page
Tool scope and extension guidelines for HIT turbulence analysis
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
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
    
    # Main scope statement
    st.markdown(f"""
    <div style='background: {card_bg}; padding: 1rem; border-radius: 6px; border-left: 4px solid {accent_color}; margin-bottom: 1.5rem;'>
        <h3 style='margin: 0 0 0.5rem 0; color: {text_color}; font-size: 1.1rem;'>Tool Scope</h3>
        <p style='margin: 0; color: {text_color}; line-height: 1.6; font-size: 0.95rem;'>
            <strong>KI-TURB 3D</strong> is designed for <strong>Homogeneous Isotropic Turbulence (HIT)</strong> 
            data analysis and visualization. The tool focuses on <strong>DNS/LES</strong> simulations using 
            <strong>LBM</strong>, with primary emphasis on <strong>MRT</strong> 
            collision operators. The analysis framework assumes <strong>periodic boundary conditions</strong> 
            and is optimized for HIT configurations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Method support
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background: {card_bg}; padding: 0.8rem; border-radius: 6px; border-left: 3px solid {accent_color}; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: {text_color}; font-size: 1rem;'>Primary Method</h4>
            <p style='margin: 0; color: {secondary_text}; font-size: 0.9rem; line-height: 1.5;'>
                <strong style='color: {text_color};'>MRT</strong><br>
                D3Q19 lattice, full 19×19 transformation matrix, optimized for DNS/LES HIT analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: {card_bg}; padding: 0.8rem; border-radius: 6px; border-left: 3px solid {accent_color}; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: {text_color}; font-size: 1rem;'>Extended Support</h4>
            <p style='margin: 0; color: {secondary_text}; font-size: 0.9rem; line-height: 1.5;'>
                <strong style='color: {text_color};'>SRT, BGK, TRT</strong> data can be analyzed using the same 
                framework, with method-specific parameters handled appropriately.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Extension invitation
    st.markdown("---")
    st.markdown(f"""
    <div style='background: {card_bg}; padding: 1.2rem; border-radius: 6px; border: 2px solid {accent_color}; margin-top: 1.5rem;'>
        <h3 style='margin: 0 0 0.8rem 0; color: {text_color}; font-size: 1.1rem; display: flex; align-items: center;'>
            Extend & Contribute
        </h3>
        <p style='margin: 0 0 0.8rem 0; color: {text_color}; line-height: 1.7; font-size: 0.95rem;'>
            Users are welcome to extend this page to implement additional turbulence analysis quantities 
            and methods. Contributions can include:
        </p>
        <ul style='margin: 0; padding-left: 1.5rem; color: {text_color}; line-height: 1.8; font-size: 0.9rem;'>
            <li>Additional MRT-specific analysis tools and diagnostics</li>
            <li>SRT, BGK, and TRT-specific implementations and comparisons</li>
            <li>Navier-Stokes based HIT analysis integration</li>
            <li>New turbulence statistics and visualization methods</li>
            <li>Extended boundary condition support (beyond periodic)</li>
        </ul>
        <p style='margin: 0.8rem 0 0 0; color: {secondary_text}; font-size: 0.85rem; font-style: italic;'>
            The codebase is structured to facilitate extensions while maintaining the core HIT analysis framework.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

