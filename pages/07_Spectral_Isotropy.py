"""
Spectral Isotropy Page
Spectral isotropy analysis using isotropy_coeff files
"""

import streamlit as st
from pathlib import Path
import sys

# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualizations.isotropy_spectral import build_spectral_isotropy_fig
from utils.file_detector import detect_simulation_files


def main():
    st.title("🔄 Spectral Isotropy")
    
    # Get data directory from session state
    data_dir = st.session_state.get('data_directory', None)
    if not data_dir:
        st.warning("Please select a data directory from the Overview page.")
        return
    
    data_dir = Path(data_dir)
    
    # Detect files
    files_dict = detect_simulation_files(str(data_dir))
    isotropy_files = files_dict.get('isotropy', [])
    
    if not isotropy_files:
        st.info("No spectral isotropy files found. Expected format: `isotropy_coeff_*.dat`")
        return
    
    try:
        fig = build_spectral_isotropy_fig(data_dir)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting spectral isotropy: {e}")
        st.exception(e)
    
    # Theory section
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(r"""
        **Spectral Isotropy Coefficient**
        
        \[
        IC(k) = \frac{E_{22}(k)}{E_{11}(k)}
        \]
        
        Where $E_{ii}(k)$ is the energy spectrum in direction $i$.
        
        - $IC(k) = 1$: Perfect isotropy
        - $IC(k) \neq 1$: Anisotropy
        
        **Alternative Form (Derivative-based)**
        \[
        IC_{deriv}(k) = \frac{dE_{22}/dk}{dE_{11}/dk}
        \]
        
        The derivative-based form is more robust for numerical data.
        """)


if __name__ == "__main__":
    main()

