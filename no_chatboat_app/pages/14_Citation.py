"""
Citation Page
How to cite this software
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
    st.title("📖 Citation")
    
    st.markdown("""
    ## How to Cite This Software
    
    [Citation information will be added here]
    
    ### BibTeX
    
    ```bibtex
    @software{turbulence_dashboard_2024,
      title = {LBM-MRT Turbulence Analysis Dashboard},
      author = {[Your Name]},
      year = {2024},
      url = {[GitHub URL]},
      doi = {[DOI when available]}
    }
    ```
    """)

if __name__ == "__main__":
    main()

