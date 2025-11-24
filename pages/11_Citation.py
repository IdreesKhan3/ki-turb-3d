"""
Citation Page
How to cite this software
"""

import streamlit as st

def main():
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

