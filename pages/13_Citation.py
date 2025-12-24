"""
Citation Page
How to cite this software + key references
"""

from __future__ import annotations

import streamlit as st
import sys
from pathlib import Path
from textwrap import dedent

# --- App imports (keep consistent with your project structure) ---
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css  # noqa: E402


# IMPORTANT: set_page_config must be called before any other Streamlit calls
st.set_page_config(page_icon="⚫", layout="wide")


# ==========================================================
# BibTeX Data
# ==========================================================
BIBTEX = dedent(
    r"""
    @article{d2002multiple,
      title     = {Multiple-relaxation-time lattice Boltzmann models in three dimensions},
      author    = {d'Humi{\`e}res, Dominique},
      journal   = {Philosophical Transactions of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences},
      volume    = {360},
      number    = {1792},
      pages     = {437-451},
      year      = {2002},
      publisher = {The Royal Society}
    }

    @article{pope2001turbulent,
      title   = {Turbulent flows},
      author  = {Pope, Stephen B},
      journal = {Measurement Science and Technology},
      volume  = {12},
      number  = {11},
      pages   = {2020-2021},
      year    = {2001}
    }

    @article{yu2006turbulent,
      title     = {LES of turbulent square jet flow using an MRT lattice Boltzmann model},
      author    = {Yu, Huidan and Luo, Li-Shi and Girimaji, Sharath S},
      journal   = {Computers \& fluids},
      volume    = {35},
      number    = {8-9},
      pages     = {957-965},
      year      = {2006},
      publisher = {Elsevier}
    }

    @article{kruger2017lattice,
      title     = {The lattice Boltzmann method},
      author    = {Kr{\"u}ger, Timm and Kusumaatmaja, Halim and Kuzmin, Alexandr and Shardt, Orest and Silva, Goncalo and Viggen, Erlend Magnus},
      journal   = {Springer International Publishing},
      volume    = {10},
      number    = {978-3},
      pages     = {4-15},
      year      = {2017},
      publisher = {Springer}
    }

    @article{benzi1993extended,
      title     = {Extended self-similarity in turbulent flows},
      author    = {Benzi, Roberto and Ciliberto, Sergio and Tripiccione, Raffaele and Baudet, Christophe and Massaioli, F and Succi, S},
      journal   = {Physical review E},
      volume    = {48},
      number    = {1},
      pages     = {R29},
      year      = {1993},
      publisher = {APS}
    }

    @article{she1994universal,
      title     = {Universal scaling laws in fully developed turbulence},
      author    = {She, Zhen-Su and Leveque, Emmanuel},
      journal   = {Physical review letters},
      volume    = {72},
      number    = {3},
      pages     = {336},
      year      = {1994},
      publisher = {APS}
    }

    @article{kareem2022simulations,
      title     = {Simulations of isotropic turbulent flows using lattice Boltzmann method with different forcing functions},
      author    = {Kareem, Waleed Abdel and Asker, Zafer M},
      journal   = {International journal of modern Physics C},
      volume    = {33},
      number    = {11},
      pages     = {2250145},
      year      = {2022},
      publisher = {World Scientific}
    }

    @book{batchelor1953theory,
      title     = {The theory of homogeneous turbulence},
      author    = {Batchelor, George Keith},
      year      = {1953},
      publisher = {Cambridge university press}
    }

    @article{singh2024comparison,
      title     = {Comparison of forcing schemes to sustain homogeneous isotropic turbulence},
      author    = {Singh, Kamaljit and Komrakova, Alexandra},
      journal   = {Physics of Fluids},
      volume    = {36},
      number    = {1},
      year      = {2024},
      publisher = {AIP Publishing}
    }
    """
).strip()


# ==========================================================
# Human-readable Reference List
# ==========================================================
REFERENCES = [
    {
        "id": "dhumieres2002",
        "key": "d2002multiple",
        "short": "d’Humières (2002)",
        "full": (
            "d’Humières, D. (2002). Multiple-relaxation-time lattice Boltzmann models in three dimensions. "
            "Philosophical Transactions of the Royal Society of London. Series A: Mathematical, Physical and "
            "Engineering Sciences, 360(1792), 437–451."
        ),
    },
    {
        "id": "pope2001",
        "key": "pope2001turbulent",
        "short": "Pope (2001)",
        "full": (
            "Pope, S. B. (2001). Turbulent flows. Measurement Science and Technology, 12(11), 2020–2021."
        ),
    },
    {
        "id": "yu2006",
        "key": "yu2006turbulent",
        "short": "Yu et al. (2006)",
        "full": (
            "Yu, H., Luo, L.-S., & Girimaji, S. S. (2006). LES of turbulent square jet flow using an MRT lattice "
            "Boltzmann model. Computers & Fluids, 35(8-9), 957-965."
        ),
    },
    {
        "id": "kruger2017",
        "key": "kruger2017lattice",
        "short": "Krüger et al. (2017)",
        "full": (
            "Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., & Viggen, E. M. (2017). "
            "The lattice Boltzmann method. Springer International Publishing."
        ),
    },
    {
        "id": "benzi1993",
        "key": "benzi1993extended",
        "short": "Benzi et al. (1993)",
        "full": (
            "Benzi, R., Ciliberto, S., Tripiccione, R., Baudet, C., Massaioli, F., & Succi, S. (1993). "
            "Extended self-similarity in turbulent flows. Physical Review E, 48(1), R29."
        ),
    },
    {
        "id": "she1994",
        "key": "she1994universal",
        "short": "She & Leveque (1994)",
        "full": (
            "She, Z.-S., & Leveque, E. (1994). Universal scaling laws in fully developed turbulence. "
            "Physical Review Letters, 72(3), 336."
        ),
    },
    {
        "id": "kareem2022",
        "key": "kareem2022simulations",
        "short": "Kareem & Asker (2022)",
        "full": (
            "Kareem, W. A., & Asker, Z. M. (2022). Simulations of isotropic turbulent flows using lattice Boltzmann "
            "method with different forcing functions. International Journal of Modern Physics C, 33(11), 2250145."
        ),
    },
    {
        "id": "batchelor1953",
        "key": "batchelor1953theory",
        "short": "Batchelor (1953)",
        "full": (
            "Batchelor, G. K. (1953). The theory of homogeneous turbulence. Cambridge University Press."
        ),
    },
    {
        "id": "singh2024",
        "key": "singh2024comparison",
        "short": "Singh & Komrakova (2024)",
        "full": (
            "Singh, K., & Komrakova, A. (2024). Comparison of forcing schemes to sustain homogeneous isotropic "
            "turbulence. Physics of Fluids, 36(1)."
        ),
    },
]

# ==========================================================
# Helper Functions
# ==========================================================
def _anchor(anchor_id: str) -> None:
    """Create a stable HTML anchor for deep-linking to a reference."""
    st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)

# ==========================================================
# Main
# ==========================================================
def main() -> None:
    inject_theme_css()

    st.title("Citation & References")

    # =========================
    # Software Citation Section
    # =========================
    st.subheader("How to cite this software")
    st.markdown(
        "If you use **KI-TURB 3D** in your research, please cite the specific released version used."
    )

    # BibTeX format
    st.markdown("**BibTeX format:**")
    bibtex_citation = dedent(
        r"""
        @software{ki_turb_3d,
          title   = {KI-TURB 3D: Turbulence Analysis and Visualization Suite},
          author  = {Muhammad Idrees Khan},
          year    = {2025},
          version = {1.0.0},
          url     = {https://github.com/IdreesKhan3/ki-turb-3d},
          license = {MIT}
        }
        """
    ).strip()
    st.code(bibtex_citation, language="bibtex")

    # APA format
    st.markdown("**APA format:**")
    apa_citation = (
        "Khan, M. I. (2025). "
        "KI-TURB 3D: Turbulence Analysis and Visualization Suite "
        "(Version 1.0.0) [Computer software]. "
        "https://github.com/IdreesKhan3/ki-turb-3d"
    )
    st.code(apa_citation, language="text")

    st.info(
        "A persistent DOI is not yet assigned for this software. "
        "Please use the software version and repository URL for citation. "
        "A DOI will be added in a future release."
    )

    st.divider()

    # =========================
    # Scientific References Section
    # =========================
    st.subheader("Key scientific references")
    st.caption("These are the core methodological and turbulence references used in this project.")

    # Render references with anchors so other pages can link to /Citation#<id>
    for ref in REFERENCES:
        _anchor(ref["id"])
        st.markdown(f"**[{ref['short']}]**  \n{ref['full']}  \n`BibTeX key:` `{ref['key']}`")

    st.divider()

    # =========================
    # BibTeX Export Section
    # =========================
    st.subheader("BibTeX (all references)")
    st.code(BIBTEX, language="bibtex")

    st.download_button(
        label="Download references.bib",
        data=(BIBTEX + "\n").encode("utf-8"),
        file_name="references.bib",
        mime="application/x-bibtex",
    )


if __name__ == "__main__":
    main()
