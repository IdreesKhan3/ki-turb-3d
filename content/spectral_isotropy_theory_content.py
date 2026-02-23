"""
Spectral Isotropy Theory Content — Single source of truth for Spectral Isotropy page Theory & Equations.

Pure content functions (no Streamlit). Used by pages/05_Spectral_Isotropy.py and agent tools.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
"""


def get_spectral_isotropy_theory_markdown() -> str:
    """Return Spectral Isotropy page Theory & Equations as markdown."""
    return r"""
**One-dimensional energy spectra:**
$$E_{11}(k) = |\hat{u}(k)|^2, \quad E_{22}(k) = |\hat{v}(k)|^2, \quad E_{33}(k) = |\hat{w}(k)|^2$$

where $\hat{u}(k)$, $\hat{v}(k)$, and $\hat{w}(k)$ are the Fourier transforms of velocity components $u$, $v$, and $w$ in the $x$, $y$, and $z$ directions, respectively. These are plotted in the **Component Spectra** tab.

**Derivative-based Spectral Isotropy Ratio:**
$$\text{IC}_{\text{deriv}}(k) = \frac{2E_{22}(k) - k \frac{dE_{11}}{dk}}{2E_{11}(k)}$$

The derivative-based formula includes the spectral derivative term, making it less sensitive to numerical noise when $E_{22}(k)$ is small. The ratio $\text{IC}_{\text{deriv}}(k)$ is plotted as a function of wavenumber $k$ in the **IC(k) Time-Avg** tab, and summary statistics (mean, std, min, max) are shown in the **Summary** tab.

**For isotropic turbulence:**
$$E_{11}(k) = E_{22}(k) = E_{33}(k) \quad \Rightarrow \quad \text{IC}_{\text{deriv}}(k) \approx 1$$

---

**References:** [Batchelor (1953)](/Citation#batchelor1953) — The theory of homogeneous turbulence; [Singh & Komrakova (2024)](/Citation#singh2024) — Comparison of forcing schemes to sustain homogeneous isotropic turbulence
""".strip()
