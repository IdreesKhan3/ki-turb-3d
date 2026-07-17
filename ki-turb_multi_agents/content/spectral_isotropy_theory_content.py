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

**Standard Spectral Isotropy Ratio (plotted in IC(k) Time-Avg):**
$$\text{IC}(k) = \frac{E_{22}(k)}{E_{11}(k)}$$

For isotropic turbulence the three shell-component spectra are equal, so $\text{IC}(k)\approx 1$.
The time-average curve uses $\langle E_{22}\rangle/\langle E_{11}\rangle$ across snapshots
(under-resolved shells with $E_{11}$ below threshold are omitted).

**Derivative-based Spectral Isotropy Ratio (diagnostic):**
$$\text{IC}_{\text{deriv}}(k) = \frac{2E_{22}(k) - k_{\mathrm{phys}} \frac{dE_{11}}{dk}}{2E_{11}(k)}$$

with $k_{\mathrm{phys}} = k\cdot 2\pi/N$. This is column 7 of `isotropy_coeff_*.dat`;
it can be noisy near the Nyquist shell where $E_{11}$ is tiny.

---

**References:** [Batchelor (1953)](/Citation#batchelor1953) — The theory of homogeneous turbulence; [Singh & Komrakova (2024)](/Citation#singh2024) — Comparison of forcing schemes to sustain homogeneous isotropic turbulence
""".strip()
