"""
Energy Spectra Theory Content — Single source of truth for Energy Spectra page Theory & Equations.

Pure content functions (no Streamlit). Used by pages/06_Energy_Spectra.py and agent tools.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
"""


def get_energy_spectra_theory_markdown() -> str:
    """Return Energy Spectra page Theory & Equations as markdown."""
    return r"""
**3D kinetic energy spectrum (Fourier space):**
$$E(k) = \sum_{k \le |\mathbf{k}| < k + \Delta k} \frac{1}{2} \left( |\hat{u}(\mathbf{k})|^2 + |\hat{v}(\mathbf{k})|^2 + |\hat{w}(\mathbf{k})|^2 \right)$$

**Total kinetic energy and RMS velocity:**
$$\mathrm{TKE} = \sum_k E(k), \qquad u_{\mathrm{rms}} = \sqrt{\frac{2}{3} \mathrm{TKE}}$$

**Kolmogorov inertial-range scaling:**
$$E(k) \propto k^{-5/3}$$

**Pope model spectrum (FHIT spectra validation):**
$$E_{\text{pope}}(k) = C \varepsilon^{2/3} k^{-5/3} f_L(kL) f_\eta(k\eta)$$

with $C=1.5$, $c_L=6.78$, $c_\eta=0.40$, $\beta=5.2$. Here $f_L$ and $f_\eta$ are large-scale and dissipation-range corrections.

**Spectral dissipation:**
$$\varepsilon = 2\nu \sum_k k^2 E(k)$$

**Normalized spectrum:**
$$E_{\text{norm}}(k\eta) = \frac{E(k)}{\varepsilon^{2/3} \eta^{5/3}}, \quad k\eta = k \cdot \eta$$

where $\eta = (\nu^3/\varepsilon)^{1/4}$ is the Kolmogorov length scale and $k\eta$ is the normalized wavenumber (dimensionless). The normalized spectrum is plotted in the **Normalized Spectrum** tab.

---

**Reference:** [Pope (2001)](/Citation#pope2001) — Turbulent flows
""".strip()
