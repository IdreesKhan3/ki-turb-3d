"""
Real Isotropy Theory Content — Single source of truth for Real Isotropy page Theory & Equations.

Pure content functions (no Streamlit). Used by pages/04_Real_Isotropy.py and agent tools.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
Subplot-specific content for agent (A–F); full page content for manual page.
"""

from typing import Optional

# Full page theory (matches manual page 04_Real_Isotropy.py expander)
FULL_PAGE_MARKDOWN = r"""
**Reynolds stress tensor:**
$$R_{ij} = \langle u'_i u'_j \rangle$$

where $u'_i = u_i - \langle u_i \rangle$ are velocity fluctuations and $\langle \cdot \rangle$ denotes ensemble or spatial average.

**Turbulent kinetic energy:**
$$k = \frac{1}{2}\langle u'_i u'_i \rangle = \frac{1}{2}(R_{11} + R_{22} + R_{33})$$

**Energy fractions:**
$$\frac{E_x}{E_{\text{tot}}} = \frac{R_{11}}{2k}, \quad \frac{E_y}{E_{\text{tot}}} = \frac{R_{22}}{2k}, \quad \frac{E_z}{E_{\text{tot}}} = \frac{R_{33}}{2k}$$
Isotropy implies each approaches $1/3$.

**Reynolds stress anisotropy tensor:**
$$b_{ij} = \frac{R_{ij}}{2k} - \frac{1}{3}\delta_{ij}$$

**Component form:**
$$
\begin{aligned}
b_{ii} &= \frac{R_{ii}}{2k} - \frac{1}{3}, \quad i = 1,2,3 \\
b_{ij} &= \frac{R_{ij}}{2k}, \quad i \neq j
\end{aligned}
$$

**Invariants:**
$$\text{II}_b = -\frac{1}{2}\mathrm{tr}(b^2), \qquad \text{III}_b = \frac{1}{3}\mathrm{tr}(b^3)$$

**Lumley coordinates:**
$$\eta = \left(-\frac{\text{II}_b}{3}\right)^{1/2}, \quad \xi = \left(\frac{\text{III}_b}{2}\right)^{1/3}$$

**Anisotropy index:**
$$A = \sqrt{-2 \text{II}_b}$$

---

**Reference:** [Pope (2001)](/Citation#pope2001) — Turbulent flows
""".strip()

# Subplot-specific theory (for agent get_real_isotropy_theory)
THEORY_BY_SUBPLOT = {
    "A": r"""
**Reynolds stress tensor:**
$$R_{ij} = \langle u'_i u'_j \rangle$$

where $u'_i = u_i - \langle u_i \rangle$ are velocity fluctuations and $\langle \cdot \rangle$ denotes ensemble or spatial average.

**Turbulent kinetic energy:**
$$k = \frac{1}{2}\langle u'_i u'_i \rangle = \frac{1}{2}(R_{11} + R_{22} + R_{33})$$

**Energy fractions (Subplot A):**
$$\frac{E_x}{E_{\text{tot}}} = \frac{R_{11}}{2k}, \quad \frac{E_y}{E_{\text{tot}}} = \frac{R_{22}}{2k}, \quad \frac{E_z}{E_{\text{tot}}} = \frac{R_{33}}{2k}$$

Isotropy implies each approaches $1/3$.

---

**Reference:** Pope (2001) — Turbulent flows
""",
    "B": r"""
**Reynolds stress anisotropy tensor:**
$$b_{ij} = \frac{R_{ij}}{2k} - \frac{1}{3}\delta_{ij}$$

**Invariants:**
$$\text{II}_b = -\frac{1}{2}\mathrm{tr}(b^2), \qquad \text{III}_b = \frac{1}{3}\mathrm{tr}(b^3)$$

**Lumley coordinates (Subplot B):**
$$\eta = \left(-\frac{\text{II}_b}{3}\right)^{1/2}, \quad \xi = \left(\frac{\text{III}_b}{2}\right)^{1/3}$$

---

**Reference:** Pope (2001) — Turbulent flows
""",
    "C": r"""
**Reynolds stress anisotropy tensor:**
$$b_{ij} = \frac{R_{ij}}{2k} - \frac{1}{3}\delta_{ij}$$

**Diagonal components (Subplot C):**
$$b_{ii} = \frac{R_{ii}}{2k} - \frac{1}{3}, \quad i = 1,2,3$$

Isotropy implies $b_{ii} \to 0$.

---

**Reference:** Pope (2001) — Turbulent flows
""",
    "D": r"""
**Reynolds stress anisotropy tensor:**
$$b_{ij} = \frac{R_{ij}}{2k} - \frac{1}{3}\delta_{ij}$$

**Component form:**
$$b_{ii} = \frac{R_{ii}}{2k} - \frac{1}{3}, \quad i = 1,2,3$$
$$b_{ij} = \frac{R_{ij}}{2k}, \quad i \neq j$$

**Invariants:**
$$\text{II}_b = -\frac{1}{2}\mathrm{tr}(b^2), \qquad \text{III}_b = \frac{1}{3}\mathrm{tr}(b^3)$$

**Anisotropy index (Subplot D):**
$$A = \sqrt{-2 \text{II}_b}$$

---

**Reference:** Pope (2001) — Turbulent flows
""",
    "E": r"""
**Energy fraction deviations (Subplot E):**
$$|E_x - 1/3|, \quad |E_y - 1/3|, \quad |E_z - 1/3|$$

where $E_x = R_{11}/(2k)$, etc. Isotropy implies each $E_i \to 1/3$, so deviations $\to 0$.

---

**Reference:** Pope (2001) — Turbulent flows
""",
    "F": r"""
**Running standard deviation (Subplot F):**
Measures temporal convergence of $E_x$, $E_y$, $E_z$ toward $1/3$. Decreasing std indicates statistical stationarity.

---

**Reference:** Pope (2001) — Turbulent flows
""",
}


def get_real_isotropy_theory_markdown(subplot: Optional[str] = None) -> str:
    """
    Return Real Isotropy theory markdown.

    - subplot=None: Full page content (for manual page and agent "full theory")
    - subplot in ("A","B","C","D","E","F"): Subplot-specific content (for agent)
    """
    if subplot and str(subplot).upper() in THEORY_BY_SUBPLOT:
        return THEORY_BY_SUBPLOT[str(subplot).upper()].strip()
    return FULL_PAGE_MARKDOWN
