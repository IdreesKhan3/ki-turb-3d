"""
Flatness Theory Content — Single source of truth for Flatness page Theory & Equations.

Pure content functions (no Streamlit). Used by pages/07_Flatness.py and agent tools.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
"""


def get_flatness_theory_markdown() -> str:
    """Return Flatness page Theory & Equations as markdown."""
    return r"""
**Longitudinal flatness factor:**
$$F_L(r) = \frac{\langle [\delta u_L(r)]^4 \rangle}{\langle [\delta u_L(r)]^2 \rangle^2}$$

where $\delta u_L(r) = u_L(\mathbf{x} + r\mathbf{e}_L) - u_L(\mathbf{x})$ is the longitudinal velocity increment.

**Interpretation:**
- $F_L(r) = 3$: Gaussian increments (no intermittency)
- $F_L(r) > 3$: Intermittent, fat-tailed PDFs
- $F_L(r) < 3$: Sub-Gaussian

---

**Reference:** [Pope (2001)](/Citation#pope2001) — Turbulent flows
""".strip()
