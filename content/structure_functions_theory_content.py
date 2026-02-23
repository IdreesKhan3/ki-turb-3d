"""
Structure Functions Theory Content — Single source of truth for Structure Functions page Theory & Equations.

Pure content functions (no Streamlit). Used by pages/08_Structure_Functions.py and agent tools.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
"""


def get_structure_functions_theory_markdown() -> str:
    """Return Structure Functions page Theory & Equations as markdown."""
    return r"""
**Structure functions:**
$$S_p(r) = \langle |\delta u_L(r)|^p \rangle$$

where $\delta u_L(r) = u_L(\mathbf{x} + r\mathbf{e}_L) - u_L(\mathbf{x})$ is the longitudinal velocity increment.

**Extended Self-Similarity (ESS):** ([Benzi et al., 1993](/Citation#benzi1993))
$$S_p(r) \propto S_3(r)^{\xi_p}$$

The scaling exponent $\xi_p$ is obtained from the slope of $\log S_p$ vs $\log S_3$.

**She–Leveque 1994 scaling (theoretical):** ([She & Leveque, 1994](/Citation#she1994))
$$\zeta_p = \frac{p}{9} + 2\left(1 - \left(\frac{2}{3}\right)^{p/3}\right)$$

Anomalies are plotted as $\xi_p - p/3$ to compare with theoretical predictions.
""".strip()
