"""
3D Volume Viewer Theory Content — Single source of truth for Theory & Equations.

Pure content functions (no Streamlit). Used by pages/11_3D_Volume_Viewer.py and agent tools.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
"""


def get_volume_viewer_theory_markdown() -> str:
    """Return 3D Volume Viewer Theory & Equations as markdown."""
    return """### Velocity Fields
**Velocity magnitude:**
$$|\\mathbf{u}| = \\sqrt{u_x^2 + u_y^2 + u_z^2}$$

### Vorticity
**Vorticity vector:**
$$\\boldsymbol{\\omega} = \\nabla \\times \\mathbf{u}$$

**Components:**
$$\\omega_x = \\frac{\\partial u_z}{\\partial y} - \\frac{\\partial u_y}{\\partial z}, \\quad \\omega_y = \\frac{\\partial u_x}{\\partial z} - \\frac{\\partial u_z}{\\partial x}, \\quad \\omega_z = \\frac{\\partial u_y}{\\partial x} - \\frac{\\partial u_x}{\\partial y}$$

**Vorticity magnitude:**
$$|\\boldsymbol{\\omega}| = \\sqrt{\\omega_x^2 + \\omega_y^2 + \\omega_z^2}$$

### Q_S^S Method for Vortex Visualization
**Main equation:**
$$Q_S^S = \\left[(Q_W^3 + Q_S^3) + (\\Sigma^2 - R_s^2)\\right]^{1/3}$$

**Component equations:**
- **Rotation Rate Strength:** $Q_W = \\frac{1}{2}\\Omega_{ij}\\Omega_{ij}$
- **Deformation Rate Strength:** $Q_S = -\\frac{1}{2}S_{ij}S_{ij}$
- **Enstrophy Production Term:** $\\Sigma = \\omega_i S_{ij} \\omega_j$
- **Strain Rate Production:** $R_s = -\\frac{1}{3}S_{ij}S_{jk}S_{ki}$

**Tensor definitions:**
- $\\Omega_{ij}$: Rotation tensor (antisymmetric part of velocity gradient)
- $S_{ij}$: Deformation tensor (symmetric part of velocity gradient)
- $\\omega_i$: Vorticity vector

### Velocity Gradient Tensor Invariants
**Second Invariant Q:**
$$Q = -\\frac{1}{2}A_{ij}A_{ij} = \\frac{1}{4}(\\omega_i\\omega_i - 2S_{ij}S_{ij})$$

**Third Invariant R:**
$$R = -\\frac{1}{3}A_{ij}A_{jk}A_{ki} = -\\frac{1}{3}\\left(S_{ij}S_{jk}S_{ki} + \\frac{3}{4}\\omega_i\\omega_j S_{ij}\\right)$$

where $A_{ij} = \\partial u_i/\\partial x_j$ is the velocity gradient tensor.
"""
