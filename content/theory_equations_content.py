"""
Theory Equations Content — Single source of truth for NS-Equations and LBM Formulation tabs.

Pure content functions (no Streamlit). Used by pages/02_Theory_Equations.py and agent tools.
Returns structured sections: (title, content, expanded) for each expander.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
"""

from typing import List, Optional, Tuple

# For LBM: (title, content, expanded). When title is None, content is rendered as header markdown (no expander).
LbmSection = Tuple[Optional[str], str, bool]


def get_ns_equations_sections() -> List[Tuple[str, str, bool]]:
    """Return NS-Equations tab content as (title, markdown_content, expanded) sections."""
    return [
        (
            "**1. Navier-Stokes Equations**",
            """
**Incompressible flow equations:**

$$
\\begin{aligned}
\\nabla \\cdot \\mathbf{u} &= 0 \\\\
\\frac{\\partial \\mathbf{u}}{\\partial t} + (\\mathbf{u}\\cdot\\nabla)\\mathbf{u} &= -\\frac{1}{\\rho}\\nabla p + \\nu\\nabla^2\\mathbf{u} + \\mathbf{f}
\\end{aligned}
$$

*Continuity and momentum conservation equations*
            """.strip(),
            True,
        ),
        (
            "**2. Filtered Navier-Stokes (LES)**",
            r"""
Applying spatial filter $\overline{(\cdot)}$ to Navier-Stokes:

$$
\frac{\partial \bar{u}_i}{\partial t} + \frac{\partial}{\partial x_j}(\overline{u_i u_j}) = -\frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \nu \nabla^2 \bar{u}_i + \bar{f}_i
$$

**Decomposition:** $\overline{u_i u_j} = \bar{u}_i \bar{u}_j + \tau_{ij}^{\mathrm{sgs}}$

$$
\frac{\partial \bar{u}_i}{\partial t} + \frac{\partial}{\partial x_j} \left( \bar{u}_i \bar{u}_j + \tau_{ij}^{\mathrm{sgs}} \right) = - \frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \nu \, \nabla^2 \bar{u}_i + \bar{f}_i
$$

where $\tau_{ij}^{\mathrm{sgs}} = \overline{u_i u_j} - \bar{u}_i \bar{u}_j$ is the **subgrid-scale stress tensor** requiring closure.

---

**Eddy-viscosity closure (Smagorinsky):**

$$
\begin{aligned}
\tau_{ij}^{\mathrm{sgs}} - \tfrac{1}{3}\tau_{kk}^{\mathrm{sgs}} \, \delta_{ij} &= -2 \, \nu_t \, \bar{S}_{ij} \\
\nu_t &= (C_s \, \Delta)^2 \, |\bar{S}|
\end{aligned}
$$

where $\bar{S}_{ij} = \tfrac{1}{2}(\partial_i \bar{u}_j + \partial_j \bar{u}_i)$ and $|\bar{S}| = (2\,\bar{S}_{ij}\bar{S}_{ij})^{1/2}$
            """.strip(),
            False,
        ),
    ]


def get_ns_equations_footer() -> str:
    """Return NS-Equations tab footer/reference."""
    return "**Reference:** All equations in this section can be found in [Pope (2001)](/Citation#pope2001)."


def get_ns_equations_markdown() -> str:
    """Return full NS-Equations tab as a single markdown string (for agent/export)."""
    sections = get_ns_equations_sections()
    parts = ["# From Navier-Stokes to LBM\n"]
    for title, content, _ in sections:
        parts.append(f"## {title.strip('*').replace('*', '')}\n")
        parts.append(content)
        parts.append("")
    parts.append(get_ns_equations_footer())
    return "\n".join(parts)


def get_lbm_formulation_sections() -> List[LbmSection]:
    """Return LBM Formulation tab content. Each item is (title, content, expanded).
    When title is None, content is rendered as header markdown (no expander).
    """
    return [
        (
            "**MRT - DNS Formulation** (Primary)",
            """
**MRT-LBM evolution equation (D3Q19):**

$$
f_\\alpha(\\mathbf{x} + \\mathbf{c}_\\alpha \\delta t, t + \\delta t) - f_\\alpha(\\mathbf{x}, t) = 
- \\left[ \\mathbf{M}^{-1} \\mathbf{\\Lambda} \\mathbf{M} (\\mathbf{f} - \\mathbf{f}^{eq}) \\right]_\\alpha 
+ \\delta t \\left[ \\mathbf{M}^{-1} \\left( \\mathbf{I} - \\frac{\\mathbf{\\Lambda}}{2} \\right) \\mathbf{M} \\mathbf{\\Phi} \\right]_\\alpha
$$

*Multiple relaxation times provide better stability and accuracy*

**Relationship between $\\nu$ and $\\tau$ in MRT:**

$$
\\tau = 3\\nu + 0.5, \\quad \\nu = \\frac{1}{3}\\left(\\frac{1}{s_\\nu} - \\frac{1}{2}\\right)
$$

*General form relating relaxation time $\\tau$ to kinematic viscosity $\\nu$*

**Transformation matrix $\\mathbf{M}$ (D3Q19):**

The 19×19 transformation matrix $\\mathbf{M}$ is constructed using the orthogonal moment basis ([d'Humières, 2002](/Citation#dhumieres2002)).
The matrix transforms distribution functions $f_\\alpha$ to moment space: $\\mathbf{m} = \\mathbf{M} \\mathbf{f}$.

*Full 19×19 matrix display available in MRT Matrix Generator tool*

**Relaxation matrix $\\mathbf{\\Lambda}$ (diagonal):**

$$
\\mathbf{\\Lambda} = \\text{diag}(1.0, 1.19, 1.4, 1.0, 1.2, 1.0, 1.2, 1.0, 1.2, s_\\nu, 1.4, s_\\nu, 1.4, s_\\nu, s_\\nu, s_\\nu, 1.98, 1.98, 1.98)
$$

*where $s_\\nu$ is the viscosity-related relaxation parameter*

**Equilibrium moments $\\mathbf{m}^{(eq)}$:**

$$
\\mathbf{m}^{(eq)} = \\begin{bmatrix}
\\delta\\rho = \\rho - \\rho_0 \\\\
-11\\delta\\rho + 19\\rho(u_x^2 + u_y^2 + u_z^2) \\\\
-\\frac{475}{63}\\rho(u_x^2 + u_y^2 + u_z^2) \\\\
\\rho u_x, \\quad -\\frac{2\\rho u_x}{3} \\\\
\\rho u_y, \\quad -\\frac{2\\rho u_y}{3} \\\\
\\rho u_z, \\quad -\\frac{2\\rho u_z}{3} \\\\
\\rho(2u_x^2 - u_y^2 - u_z^2), \\quad 0 \\\\
\\rho(u_y^2 - u_z^2), \\quad 0 \\\\
\\rho u_x u_y, \\quad \\rho u_y u_z, \\quad \\rho u_x u_z \\\\
0, \\quad 0, \\quad 0
\\end{bmatrix}
$$

**Force moments $\\mathbf{F}_m$:**

$$
\\mathbf{F}_m = \\begin{bmatrix}
0 \\\\
38(u_x F_x + u_y F_y + u_z F_z) \\\\
-11(u_x F_x + u_y F_y + u_z F_z) \\\\
F_x, \\quad -\\frac{2F_x}{3} \\\\
F_y, \\quad -\\frac{2F_y}{3} \\\\
F_z, \\quad -\\frac{2F_z}{3} \\\\
2(2u_x F_x - u_y F_y - u_z F_z), \\quad -(2u_x F_x - u_y F_y - u_z F_z) \\\\
2(u_y F_y - u_z F_z), \\quad -(u_y F_y - u_z F_z) \\\\
u_x F_y + u_y F_x, \\quad u_y F_z + u_z F_y, \\quad u_x F_z + u_z F_x \\\\
0, \\quad 0, \\quad 0
\\end{bmatrix}
$$

**Equilibrium distribution:**

$$
f_\\alpha^{eq} = w_\\alpha \\rho \\left[ 1 + \\frac{\\mathbf{c}_\\alpha \\cdot \\mathbf{u}}{c_s^2} + \\frac{(\\mathbf{c}_\\alpha \\cdot \\mathbf{u})^2}{2c_s^4} - \\frac{\\mathbf{u} \\cdot \\mathbf{u}}{2c_s^2} \\right]
$$

**Guo's forcing term:**

$$
\\Phi_\\alpha = w_\\alpha \\left[ \\frac{\\mathbf{c}_\\alpha - \\mathbf{u}}{c_s^2} + \\frac{(\\mathbf{c}_\\alpha \\cdot \\mathbf{u})\\mathbf{c}_\\alpha}{c_s^4} \\right] \\cdot \\mathbf{F}^{\\text{ext}}
$$

**External forcing components** $\\mathbf{F}^{\\mathrm{ext}}$:

$$
\\begin{aligned}
F_x &= \\rho \\left[ A \\sin(\\kappa z) + C \\cos(\\kappa y) \\right] \\\\
F_y &= \\rho \\left[ B \\sin(\\kappa x) + A \\cos(\\kappa z) \\right] \\\\
F_z &= \\rho \\left[ C \\sin(\\kappa y) + B \\cos(\\kappa x) \\right]
\\end{aligned}
$$

where $A$, $B$, and $C$ are forcing amplitudes and $\\kappa$ is the wavenumber.

**Macroscopic quantities:**

$$
\\rho = \\sum_\\alpha f_\\alpha, \\quad \\rho \\mathbf{u} = \\sum_\\alpha f_\\alpha \\mathbf{c}_\\alpha
$$
            """.strip(),
            True,
        ),
        (
            "**MRT - LES Formulation** (Primary)",
            """
**Effective viscosity approach:** ([Yu et al., 2006](/Citation#yu2006))

$$
\\begin{aligned}
\\nu_e &= \\nu_0 + \\nu_t \\\\
\\frac{1}{s_\\nu} &= \\frac{1}{2} + 3(\\nu_0 + \\nu_t) \\equiv \\tau_e
\\end{aligned}
$$

**LES-MRT evolution:**

$$
f_\\alpha(\\mathbf{x} + \\mathbf{c}_\\alpha \\delta t, t + \\delta t) - f_\\alpha(\\mathbf{x}, t) = 
- \\left[ \\mathbf{M}^{-1} \\mathbf{\\Lambda}(\\nu_e) \\mathbf{M} (\\mathbf{f} - \\mathbf{f}^{eq}) \\right]_\\alpha 
+ \\delta t \\left[ \\mathbf{M}^{-1} \\left( \\mathbf{I} - \\frac{\\mathbf{\\Lambda}(\\nu_e)}{2} \\right) \\mathbf{M} \\mathbf{\\Phi} \\right]_\\alpha
$$

where $\\mathbf{\\Lambda}(\\nu_e)$ uses effective viscosity $\\nu_e = \\nu_0 + \\nu_t$

**Strain rate tensor from non-equilibrium moments:**

The components of the filtered strain-rate tensor are computed from non-equilibrium moments:

$$
\\begin{aligned}
S_{xx} &= -\\frac{s_1 m_1^{(neq)} + 19s_9 m_9^{(neq)}}{38\\rho_0\\delta_t} \\\\
S_{yy} &= -\\frac{2s_1 m_1^{(neq)} - 19s_9(m_9^{(neq)} - 3m_{11}^{(neq)})}{76\\rho_0\\delta_t} \\\\
S_{zz} &= -\\frac{2s_1 m_1^{(neq)} - 19s_9(m_9^{(neq)} + 3m_{11}^{(neq)})}{76\\rho_0\\delta_t} \\\\
S_{xy} &= -\\frac{3s_9}{2\\rho_0\\delta_t} m_{13}^{(neq)}, \\quad
S_{xz} = -\\frac{3s_9}{2\\rho_0\\delta_t} m_{15}^{(neq)}, \\quad
S_{yz} = -\\frac{3s_9}{2\\rho_0\\delta_t} m_{14}^{(neq)}
\\end{aligned}
$$

where $m_i^{(neq)}$ are non-equilibrium moments and $s_i$ are relaxation parameters

**Alternative form (from non-equilibrium stress tensor):**

$$
S_{ab} = \\frac{P_{ab}^{\\text{ne}}}{\\rho c_s^2 \\tau_e}, \\quad P_{ab}^{\\text{ne}} = \\sum_\\alpha f_\\alpha^{\\text{ne}} c_{\\alpha a} c_{\\alpha b}
$$
            """.strip(),
            True,
        ),
        (None, "---\n### BGK/SRT", False),
        (
            "**BGK (SRT) - DNS**",
            """
$$
f_\\alpha(\\mathbf{x} + \\mathbf{c}_\\alpha \\delta t, t + \\delta t) - f_\\alpha(\\mathbf{x}, t) = 
-\\frac{1}{\\tau} \\left(f_\\alpha(\\mathbf{x}, t) - f_\\alpha^{eq}(\\mathbf{x}, t)\\right) + \\mathbf{F}_\\alpha^{\\text{ext}}
$$

$$
\\nu = c_s^2 \\left(\\tau - \\frac{1}{2}\\right) \\delta t, \\quad \\mathbf{F}_\\alpha^{\\text{ext}} = f_\\alpha^{eq,\\text{shift}} - f_\\alpha^{eq}
$$
            """.strip(),
            False,
        ),
        (
            "**BGK (SRT) - LES**",
            """
*Shown for reference - app can analyze BGK/SRT data, but MRT is primary*

$$
\\bar{f}_\\alpha(\\mathbf{x} + \\mathbf{c}_\\alpha \\delta t, t + \\delta t) - \\bar{f}_\\alpha(\\mathbf{x}, t) = 
-\\frac{1}{\\tau_e} \\left(\\bar{f}_\\alpha(\\mathbf{x}, t) - \\bar{f}_\\alpha^{eq}(\\mathbf{x}, t)\\right) + 3\\rho w_\\alpha (\\mathbf{c}_\\alpha \\cdot \\bar{\\mathbf{F}})
$$

$$
\\tau_e = 3(\\nu_0 + C \\Delta^2 |\\bar{S}_{ab}|) + \\frac{1}{2}
$$
            """.strip(),
            False,
        ),
        (
            "**Continuous Boltzmann Equation** (Reference)",
            """
$$
\\frac{\\partial f}{\\partial t} + \\xi_\\alpha \\frac{\\partial f}{\\partial x_\\alpha} + \\mathbf{F}_\\alpha \\frac{\\partial f}{\\partial \\xi_\\alpha} = \\Omega(f)
$$

*Foundation: continuous kinetic equation*
            """.strip(),
            False,
        ),
        (None, "---\n### Validation and Flow Characterization", False),
        (
            "**Compressibility and Incompressibility**",
            r"""
**Velocity divergence (incompressibility condition):**

$$
\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z} = 0
$$

**Compressibility metrics:**

$$
|\nabla \cdot \mathbf{u}|_{\max} = \max_{x,y,z} |\nabla \cdot \mathbf{u}|, \quad 
\text{RMS}(\nabla \cdot \mathbf{u}) = \sqrt{\frac{1}{V} \int_V (\nabla \cdot \mathbf{u})^2 \, dV}
$$

For incompressible flow: $|\nabla \cdot \mathbf{u}|_{\max} < 10^{-5}$ (valid), 
$10^{-5} < |\nabla \cdot \mathbf{u}|_{\max} < 10^{-3}$ (warning), 
$|\nabla \cdot \mathbf{u}|_{\max} > 10^{-3}$ (invalid).
            """.strip(),
            False,
        ),
        (
            "**Physics Validation Parameters**",
            r"""
**Mach Number:**

$$
\text{Ma} = \frac{u_{\text{rms}}}{c_s}
$$

where $c_s = 1/\sqrt{3}$ is the lattice sound speed. For incompressible flow: $\text{Ma} < 0.1$

**Knudsen Number (DNS/Continuum Check):**

$$
\text{Kn} = \frac{c_s (\tau_0 - 1/2) \Delta x}{\Delta x} = c_s \left(\tau_0 - \frac{1}{2}\right)
$$

where $\tau_0 = \nu_0/c_s^2 + 1/2$ is the molecular relaxation time. Continuum regime: $\text{Kn} < 0.01$

**Knudsen Number (LES):**

$$
\text{Kn}_t = \frac{(\tau_e - 1/2) \sqrt{3} \Delta x}{\Delta x} = \sqrt{3} \left(\tau_e - \frac{1}{2}\right)
$$

where $\tau_e = 3(\nu_0 + \nu_t) + 1/2$ is the effective relaxation time. Continuum regime: $\text{Kn}_t < 0.01$
            """.strip(),
            False,
        ),
        (
            "**Reynolds Numbers**",
            r"""
**Bulk Reynolds number:**

$$
\text{Re}_B = \frac{U L}{\nu}
$$

where $U$ is the characteristic velocity (typically reference or bulk velocity) and $L$ is the characteristic length scale (typically domain size or reference length).

**Taylor Reynolds number:**

$$
\text{Re}_T = \frac{u_{\text{rms}} \lambda_T}{\nu}
$$

where $\lambda_T$ is the Taylor microscale.

**Integral Reynolds number:**

$$
\text{Re}_L = \frac{u_{\text{rms}} L_I}{\nu}
$$

where $L_I$ is the integral length scale.
            """.strip(),
            False,
        ),
    ]


def get_lbm_formulation_footer() -> str:
    """Return LBM Formulation tab footer/references."""
    return """
**References:**
- **MRT formulation:** [d'Humières (2002)](/Citation#dhumieres2002) — Multiple-relaxation-time lattice Boltzmann models
- **LES with MRT:** [Yu et al. (2006)](/Citation#yu2006) — LES of turbulent flows using MRT-LBM
- **General LBM method:** [Krüger et al. (2017)](/Citation#kruger2017) — The lattice Boltzmann method
    """.strip()


def get_lbm_formulation_markdown() -> str:
    """Return full LBM Formulation tab as a single markdown string (for agent/export)."""
    sections = get_lbm_formulation_sections()
    parts = [
        "# Lattice Boltzmann Method\n",
        "**Primary focus:** MRT (Multiple Relaxation Time) | **Reference:** BGK/SRT (shown for app flexibility)\n",
    ]
    for title, content, _ in sections:
        if title is None:
            parts.append(content)
        else:
            parts.append(f"## {title.strip('*').replace('*', '')}\n")
            parts.append(content)
        parts.append("")
    parts.append(get_lbm_formulation_footer())
    return "\n".join(parts)
