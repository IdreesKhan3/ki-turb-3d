"""
Overview Theory Content — Single source of truth for Overview page Physics Validation Equations.

Pure content functions (no Streamlit). Used by pages/01_Overview.py and agent tools.
Content uses markdown with LaTeX ($...$ inline, $$...$$ block).
"""


def get_overview_theory_markdown() -> str:
    """Return Overview page Physics Validation Equations as a single markdown string (for agent/export)."""
    return r"""# Overview — Physics Validation Equations

## Mach Number (LBM)

$$\text{Ma} = \frac{u_{\text{rms}}}{c_s}, \quad c_s = \frac{1}{\sqrt{3}}$$

where $u_{\text{rms}}$ is the root-mean-square velocity and $c_s$ is the lattice sound speed. For incompressible flow: $\text{Ma} < 0.1$

## Mach Number (Navier-Stokes)

$$\text{Ma} = \frac{u_{\text{rms}}}{c_{\text{sound}}}$$

where $c_{\text{sound}}$ is the physical speed of sound (from simulation.json). For incompressible flow: $\text{Ma} < 0.1$

---

## Knudsen Number — LBM (DNS/Continuum)

$$\text{Kn} = c_s \left(\tau_0 - \frac{1}{2}\right), \quad \tau_0 = \frac{\nu_0}{c_s^2} + \frac{1}{2}$$

where $\nu_0$ is the molecular viscosity and $c_s = 1/\sqrt{3}$. Continuum regime: $\text{Kn} < 0.01$

## Knudsen Number — LBM (LES/Turbulent)

$$\text{Kn}_t = \sqrt{3} \left(\tau_e - \frac{1}{2}\right)$$

where $\tau_e$ is the effective relaxation time from tau_analysis files.

## Knudsen Number — Navier-Stokes

$$\text{Kn} = \frac{\lambda}{L} \approx \frac{\nu}{c_{\text{sound}} \cdot L}$$

where $\lambda$ is the mean free path, $L$ is the characteristic length (from simulation.json), $\nu$ is viscosity, and $c_{\text{sound}}$ is the speed of sound. Continuum regime: $\text{Kn} < 0.01$

---

## Velocity Divergence (Compressibility Check)

$$\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z}$$

For incompressible flow, the divergence should be zero: $\nabla \cdot \mathbf{u} = 0$. The maximum absolute divergence $|\nabla \cdot \mathbf{u}|_{\max}$ is used to validate incompressibility.

## Compressibility Metrics

$$
\begin{align}
|\nabla \cdot \mathbf{u}|_{\max} &= \max_{x,y,z} |\nabla \cdot \mathbf{u}| \\
\text{RMS}(\nabla \cdot \mathbf{u}) &= \sqrt{\frac{1}{V} \int_V (\nabla \cdot \mathbf{u})^2 \, dV}
\end{align}
$$

Validation thresholds: $|\nabla \cdot \mathbf{u}|_{\max} < 10^{-5}$ (valid), $10^{-5} < |\nabla \cdot \mathbf{u}|_{\max} < 10^{-3}$ (warning), $|\nabla \cdot \mathbf{u}|_{\max} > 10^{-3}$ (invalid)
"""
