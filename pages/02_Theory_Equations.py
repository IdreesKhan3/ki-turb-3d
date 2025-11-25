"""
Theory and Equations Page
D3Q19 lattice visualization, MRT matrix generator, all mathematical equations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    st.title("📚 Theory & Equations")
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📐 Governing Equations", "🔬 LBM Formulation", "📊 Analysis Equations"])
    
    with tab1:
        st.header("From Navier-Stokes to LBM")
        
        # Step 1: Navier-Stokes
        with st.expander("**1. Navier-Stokes Equations**", expanded=True):
            st.markdown("""
            **Incompressible flow equations:**
            """)
            st.latex(r"""
            \begin{align}
            \nabla \cdot \mathbf{u} &= 0 \\
            \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} &= -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f}
            \end{align}
            """)
            st.caption("Continuity and momentum conservation equations")
        
        # Step 2: Filtered Navier-Stokes (LES)
        with st.expander("**2. Filtered Navier-Stokes (LES)**", expanded=False):
            st.markdown("""
            Applying spatial filter $\overline{(\cdot)}$ to Navier-Stokes:
            """)
            st.latex(r"""
            \frac{\partial \bar{u}_i}{\partial t} + \frac{\partial}{\partial x_j}(\overline{u_i u_j}) = -\frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \nu \nabla^2 \bar{u}_i + \bar{f}_i
            """)
            st.markdown("""
            **Decomposition:** $\overline{u_i u_j} = \bar{u}_i \bar{u}_j + \tau_{ij}^{\mathrm{sgs}}$
            """)
            st.latex(r"""
            \frac{\partial \bar{u}_i}{\partial t} + \frac{\partial}{\partial x_j} \left( \bar{u}_i \bar{u}_j + \tau_{ij}^{\mathrm{sgs}} \right) = - \frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \nu \, \nabla^2 \bar{u}_i + \bar{f}_i
            """)
            st.markdown("""
            where $\tau_{ij}^{\mathrm{sgs}} = \overline{u_i u_j} - \bar{u}_i \bar{u}_j$ is the **subgrid-scale stress tensor** requiring closure.
            """)
            
            st.markdown("---")
            st.markdown("**Eddy-viscosity closure (Smagorinsky):**")
            st.latex(r"""
            \begin{align}
            \tau_{ij}^{\mathrm{sgs}} - \tfrac{1}{3}\tau_{kk}^{\mathrm{sgs}} \, \delta_{ij} &= -2 \, \nu_t \, \bar{S}_{ij} \\
            \nu_t &= (C_s \, \Delta)^2 \, |\bar{S}|
            \end{align}
            """)
            st.caption("where $\bar{S}_{ij} = \tfrac{1}{2}(\partial_i \bar{u}_j + \partial_j \bar{u}_i)$ and $|\bar{S}| = (2\,\bar{S}_{ij}\bar{S}_{ij})^{1/2}$")
    
    with tab2:
        st.header("Lattice Boltzmann Method")
        st.info("**Primary focus:** MRT (Multiple Relaxation Time) | **Reference:** BGK/SRT (shown for app flexibility)")
        
        # MRT DNS - PRIMARY, EXPANDED
        with st.expander("**MRT - DNS Formulation** (Primary)", expanded=True):
            st.markdown("**MRT-LBM evolution equation (D3Q19):**")
            st.latex(r"""
            f_\alpha(\mathbf{x} + \mathbf{c}_\alpha \delta t, t + \delta t) - f_\alpha(\mathbf{x}, t) = 
            - \left[ \mathbf{M}^{-1} \mathbf{\Lambda} \mathbf{M} (\mathbf{f} - \mathbf{f}^{eq}) \right]_\alpha 
            + \delta t \left[ \mathbf{M}^{-1} \left( \mathbf{I} - \frac{\mathbf{\Lambda}}{2} \right) \mathbf{M} \mathbf{\Phi} \right]_\alpha
            """)
            st.caption("Multiple relaxation times provide better stability and accuracy")
            
            st.markdown("**Transformation matrix $\mathbf{M}$ (D3Q19):**")
            st.markdown("""
            The 19×19 transformation matrix $\mathbf{M}$ is constructed using the orthogonal moment basis (d'Humières et al. 2002).
            The matrix transforms distribution functions $f_\alpha$ to moment space: $\mathbf{m} = \mathbf{M} \mathbf{f}$.
            """)
            st.info("💡 *Full 19×19 matrix display available in MRT Matrix Generator tool (coming soon)*")
            
            st.markdown("**Relaxation matrix $\mathbf{\Lambda}$ (diagonal):**")
            st.latex(r"""
            \mathbf{\Lambda} = \text{diag}(1.0, 1.19, 1.4, 1.0, 1.2, 1.0, 1.2, 1.0, 1.2, s_\nu, 1.4, s_\nu, 1.4, s_\nu, s_\nu, s_\nu, 1.98, 1.98, 1.98)
            """)
            st.caption("where $s_\nu$ is the viscosity-related relaxation parameter")
            
            st.markdown("**Equilibrium moments $\mathbf{m}^{(eq)}$:**")
            st.latex(r"""
            \mathbf{m}^{(eq)} = \begin{bmatrix}
            \delta\rho = \rho - \rho_0 \\
            -11\delta\rho + 19\rho(u_x^2 + u_y^2 + u_z^2) \\
            -\frac{475}{63}\rho(u_x^2 + u_y^2 + u_z^2) \\
            \rho u_x, \quad -\frac{2\rho u_x}{3} \\
            \rho u_y, \quad -\frac{2\rho u_y}{3} \\
            \rho u_z, \quad -\frac{2\rho u_z}{3} \\
            \rho(2u_x^2 - u_y^2 - u_z^2), \quad 0 \\
            \rho(u_y^2 - u_z^2), \quad 0 \\
            \rho u_x u_y, \quad \rho u_y u_z, \quad \rho u_x u_z \\
            0, \quad 0, \quad 0
            \end{bmatrix}
            """)
            
            st.markdown("**Force moments $\mathbf{F}_m$:**")
            st.latex(r"""
            \mathbf{F}_m = \begin{bmatrix}
            0 \\
            38(u_x F_x + u_y F_y + u_z F_z) \\
            -11(u_x F_x + u_y F_y + u_z F_z) \\
            F_x, \quad -\frac{2F_x}{3} \\
            F_y, \quad -\frac{2F_y}{3} \\
            F_z, \quad -\frac{2F_z}{3} \\
            2(2u_x F_x - u_y F_y - u_z F_z), \quad -(2u_x F_x - u_y F_y - u_z F_z) \\
            2(u_y F_y - u_z F_z), \quad -(u_y F_y - u_z F_z) \\
            u_x F_y + u_y F_x, \quad u_y F_z + u_z F_y, \quad u_x F_z + u_z F_x \\
            0, \quad 0, \quad 0
            \end{bmatrix}
            """)
            
            st.markdown("**Equilibrium distribution:**")
            st.latex(r"""
            f_\alpha^{eq} = w_\alpha \rho \left[ 1 + \frac{\mathbf{c}_\alpha \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_\alpha \cdot \mathbf{u})^2}{2c_s^4} - \frac{\mathbf{u} \cdot \mathbf{u}}{2c_s^2} \right]
            """)
            
            st.markdown("**Guo's forcing term:**")
            st.latex(r"""
            \Phi_\alpha = w_\alpha \left[ \frac{\mathbf{c}_\alpha - \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_\alpha \cdot \mathbf{u})\mathbf{c}_\alpha}{c_s^4} \right] \cdot \mathbf{F}^{\text{ext}}
            """)
            
            st.markdown("**Macroscopic quantities:**")
            st.latex(r"""
            \rho = \sum_\alpha f_\alpha, \quad \rho \mathbf{u} = \sum_\alpha f_\alpha \mathbf{c}_\alpha
            """)
        
        # MRT LES - PRIMARY, EXPANDED
        with st.expander("**MRT - LES Formulation** (Primary)", expanded=True):
            st.markdown("**Effective viscosity approach:**")
            st.latex(r"""
            \begin{align}
            \nu_e &= \nu_0 + \nu_t \\
            \frac{1}{s_\nu} &= \frac{1}{2} + 3(\nu_0 + \nu_t) \equiv \tau_e
            \end{align}
            """)
            
            st.markdown("**LES-MRT evolution:**")
            st.latex(r"""
            f_\alpha(\mathbf{x} + \mathbf{c}_\alpha \delta t, t + \delta t) - f_\alpha(\mathbf{x}, t) = 
            - \left[ \mathbf{M}^{-1} \mathbf{\Lambda}(\nu_e) \mathbf{M} (\mathbf{f} - \mathbf{f}^{eq}) \right]_\alpha 
            + \delta t \left[ \mathbf{M}^{-1} \left( \mathbf{I} - \frac{\mathbf{\Lambda}(\nu_e)}{2} \right) \mathbf{M} \mathbf{\Phi} \right]_\alpha
            """)
            st.caption("where $\mathbf{\Lambda}(\nu_e)$ uses effective viscosity $\nu_e = \nu_0 + \nu_t$")
            
            st.markdown("**Strain rate tensor from non-equilibrium moments:**")
            st.markdown("""
            The components of the filtered strain-rate tensor are computed from non-equilibrium moments:
            """)
            st.latex(r"""
            \begin{align}
            S_{xx} &= -\frac{s_1 m_1^{(neq)} + 19s_9 m_9^{(neq)}}{38\rho_0\delta_t} \\
            S_{yy} &= -\frac{2s_1 m_1^{(neq)} - 19s_9(m_9^{(neq)} - 3m_{11}^{(neq)})}{76\rho_0\delta_t} \\
            S_{zz} &= -\frac{2s_1 m_1^{(neq)} - 19s_9(m_9^{(neq)} + 3m_{11}^{(neq)})}{76\rho_0\delta_t} \\
            S_{xy} &= -\frac{3s_9}{2\rho_0\delta_t} m_{13}^{(neq)}, \quad
            S_{xz} = -\frac{3s_9}{2\rho_0\delta_t} m_{15}^{(neq)}, \quad
            S_{yz} = -\frac{3s_9}{2\rho_0\delta_t} m_{14}^{(neq)}
            \end{align}
            """)
            st.caption("where $m_i^{(neq)}$ are non-equilibrium moments and $s_i$ are relaxation parameters")
            
            st.markdown("**Alternative form (from non-equilibrium stress tensor):**")
            st.latex(r"""
            S_{ab} = \frac{P_{ab}^{\text{ne}}}{\rho c_s^2 \tau_e}, \quad P_{ab}^{\text{ne}} = \sum_\alpha f_\alpha^{\text{ne}} c_{\alpha a} c_{\alpha b}
            """)
        
        st.markdown("---")
        st.markdown("### Reference: BGK/SRT (for app flexibility demonstration)")
        
        # BGK (SRT) DNS - REFERENCE ONLY, COMPACT
        with st.expander("**BGK (SRT) - DNS** (Reference)", expanded=False):
            st.markdown("*Shown for reference - app can analyze BGK/SRT data, but MRT is primary*")
            st.latex(r"""
            f_\alpha(\mathbf{x} + \mathbf{c}_\alpha \delta t, t + \delta t) - f_\alpha(\mathbf{x}, t) = 
            -\frac{1}{\tau} \left(f_\alpha(\mathbf{x}, t) - f_\alpha^{eq}(\mathbf{x}, t)\right) + \mathbf{F}_\alpha^{\text{ext}}
            """)
            st.latex(r"""
            \nu = c_s^2 \left(\tau - \frac{1}{2}\right) \delta x, \quad \mathbf{F}_\alpha^{\text{ext}} = f_\alpha^{eq,\text{shift}} - f_\alpha^{eq}
            """)
        
        # BGK LES - REFERENCE ONLY, COMPACT
        with st.expander("**BGK (SRT) - LES** (Reference)", expanded=False):
            st.markdown("*Shown for reference - app can analyze BGK/SRT data, but MRT is primary*")
            st.latex(r"""
            \bar{f}_\alpha(\mathbf{x} + \mathbf{c}_\alpha \delta t, t + \delta t) - \bar{f}_\alpha(\mathbf{x}, t) = 
            -\frac{1}{\tau_e} \left(\bar{f}_\alpha(\mathbf{x}, t) - \bar{f}_\alpha^{eq}(\mathbf{x}, t)\right) + 3\rho w_\alpha (\mathbf{c}_\alpha \cdot \bar{\mathbf{F}})
            """)
            st.latex(r"""
            \tau_e = 3(\nu_0 + C \Delta^2 |\bar{S}_{ab}|) + \frac{1}{2}
            """)
        
        # Continuous Boltzmann - REFERENCE
        with st.expander("**Continuous Boltzmann Equation** (Reference)", expanded=False):
            st.latex(r"""
            \frac{\partial f}{\partial t} + \xi_\alpha \frac{\partial f}{\partial x_\alpha} + \mathbf{F}_\alpha \frac{\partial f}{\partial \xi_\alpha} = \Omega(f)
            """)
            st.caption("Foundation: continuous kinetic equation")
    
    with tab3:
        st.header("Turbulence Analysis Equations")
        
        # Energy Spectrum
        with st.expander("**Energy Spectrum**", expanded=False):
            st.latex(r"""
            E(k) = \frac{1}{2} \left( |\hat{u}(k)|^2 + |\hat{v}(k)|^2 + |\hat{w}(k)|^2 \right)
            """)
            st.markdown("**Pope model spectrum:**")
            st.latex(r"""
            E_{\text{pope}}(k) = C \varepsilon^{2/3} k^{-5/3} f_L(kL) f_\eta(k\eta)
            """)
            st.caption("where $f_L$ and $f_\eta$ are large-scale and dissipation-range corrections")
        
        # Structure Functions
        with st.expander("**Structure Functions**", expanded=False):
            st.latex(r"""
            S_p(r) = \frac{1}{3} \left\langle |\delta u_x|^p + |\delta u_y|^p + |\delta u_z|^p \right\rangle
            """)
            st.markdown("**Extended Self-Similarity (ESS):**")
            st.latex(r"""
            S_p(r) \propto S_3(r)^{\zeta_p}
            """)
        
        # Flatness
        with st.expander("**Flatness Factor**", expanded=False):
            st.latex(r"""
            F(r) = \frac{1}{3} \left[ \frac{\langle (\delta u_x)^4 \rangle}{\langle (\delta u_x)^2 \rangle^2} + 
            \frac{\langle (\delta u_y)^4 \rangle}{\langle (\delta u_y)^2 \rangle^2} + 
            \frac{\langle (\delta u_z)^4 \rangle}{\langle (\delta u_z)^2 \rangle^2} \right]
            """)
            st.caption("Gaussian reference: $F = 3$")
        
        # Isotropy
        with st.expander("**Isotropy Coefficient**", expanded=False):
            st.latex(r"""
            \text{IC}(k) = \frac{E_{22}(k)}{E_{11}(k)} = \frac{|\hat{v}(k)|^2}{|\hat{u}(k)|^2}
            """)
            st.caption("Isotropic flow: $\text{IC}(k) = 1$")
        
        # Physics Validation
        with st.expander("**Physics Validation Parameters**", expanded=False):
            st.markdown("**Mach Number:**")
            st.latex(r"""
            \text{Ma} = \frac{u_{\text{rms}}}{c_s}
            """)
            st.caption("where $c_s = 1/\sqrt{3}$ is the lattice sound speed. For incompressible flow: $\text{Ma} < 0.1$")
            
            st.markdown("**Knudsen Number (DNS/Continuum Check):**")
            st.latex(r"""
            \text{Kn} = \frac{c_s (\tau_0 - 1/2) \Delta x}{\Delta x} = c_s \left(\tau_0 - \frac{1}{2}\right)
            """)
            st.caption("where $\tau_0 = \nu_0/c_s^2 + 1/2$ is the molecular relaxation time. Continuum regime: $\text{Kn} < 0.01$")
            
            st.markdown("**Knudsen Number (LES):**")
            st.latex(r"""
            \text{Kn}_t = \frac{(\tau_e - 1/2) \sqrt{3} \Delta x}{\Delta x} = \sqrt{3} \left(\tau_e - \frac{1}{2}\right)
            """)
            st.caption("where $\tau_e = 3(\nu_0 + \nu_t) + 1/2$ is the effective relaxation time. Continuum regime: $\text{Kn}_t < 0.01$")
    
    # Additional sections (placeholders for now)
    st.markdown("---")
    st.header("Additional Tools")
    
    with st.expander("**D3Q19 Lattice Visualization**", expanded=False):
        st.info("D3Q19 lattice visualization - Implementation in progress")
    
    with st.expander("**MRT Matrix Generator**", expanded=False):
        st.info("MRT matrix generator - Implementation in progress")

if __name__ == "__main__":
    main()

