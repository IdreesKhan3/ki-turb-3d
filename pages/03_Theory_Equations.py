"""
Theory and Equations Page
D3Q19 lattice visualization, MRT matrix generator, all mathematical equations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from utils.export_figs import export_panel
from utils.mrt_matrix import render_mrt_matrix_generator
from visualizations.d3q19_lattice import plot_d3q19_lattice, DEFAULT_LATTICE_COLORS
st.set_page_config(page_icon="⚫")

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    st.title("📚 Theory & Equations")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📐 Governing Equations", "🔬 LBM Formulation", "📊 Analysis Equations", "⚛️ D3Q19 Lattice Visualization", "🔧 MRT Matrix Generator"])
    
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
            st.markdown(r"""
            Applying spatial filter $\overline{(\cdot)}$ to Navier-Stokes:
            """)
            st.latex(r"""
            \frac{\partial \bar{u}_i}{\partial t} + \frac{\partial}{\partial x_j}(\overline{u_i u_j}) = -\frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \nu \nabla^2 \bar{u}_i + \bar{f}_i
            """)
            st.markdown(r"""
            **Decomposition:** $\overline{u_i u_j} = \bar{u}_i \bar{u}_j + \tau_{ij}^{\mathrm{sgs}}$
            """)
            st.latex(r"""
            \frac{\partial \bar{u}_i}{\partial t} + \frac{\partial}{\partial x_j} \left( \bar{u}_i \bar{u}_j + \tau_{ij}^{\mathrm{sgs}} \right) = - \frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \nu \, \nabla^2 \bar{u}_i + \bar{f}_i
            """)
            st.markdown(r"""
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
            st.markdown(r"""
            where $\bar{S}_{ij} = \tfrac{1}{2}(\partial_i \bar{u}_j + \partial_j \bar{u}_i)$ and $|\bar{S}| = (2\,\bar{S}_{ij}\bar{S}_{ij})^{1/2}$
            """)
    
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
            st.markdown(r"""
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
            st.markdown(r"""
            where $\mathbf{\Lambda}(\nu_e)$ uses effective viscosity $\nu_e = \nu_0 + \nu_t$
            """)
            
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
            st.markdown(r"""
            where $m_i^{(neq)}$ are non-equilibrium moments and $s_i$ are relaxation parameters
            """)
            
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
            st.markdown(r"""
            where $f_L$ and $f_\eta$ are large-scale and dissipation-range corrections
            """)
        
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
            st.markdown(r"""
            Gaussian reference: $F = 3$
            """)
        
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
            st.markdown(r"""
            where $c_s = 1/\sqrt{3}$ is the lattice sound speed. For incompressible flow: $\text{Ma} < 0.1$
            """)
            
            st.markdown("**Knudsen Number (DNS/Continuum Check):**")
            st.latex(r"""
            \text{Kn} = \frac{c_s (\tau_0 - 1/2) \Delta x}{\Delta x} = c_s \left(\tau_0 - \frac{1}{2}\right)
            """)
            st.markdown(r"""
            where $\tau_0 = \nu_0/c_s^2 + 1/2$ is the molecular relaxation time. Continuum regime: $\text{Kn} < 0.01$
            """)
            
            st.markdown("**Knudsen Number (LES):**")
            st.latex(r"""
            \text{Kn}_t = \frac{(\tau_e - 1/2) \sqrt{3} \Delta x}{\Delta x} = \sqrt{3} \left(\tau_e - \frac{1}{2}\right)
            """)
            st.markdown(r"""
            where $\tau_e = 3(\nu_0 + \nu_t) + 1/2$ is the effective relaxation time. Continuum regime: $\text{Kn}_t < 0.01$
            """)
    
    # D3Q19 Lattice Visualization Tab
    with tab4:
        st.header("⚛️ D3Q19 Lattice Stencil Visualization")
        st.markdown("Interactive 3D visualization of the D3Q19 lattice stencil with full customization controls.")
        
        # Initialize session state for D3Q19 settings
        if 'd3q19_settings' not in st.session_state:
            st.session_state.d3q19_settings = _default_d3q19_settings()
        
        # Load saved settings if available
        settings_file = project_root / "d3q19_settings.json"
        # Valid symbols for Plotly Scatter3d
        valid_symbols = ['circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x']
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    # Merge with defaults
                    for key, value in saved_settings.items():
                        if key in st.session_state.d3q19_settings:
                            # Validate symbol values
                            if key in ['node_style', 'origin_style'] and value not in valid_symbols:
                                # Convert invalid symbol to default
                                st.session_state.d3q19_settings[key] = 'circle'
                            else:
                                st.session_state.d3q19_settings[key] = value
            except Exception:
                pass
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎨 D3Q19 Visualization Controls")
            
            # Stencil Configuration
            with st.expander("📐 Stencil Configuration", expanded=True):
                st.session_state.d3q19_settings['show_vectors'] = st.checkbox(
                    "Show Vectors", 
                    value=st.session_state.d3q19_settings.get('show_vectors', True),
                    key="d3q19_show_vectors"
                )
                st.session_state.d3q19_settings['vector_scale'] = st.slider(
                    "Vector Length Scale", 
                    0.1, 2.0, 
                    value=st.session_state.d3q19_settings.get('vector_scale', 1.0),
                    step=0.1,
                    key="d3q19_vector_scale"
                )
                st.session_state.d3q19_settings['vector_width'] = st.slider(
                    "Vector Width", 
                    1.0, 10.0, 
                    value=st.session_state.d3q19_settings.get('vector_width', 3.0),
                    step=0.5,
                    key="d3q19_vector_width"
                )
            
            # Node Appearance
            with st.expander("🔵 Node Appearance", expanded=False):
                node_style_options = ['circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x']
                current_node_style = st.session_state.d3q19_settings.get('node_style', 'circle')
                st.session_state.d3q19_settings['node_style'] = st.selectbox(
                    "Node Style",
                    node_style_options,
                    index=node_style_options.index(current_node_style) if current_node_style in node_style_options else 0,
                    key="d3q19_node_style"
                )
                st.session_state.d3q19_settings['node_size'] = st.slider(
                    "Node Size", 
                    5.0, 50.0, 
                    value=st.session_state.d3q19_settings.get('node_size', 10.0),
                    step=1.0,
                    key="d3q19_node_size"
                )
                st.session_state.d3q19_settings['node_opacity'] = st.slider(
                    "Node Opacity", 
                    0.0, 1.0, 
                    value=st.session_state.d3q19_settings.get('node_opacity', 0.8),
                    step=0.1,
                    key="d3q19_node_opacity"
                )
                st.session_state.d3q19_settings['node_edge_color'] = st.color_picker(
                    "Node Edge Color",
                    value=st.session_state.d3q19_settings.get('node_edge_color', '#000000'),
                    key="d3q19_node_edge_color"
                )
                st.session_state.d3q19_settings['node_edge_width'] = st.slider(
                    "Node Edge Width", 
                    0.0, 5.0, 
                    value=st.session_state.d3q19_settings.get('node_edge_width', 1.0),
                    step=0.1,
                    key="d3q19_node_edge_width"
                )
                st.divider()
                st.markdown("**Origin Marker**")
                origin_style_options = ['circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x']
                current_origin_style = st.session_state.d3q19_settings.get('origin_style', 'circle-open')
                st.session_state.d3q19_settings['origin_style'] = st.selectbox(
                    "Origin Style",
                    origin_style_options,
                    index=origin_style_options.index(current_origin_style) if current_origin_style in origin_style_options else 0,
                    key="d3q19_origin_style"
                )
                st.session_state.d3q19_settings['origin_size'] = st.slider(
                    "Origin Marker Size", 
                    5.0, 50.0, 
                    value=st.session_state.d3q19_settings.get('origin_size', 15.0),
                    step=1.0,
                    key="d3q19_origin_size"
                )
                st.session_state.d3q19_settings['origin_color'] = st.color_picker(
                    "Origin Color",
                    value=st.session_state.d3q19_settings.get('origin_color', '#052020'),
                    key="d3q19_origin_color"
                )
            
            # Vector Styling
            with st.expander("➡️ Vector Styling", expanded=False):
                st.session_state.d3q19_settings['vector_color'] = st.color_picker(
                    "Vector Color",
                    value=st.session_state.d3q19_settings.get('vector_color', '#FF0000'),
                    key="d3q19_vector_color"
                )
                st.session_state.d3q19_settings['vector_opacity'] = st.slider(
                    "Vector Opacity", 
                    0.0, 1.0, 
                    value=st.session_state.d3q19_settings.get('vector_opacity', 0.8),
                    step=0.1,
                    key="d3q19_vector_opacity"
                )
                st.session_state.d3q19_settings['vector_linestyle'] = st.selectbox(
                    "Vector Line Style",
                    ['solid', 'dash', 'dot', 'dashdot'],
                    index=['solid', 'dash', 'dot', 'dashdot'].index(
                        st.session_state.d3q19_settings.get('vector_linestyle', 'dashdot')
                    ),
                    key="d3q19_vector_linestyle"
                )
            
            # Labels
            with st.expander("🏷️ Labels", expanded=False):
                st.session_state.d3q19_settings['show_labels'] = st.checkbox(
                    "Show Labels", 
                    value=st.session_state.d3q19_settings.get('show_labels', True),
                    key="d3q19_show_labels"
                )
                st.session_state.d3q19_settings['label_prefix'] = st.text_input(
                    "Label Prefix (e.g., 'C' for C1, C2, ...)",
                    value=st.session_state.d3q19_settings.get('label_prefix', 'C'),
                    key="d3q19_label_prefix"
                )
                st.session_state.d3q19_settings['label_font_size'] = st.slider(
                    "Label Font Size", 
                    8, 24, 
                    value=st.session_state.d3q19_settings.get('label_font_size', 13),
                    step=1,
                    key="d3q19_label_font_size"
                )
                st.session_state.d3q19_settings['label_color'] = st.color_picker(
                    "Label Color",
                    value=st.session_state.d3q19_settings.get('label_color', '#000000'),
                    key="d3q19_label_color"
                )
            
            # Faces and Edges
            with st.expander("🎨 Faces & Edges", expanded=False):
                st.session_state.d3q19_settings['show_faces'] = st.checkbox(
                    "Show Colored Faces", 
                    value=st.session_state.d3q19_settings.get('show_faces', True),
                    key="d3q19_show_faces"
                )
                st.session_state.d3q19_settings['face_opacity'] = st.slider(
                    "Face Opacity", 
                    0.0, 1.0, 
                    value=st.session_state.d3q19_settings.get('face_opacity', 0.5),
                    step=0.1,
                    key="d3q19_face_opacity"
                )
                st.session_state.d3q19_settings['show_cube_edges'] = st.checkbox(
                    "Show Cube Edges", 
                    value=st.session_state.d3q19_settings.get('show_cube_edges', True),
                    key="d3q19_show_cube_edges"
                )
                st.session_state.d3q19_settings['cube_edge_color'] = st.color_picker(
                    "Cube Edge Color",
                    value=st.session_state.d3q19_settings.get('cube_edge_color', '#000000'),
                    key="d3q19_cube_edge_color"
                )
                st.session_state.d3q19_settings['cube_edge_width'] = st.slider(
                    "Cube Edge Width", 
                    0.5, 5.0, 
                    value=st.session_state.d3q19_settings.get('cube_edge_width', 1.0),
                    step=0.5,
                    key="d3q19_cube_edge_width"
                )
            
            # View Controls
            with st.expander("👁️ View Controls", expanded=False):
                # View presets - place BEFORE sliders so button updates take precedence
                st.markdown("**Quick View Presets:**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Front View", key="d3q19_front", use_container_width=True):
                        st.session_state.d3q19_settings['camera_elevation'] = 0.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 0.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                    if st.button("Side View", key="d3q19_side", use_container_width=True):
                        st.session_state.d3q19_settings['camera_elevation'] = 0.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 90.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                with col2:
                    if st.button("Top View", key="d3q19_top", use_container_width=True):
                        st.session_state.d3q19_settings['camera_elevation'] = 90.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 0.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                    if st.button("Isometric", key="d3q19_iso", use_container_width=True):
                        st.session_state.d3q19_settings['camera_elevation'] = 35.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 45.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                
                st.markdown("---")
                st.markdown("**Manual Camera Controls:**")
                st.session_state.d3q19_settings['camera_elevation'] = st.slider(
                    "Camera Elevation (degrees)", 
                    -90.0, 90.0, 
                    value=st.session_state.d3q19_settings.get('camera_elevation', 9.0),
                    step=1.0,
                    key="d3q19_camera_elevation"
                )
                st.session_state.d3q19_settings['camera_azimuth'] = st.slider(
                    "Camera Azimuth (degrees)", 
                    -180.0, 180.0, 
                    value=st.session_state.d3q19_settings.get('camera_azimuth', 16.0),
                    step=1.0,
                    key="d3q19_camera_azimuth"
                )
                st.session_state.d3q19_settings['camera_zoom'] = st.slider(
                    "Camera Zoom", 
                    0.5, 3.0, 
                    value=st.session_state.d3q19_settings.get('camera_zoom', 1.0),
                    step=0.1,
                    key="d3q19_camera_zoom"
                )
            
            # Advanced Options
            with st.expander("⚙️ Advanced Options", expanded=False):
                st.session_state.d3q19_settings['show_axes'] = st.checkbox(
                    "Show Coordinate Axes", 
                    value=st.session_state.d3q19_settings.get('show_axes', False),
                    key="d3q19_show_axes"
                )
                st.session_state.d3q19_settings['show_axis_labels'] = st.checkbox(
                    "Show Axis Labels", 
                    value=st.session_state.d3q19_settings.get('show_axis_labels', False),
                    key="d3q19_show_axis_labels"
                )
                st.session_state.d3q19_settings['show_origin_marker'] = st.checkbox(
                    "Show Origin Marker", 
                    value=st.session_state.d3q19_settings.get('show_origin_marker', True),
                    key="d3q19_show_origin_marker"
                )
                st.session_state.d3q19_settings['show_grid'] = st.checkbox(
                    "Show Grid", 
                    value=st.session_state.d3q19_settings.get('show_grid', False),
                    key="d3q19_show_grid"
                )
                st.session_state.d3q19_settings['background_color'] = st.color_picker(
                    "Background Color",
                    value=st.session_state.d3q19_settings.get('background_color', '#FFFFFF'),
                    key="d3q19_background_color"
                )
            
            # Save/Reset buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Settings", key="d3q19_save"):
                    try:
                        with open(settings_file, 'w') as f:
                            json.dump(st.session_state.d3q19_settings, f, indent=2)
                        st.success("Settings saved!")
                    except Exception as e:
                        st.error(f"Could not save settings: {e}")
            with col2:
                if st.button("♻️ Reset to Defaults", key="d3q19_reset"):
                    st.session_state.d3q19_settings = _default_d3q19_settings()
                    st.rerun()
        
        # Generate and display visualization
        fig = plot_d3q19_lattice(**st.session_state.d3q19_settings)
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'd3q19_lattice',
                    'height': 800,
                    'width': 800,
                    'scale': 2
                }
            }
        )
        
        # Export options - using comprehensive export panel like other pages
        export_panel(fig, project_root, "d3q19_lattice")
    
    # MRT Matrix Generator Tab
    with tab5:
        render_mrt_matrix_generator()
    


def _default_d3q19_settings():
    """Return default settings for D3Q19 visualization"""
    return {
        'show_vectors': True,
        'vector_scale': 1.0,
        'vector_width': 3.0,
        'node_size': 10.0,
        'node_colors': DEFAULT_LATTICE_COLORS.copy(),
        'node_opacity': 0.8,
        'node_style': 'circle',
        'node_edge_color': '#000000',
        'node_edge_width': 1.0,
        'origin_size': 15.0,
        'origin_color': '#052020',
        'origin_style': 'circle-open',
        'vector_color': '#FF0000',
        'vector_opacity': 0.8,
        'vector_linestyle': 'dashdot',
        'show_vector_arrows': False,
        'arrow_head_size': 0.1,
        'show_labels': True,
        'label_prefix': 'C',
        'label_font_size': 13,
        'label_color': '#000000',
        'label_offset': 1.19,
        'show_faces': False,
        'face_opacity': 0.5,
        'show_cube_edges': True,
        'cube_edge_color': '#000000',
        'cube_edge_width': 2.0,
        'cube_edge_style': 'solid',
        'show_grid': False,
        'grid_color': '#808080',
        'grid_opacity': 0.3,
        'background_color': '#FFFFFF',
        'show_axes': False,
        'show_axis_labels': False,
        'show_origin_marker': True,
        'camera_elevation': 9.0,
        'camera_azimuth': 16.0,
        'camera_zoom': 1.0,
        'width': 800,
        'height': 800,
        'title': 'D3Q19 Lattice Stencil'
    }

if __name__ == "__main__":
    main()

