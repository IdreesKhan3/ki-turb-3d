"""
MRT Matrix Generator Utilities

Functions for computing MRT (Multiple Relaxation Time) transformation matrices
for D3Q19 lattice, matching the Fortran init_mrt_edm subroutine exactly.
"""

import numpy as np
import streamlit as st
import pandas as pd


def compute_mrt_matrix(dirx, diry, dirz):
    """
    Compute MRT transformation matrix M using Krüger-style D3Q19 orthogonalization.
    Matches Fortran init_mrt_edm subroutine exactly.
    
    Parameters:
    -----------
    dirx, diry, dirz : array-like, shape (19,)
        Lattice direction vectors for D3Q19
    
    Returns:
    --------
    M : ndarray, shape (19, 19)
        Transformation matrix M
    M_inv : ndarray, shape (19, 19)
        Inverse transformation matrix M^-1
    """
    dirx = np.array(dirx, dtype=np.float64)
    diry = np.array(diry, dtype=np.float64)
    dirz = np.array(dirz, dtype=np.float64)
    
    M = np.zeros((19, 19), dtype=np.float64)
    
    # Build M matrix (matching Fortran lines 1001-1024)
    for alpha in range(19):
        e_alpha = np.array([dirx[alpha], diry[alpha], dirz[alpha]], dtype=np.float64)
        norm_sq = np.dot(e_alpha, e_alpha)
        
        M[0, alpha] = 1.0
        M[1, alpha] = 19.0 * norm_sq - 30.0
        M[2, alpha] = 0.5 * (21.0 * norm_sq**2 - 53.0 * norm_sq + 24.0)
        M[3, alpha] = e_alpha[0]
        M[4, alpha] = (5.0 * norm_sq - 9.0) * e_alpha[0]
        M[5, alpha] = e_alpha[1]
        M[6, alpha] = (5.0 * norm_sq - 9.0) * e_alpha[1]
        M[7, alpha] = e_alpha[2]
        M[8, alpha] = (5.0 * norm_sq - 9.0) * e_alpha[2]
        M[9, alpha] = 3.0 * e_alpha[0]**2 - norm_sq
        M[10, alpha] = (3.0 * norm_sq - 5.0) * (3.0 * e_alpha[0]**2 - norm_sq)
        M[11, alpha] = e_alpha[1]**2 - e_alpha[2]**2
        M[12, alpha] = (3.0 * norm_sq - 5.0) * (e_alpha[1]**2 - e_alpha[2]**2)
        M[13, alpha] = e_alpha[0] * e_alpha[1]
        M[14, alpha] = e_alpha[1] * e_alpha[2]
        M[15, alpha] = e_alpha[0] * e_alpha[2]
        M[16, alpha] = (e_alpha[1]**2 - e_alpha[2]**2) * e_alpha[0]
        M[17, alpha] = (e_alpha[2]**2 - e_alpha[0]**2) * e_alpha[1]
        M[18, alpha] = (e_alpha[0]**2 - e_alpha[1]**2) * e_alpha[2]
    
    # Invert matrix
    try:
        M_inv = np.linalg.inv(M)
        # Check inversion quality
        identity_check = np.dot(M, M_inv)
        diff = np.sum(np.abs(identity_check - np.eye(19)))
        if diff > 1e-8:
            st.warning(f"⚠️ Matrix inversion error: {diff:.2e} (should be < 1e-8)")
    except np.linalg.LinAlgError:
        st.error("❌ Matrix is singular and cannot be inverted!")
        M_inv = None
    
    return M, M_inv


def compute_relaxation_vector(tau=None, nu=None, s_e=1.19, s_eps=1.4, s_q=1.2, s_pi=1.4, s_other=1.98):
    """
    Compute relaxation rate vector S (matching Fortran lines 979-998).
    
    Parameters:
    -----------
    tau : float, optional
        Relaxation time. If None, computed from nu
    nu : float, optional
        Kinematic viscosity. Used to compute tau if provided
    s_e, s_eps, s_q, s_pi, s_other : float
        Relaxation rates for different modes
    
    Returns:
    --------
    S : ndarray, shape (19,)
        Relaxation rate vector
    """
    if tau is None:
        if nu is not None:
            # From Fortran: nu = (1/3)(1/s_nu - 1/2) → s_nu = 1/(3*nu + 1/2)
            # But tau is used directly: s_nu = 1.0/tau
            # So: tau = 3*nu + 0.5
            tau = 3.0 * nu + 0.5
        else:
            tau = 1.0  # Default
    
    s_nu = 1.0 / tau
    
    # Matching Fortran Svec assignment (lines 980-998)
    S = np.array([
        1.0,      # 1: ρ (conserved)
        s_e,      # 2: e
        s_eps,    # 3: ε
        1.0,      # 4: jx (conserved)
        s_q,      # 5: qx
        1.0,      # 6: jy (conserved)
        s_q,      # 7: qy
        1.0,      # 8: jz (conserved)
        s_q,      # 9: qz
        s_nu,     # 10: 3pxx
        s_pi,     # 11: 3πxx
        s_nu,     # 12: pww
        s_pi,     # 13: πww
        s_nu,     # 14: pxy
        s_nu,     # 15: pyz
        s_nu,     # 16: pxz
        s_other,  # 17: mx
        s_other,  # 18: my
        s_other   # 19: mz
    ], dtype=np.float64)
    
    return S


def render_mrt_matrix_generator():
    """
    Render the MRT Matrix Generator tab UI.
    """
    st.header("🔧 MRT Matrix Generator")
    st.markdown("""
    Generate the MRT (Multiple Relaxation Time) transformation matrix **M** and its inverse **M⁻¹** 
    for D3Q19 lattice using the same equations as in the Fortran code (`init_mrt_edm` subroutine).
    """)
    
    st.markdown("---")
    
    # Default D3Q19 directions from OpenACC_parameters.F90
    default_dirx = [1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0]
    default_diry = [0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0]
    default_dirz = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Lattice Directions")
        st.caption("Enter D3Q19 velocity directions (19 components each)")
        
        use_defaults = st.checkbox("Use default directions (from OpenACC_parameters.F90)", value=True, key="mrt_use_defaults")
        
        if use_defaults:
            dirx = default_dirx
            diry = default_diry
            dirz = default_dirz
            st.info("Using default D3Q19 directions from Fortran parameters module")
        else:
            st.markdown("**X directions:**")
            dirx = []
            cols_x = st.columns(19)
            for i in range(19):
                with cols_x[i]:
                    dirx.append(st.number_input(f"", value=default_dirx[i], key=f"dirx_{i}", step=1, format="%d"))
            
            st.markdown("**Y directions:**")
            diry = []
            cols_y = st.columns(19)
            for i in range(19):
                with cols_y[i]:
                    diry.append(st.number_input(f"", value=default_diry[i], key=f"diry_{i}", step=1, format="%d"))
            
            st.markdown("**Z directions:**")
            dirz = []
            cols_z = st.columns(19)
            for i in range(19):
                with cols_z[i]:
                    dirz.append(st.number_input(f"", value=default_dirz[i], key=f"dirz_{i}", step=1, format="%d"))
    
    with col2:
        st.subheader("Relaxation Parameters")
        st.caption("Set relaxation rates (matching Fortran init_mrt_edm)")
        
        nu = st.number_input("Kinematic viscosity (ν)", value=0.002546479089469996, format="%.10f", key="mrt_nu")
        tau = st.number_input("Relaxation time (τ)", value=3.0 * 0.002546479089469996 + 0.5, format="%.10f", key="mrt_tau",
                              help="τ = 3ν + 0.5 (from Fortran: nu = (1/3)(1/s_ν - 1/2))")
        
        st.markdown("**Mode-specific relaxation rates:**")
        s_e = st.number_input("s_e (energy mode)", value=1.19, format="%.2f", key="mrt_se")
        s_eps = st.number_input("s_eps (higher-order energy)", value=1.4, format="%.2f", key="mrt_seps")
        s_q = st.number_input("s_q (energy flux)", value=1.2, format="%.2f", key="mrt_sq")
        s_pi = st.number_input("s_pi (non-hydro stress)", value=1.4, format="%.2f", key="mrt_spi")
        s_other = st.number_input("s_other (ghost modes)", value=1.98, format="%.2f", key="mrt_sother")
    
    st.markdown("---")
    
    # Compute matrices
    if st.button("🔧 Generate MRT Matrix", type="primary", use_container_width=True):
        with st.spinner("Computing MRT matrix..."):
            M, M_inv = compute_mrt_matrix(dirx, diry, dirz)
            S = compute_relaxation_vector(tau=tau, nu=nu, s_e=s_e, s_eps=s_eps, s_q=s_q, s_pi=s_pi, s_other=s_other)
            
            if M_inv is not None:
                st.session_state.mrt_M = M
                st.session_state.mrt_M_inv = M_inv
                st.session_state.mrt_S = S
                st.session_state.mrt_dirx = dirx
                st.session_state.mrt_diry = diry
                st.session_state.mrt_dirz = dirz
                st.success("✅ MRT matrix generated successfully!")
    
    # Display results
    if "mrt_M" in st.session_state:
        M = st.session_state.mrt_M
        M_inv = st.session_state.mrt_M_inv
        S = st.session_state.mrt_S
        
        st.markdown("---")
        st.subheader("Results")
        
        # Moment names (matching Fortran comments)
        moment_names = [
            "ρ (density)", "e (energy)", "ε (higher-order energy)",
            "jx (momentum x)", "qx (energy flux x)", "jy (momentum y)", "qy (energy flux y)",
            "jz (momentum z)", "qz (energy flux z)", "3pxx", "3πxx", "pww", "πww",
            "pxy", "pyz", "pxz", "mx", "my", "mz"
        ]
        
        tab_m, tab_minv, tab_s = st.tabs(["Matrix M", "Matrix M⁻¹", "Relaxation Vector S"])
        
        with tab_m:
            st.markdown("**Transformation Matrix M (19×19)**")
            st.caption("Transforms distribution functions f to moment space: **m = M f**")
            
            # Display as DataFrame for better formatting
            df_M = pd.DataFrame(M, 
                               index=[f"Row {i+1}: {moment_names[i]}" for i in range(19)],
                               columns=[f"α={i+1}" for i in range(19)])
            
            # Round for display
            st.dataframe(df_M.style.format("{:.2f}"), height=600, use_container_width=True)
            
            # Download button
            csv_M = df_M.to_csv()
            st.download_button(
                label="📥 Download M matrix (CSV)",
                data=csv_M,
                file_name="MRT_matrix_M.csv",
                mime="text/csv"
            )
        
        with tab_minv:
            st.markdown("**Inverse Transformation Matrix M⁻¹ (19×19)**")
            st.caption("Transforms moments back to distribution functions: **f = M⁻¹ m**")
            
            df_M_inv = pd.DataFrame(M_inv,
                                    index=[f"Row {i+1}: {moment_names[i]}" for i in range(19)],
                                    columns=[f"α={i+1}" for i in range(19)])
            
            st.dataframe(df_M_inv.style.format("{:.6f}"), height=600, use_container_width=True)
            
            csv_M_inv = df_M_inv.to_csv()
            st.download_button(
                label="📥 Download M⁻¹ matrix (CSV)",
                data=csv_M_inv,
                file_name="MRT_matrix_M_inv.csv",
                mime="text/csv"
            )
        
        with tab_s:
            st.markdown("**Relaxation Rate Vector S (19×1)**")
            st.caption("Relaxation rates for each moment mode")
            
            df_S = pd.DataFrame({
                "Moment": moment_names,
                "Relaxation Rate": S
            })
            
            st.dataframe(df_S, use_container_width=True, hide_index=True)
            
            csv_S = df_S.to_csv(index=False)
            st.download_button(
                label="📥 Download S vector (CSV)",
                data=csv_S,
                file_name="MRT_relaxation_vector_S.csv",
                mime="text/csv"
            )
        
        # Verification
        st.markdown("---")
        st.subheader("Verification")
        
        # Check M * M_inv = I
        identity_check = np.dot(M, M_inv)
        identity_error = np.max(np.abs(identity_check - np.eye(19)))
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric("Matrix Inversion Error", f"{identity_error:.2e}", 
                     help="Should be < 1e-8 for good inversion")
        with col_v2:
            if identity_error < 1e-8:
                st.success("✅ Matrix inversion is accurate")
            elif identity_error < 1e-6:
                st.warning("⚠️ Matrix inversion has moderate error")
            else:
                st.error("❌ Matrix inversion has large error")
        
        # Display direction summary
        st.markdown("**Lattice Directions Used:**")
        dir_df = pd.DataFrame({
            "α": range(1, 20),
            "cx": st.session_state.mrt_dirx,
            "cy": st.session_state.mrt_diry,
            "cz": st.session_state.mrt_dirz
        })
        st.dataframe(dir_df, use_container_width=True, hide_index=True)
