"""
MRT Matrix Generator Utilities

Functions for computing MRT (Multiple Relaxation Time) transformation matrices
for D3Q19 lattice, matching the Fortran init_mrt_edm subroutine exactly.
"""

import numpy as np
import streamlit as st
import pandas as pd


def validate_d3q19_directions(dirx, diry, dirz):
    """
    Validate D3Q19 direction vectors.
    
    Parameters:
    -----------
    dirx, diry, dirz : array-like
        Lattice direction vectors
    
    Returns:
    --------
    is_valid : bool
        True if directions are valid
    errors : list
        List of error messages (empty if valid)
    warnings : list
        List of warning messages
    """
    errors = []
    warnings = []
    
    # Check length
    if len(dirx) != 19 or len(diry) != 19 or len(dirz) != 19:
        errors.append(f"Direction vectors must each contain exactly 19 entries. Got lengths: {len(dirx)}, {len(diry)}, {len(dirz)}")
        return False, errors, warnings
    
    # Check for valid D3Q19 format (each direction should be one of: (0,0,0), (±1,0,0), (0,±1,0), (0,0,±1), (±1,±1,0), (±1,0,±1), (0,±1,±1))
    # Valid speeds: 0, 1, sqrt(2)
    valid_directions = set()
    invalid_directions = []
    duplicate_directions = []
    
    for i in range(19):
        cx, cy, cz = dirx[i], diry[i], dirz[i]
        speed_sq = cx*cx + cy*cy + cz*cz
        
        # Check if speed is valid for D3Q19 (0, 1, or 2)
        if speed_sq not in [0, 1, 2]:
            invalid_directions.append(f"a={i+1}: ({cx}, {cy}, {cz}) has invalid speed²={speed_sq:.2f} (must be 0, 1, or 2)")
        
        # Check for duplicates
        direction_tuple = (int(cx), int(cy), int(cz))
        if direction_tuple in valid_directions:
            duplicate_directions.append(f"a={i+1}: ({cx}, {cy}, {cz}) is a duplicate")
        else:
            valid_directions.add(direction_tuple)
    
    if invalid_directions:
        errors.extend(invalid_directions)
    
    if duplicate_directions:
        warnings.extend(duplicate_directions)
        warnings.append("Duplicate directions may cause matrix singularity.")
    
    # Check if we have exactly 19 unique directions (required for valid basis)
    if len(valid_directions) < 19:
        errors.append(f"Found only {len(valid_directions)} unique directions. D3Q19 requires exactly 19 unique directions.")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


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
    M_inv : ndarray, shape (19, 19) or None
        Inverse transformation matrix M^-1 (None if singular)
    identity_error_max : float
        Maximum absolute error in M * M_inv - I (inf if singular)
    """
    dirx = np.array(dirx, dtype=np.float64)
    diry = np.array(diry, dtype=np.float64)
    dirz = np.array(dirz, dtype=np.float64)
    
    M = np.zeros((19, 19), dtype=np.float64)
    
    # Build M matrix (matching Fortran lines 1001-1024)
    for alpha in range(19):
        e_alpha = np.array([dirx[alpha], diry[alpha], dirz[alpha]], dtype=np.float64)
        norm_sq = np.dot(e_alpha, e_alpha)
        
        M[ 0, alpha] = 1.0
        M[ 1, alpha] = 19.0 * norm_sq - 30.0
        M[ 2, alpha] = 0.5 * (21.0 * norm_sq**2 - 53.0 * norm_sq + 24.0)
        M[ 3, alpha] = e_alpha[0]
        M[ 4, alpha] = (5.0 * norm_sq - 9.0) * e_alpha[0]
        M[ 5, alpha] = e_alpha[1]
        M[ 6, alpha] = (5.0 * norm_sq - 9.0) * e_alpha[1]
        M[ 7, alpha] = e_alpha[2]
        M[ 8, alpha] = (5.0 * norm_sq - 9.0) * e_alpha[2]
        M[ 9, alpha] = 3.0 * e_alpha[0]**2 - norm_sq
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
    identity_error_max = np.inf
    try:
        M_inv = np.linalg.inv(M)
        # Check inversion quality using max error (more meaningful than sum)
        identity_check = np.dot(M, M_inv)
        identity_error_max = np.max(np.abs(identity_check - np.eye(19)))
    except np.linalg.LinAlgError:
        M_inv = None
        identity_error_max = np.inf
    
    return M, M_inv, identity_error_max


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


def parse_directions_input(input_str, default_list):
    """
    Safely parse comma/space-separated string to a list of 19 integers.
    Accepts numbers with or without signs: 0, +1, -5, 1, 8, etc.
    
    Parameters:
    -----------
    input_str : str
        Input string with comma or space-separated values
    default_list : list
        Default list to return if parsing fails
    
    Returns:
    --------
    list
        List of 19 integers
    """
    # Split by any whitespace or comma, filter out empty strings
    # Handle both comma and space separators, and strip whitespace
    parts = [s.strip() for s in input_str.replace(',', ' ').split() if s.strip()]
    
    if len(parts) != 19:
        st.warning(f"Direction vector must have exactly 19 values. Found {len(parts)}. Using defaults temporarily.")
        return default_list
    
    try:
        # Parse each part, handling +, -, or no sign
        # int() already handles +1, -5, 0, 1, etc.
        result = []
        for p in parts:
            # Remove any extra whitespace and parse
            cleaned = p.strip()
            # Handle explicit + sign (int() accepts it, but we ensure it's clean)
            if cleaned.startswith('+'):
                cleaned = cleaned[1:].strip()
            result.append(int(cleaned))
        return result
    except (ValueError, AttributeError):
        st.error("All direction vector entries must be integers (e.g., 0, +1, -5, 1, 8).")
        return default_list


def get_column_styles(df, column_colors):
    """
    Generate column-specific background colors for a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to style
    column_colors : list
        List of colors (one per column)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with style strings for each cell
    """
    styles = {}
    for i, col in enumerate(df.columns):
        color = column_colors[i % len(column_colors)]
        styles[col] = [f'background-color: {color}'] * len(df)
    return pd.DataFrame(styles, index=df.index)


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
    
    col1, col2 = st.columns([1.5, 0.7])
    
    with col1:
        st.subheader("Lattice Directions")
        st.caption("Enter D3Q19 velocity directions (19 components each, space or comma separated)")
        
        use_defaults = st.checkbox("Use default directions (from OpenACC_parameters.F90)", value=True, key="mrt_use_defaults")
        
        # Store defaults as strings for text area
        default_dir_str_x = " ".join(map(str, default_dirx))
        default_dir_str_y = " ".join(map(str, default_diry))
        default_dir_str_z = " ".join(map(str, default_dirz))
        
        if use_defaults:
            dirx = default_dirx
            diry = default_diry
            dirz = default_dirz
            st.info("Using default D3Q19 directions from Fortran parameters module")
            
            # Show the defaults in read-only text areas for reference
            st.text_area("X Directions", value=default_dir_str_x, height=40, disabled=True, key="dirx_default_display")
            st.text_area("Y Directions", value=default_dir_str_y, height=40, disabled=True, key="diry_default_display")
            st.text_area("Z Directions", value=default_dir_str_z, height=40, disabled=True, key="dirz_default_display")
        else:
            st.markdown("**X directions:**")
            dirx_str = st.text_area("", value=default_dir_str_x, key="dirx_input", height=60, help="Enter 19 integers separated by spaces or commas")
            dirx = parse_directions_input(dirx_str, default_dirx)
            
            st.markdown("**Y directions:**")
            diry_str = st.text_area("", value=default_dir_str_y, key="diry_input", height=60, help="Enter 19 integers separated by spaces or commas")
            diry = parse_directions_input(diry_str, default_diry)
            
            st.markdown("**Z directions:**")
            dirz_str = st.text_area("", value=default_dir_str_z, key="dirz_input", height=60, help="Enter 19 integers separated by spaces or commas")
            dirz = parse_directions_input(dirz_str, default_dirz)
    
    with col2:
        st.subheader("Relaxation Parameters")
        st.caption("Set relaxation rates (matching Fortran init_mrt_edm)")
        
        nu = st.number_input("Kinematic viscosity (ν)", value=0.002546479089469996, format="%.10f", key="mrt_nu")
        tau = st.number_input("Relaxation time (τ)", value=3.0 * 0.002546479089469996 + 0.5, format="%.10f", key="mrt_tau",
                              help=r"$\tau = 3\nu + 0.5$, $\quad \nu = \frac{1}{3}\left(\frac{1}{s_\nu} - \frac{1}{2}\right)$")
        
        st.markdown("**Mode-specific relaxation rates:**")
        s_e = st.number_input("s_e (energy mode)", value=1.19, format="%.2f", key="mrt_se")
        s_eps = st.number_input("s_eps (higher-order energy)", value=1.4, format="%.2f", key="mrt_seps")
        s_q = st.number_input("s_q (energy flux)", value=1.2, format="%.2f", key="mrt_sq")
        s_pi = st.number_input("s_pi (non-hydro stress)", value=1.4, format="%.2f", key="mrt_spi")
        s_other = st.number_input("s_other (ghost modes)", value=1.98, format="%.2f", key="mrt_sother")
    
    st.markdown("---")
    
    # Compute matrices
    if st.button("🔧 Generate MRT Matrix", type="primary", width='stretch'):
        with st.spinner("Computing MRT matrix..."):
            # Validate directions
            is_valid, errors, warnings = validate_d3q19_directions(dirx, diry, dirz)
            
            if not is_valid:
                for error in errors:
                    st.error(f"❌ {error}")
                st.stop()
            
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")
            
            # Compute matrices
            M, M_inv, identity_error_max = compute_mrt_matrix(dirx, diry, dirz)
            S = compute_relaxation_vector(tau=tau, nu=nu, s_e=s_e, s_eps=s_eps, s_q=s_q, s_pi=s_pi, s_other=s_other)
            
            # Store error for display
            st.session_state.mrt_identity_error_max = identity_error_max
            
            if M_inv is not None:
                st.session_state.mrt_M = M
                st.session_state.mrt_M_inv = M_inv
                st.session_state.mrt_S = S
                st.session_state.mrt_dirx = dirx
                st.session_state.mrt_diry = diry
                st.session_state.mrt_dirz = dirz
                
                # Improved success message
                if identity_error_max < 1e-8:
                    st.success("✅ MRT matrix generated and inverted successfully!")
                elif identity_error_max < 1e-6:
                    st.warning(f"⚠️ Matrix inverted but with moderate error ({identity_error_max:.2e})")
                else:
                    st.error(f"❌ Matrix inversion error is large ({identity_error_max:.2e}). Matrix may be ill-conditioned.")
            else:
                st.error("❌ Matrix is singular and cannot be inverted!")
                st.error("This usually indicates duplicate or invalid direction vectors. Check your D3Q19 directions.")
    
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
        
        tab_m, tab_minv, tab_s = st.tabs(["Matrix M", "Matrix M⁻¹", "Vector S & Matrix S"])
        
        # Define column colors (cycle through 5 light pastel colors for 19 columns)
        base_colors = ['#E5E7F6', '#F6E5E7', '#E5F6E7', '#F6F6E7', '#E7E5F6']
        column_colors = [base_colors[i % len(base_colors)] for i in range(19)]
        
        with tab_m:
            st.markdown("**Transformation Matrix M (19x19)**")
            st.caption("Transforms distribution functions f to moment space: **m = M f**")
            
            # Display as DataFrame for better formatting
            df_M = pd.DataFrame(M, 
                               index=[f"Row {i+1}: {moment_names[i]}" for i in range(19)],
                               columns=[f"a={i+1}" for i in range(19)])
            
            # Apply column coloring and format as integers (no decimals)
            styled_M = df_M.style.apply(lambda df: get_column_styles(df, column_colors), axis=None).format("{:.0f}")
            st.dataframe(styled_M, height=600, width='stretch')
            
            # Download button
            csv_M = df_M.to_csv()
            st.download_button(
                label="📥 Download M matrix (CSV)",
                data=csv_M,
                file_name="MRT_matrix_M.csv",
                mime="text/csv"
            )
            
            # Capture to report
            try:
                from utils.report_builder import capture_button
                capture_button(df=df_M, title="MRT Transformation Matrix M (19×19)", source_page="Theory & Equations")
            except ImportError:
                pass
        
        with tab_minv:
            st.markdown("**Inverse Transformation Matrix M⁻¹ (19x19)**")
            st.caption("Transforms moments back to distribution functions: **f = M⁻¹ m**")
            
            df_M_inv = pd.DataFrame(M_inv,
                                    index=[f"Row {i+1}: {moment_names[i]}" for i in range(19)],
                                    columns=[f"a={i+1}" for i in range(19)])
            
            # Apply column coloring and format with 6 decimals
            styled_M_inv = df_M_inv.style.apply(lambda df: get_column_styles(df, column_colors), axis=None).format("{:.6f}")
            st.dataframe(styled_M_inv, height=600, width='stretch')
            
            csv_M_inv = df_M_inv.to_csv()
            st.download_button(
                label="📥 Download M⁻¹ matrix (CSV)",
                data=csv_M_inv,
                file_name="MRT_matrix_M_inv.csv",
                mime="text/csv"
            )
            
            # Capture to report
            try:
                from utils.report_builder import capture_button
                capture_button(df=df_M_inv, title="MRT Inverse Matrix M⁻¹ (19×19)", source_page="Theory & Equations")
            except ImportError:
                pass
        
        with tab_s:
            st.markdown("**Relaxation Rate Vector S (19×1)**")
            st.caption("Relaxation rates for each moment mode")
            
            df_S = pd.DataFrame({
                "Moment": moment_names,
                "Relaxation Rate": S
            })
            
            # Apply column coloring for S vector (2 columns)
            S_column_colors = ['#f0f2f6', base_colors[0]]
            styled_S = df_S.style.apply(lambda df: get_column_styles(df, S_column_colors), axis=None)
            
            # Reduce width of S vector table
            col_s1, col_s2, col_s3 = st.columns([0.2, 0.6, 0.2])
            with col_s2:
                st.dataframe(styled_S, width='stretch', hide_index=True)
            
            csv_S = df_S.to_csv(index=False)
            st.download_button(
                label="📥 Download S vector (CSV)",
                data=csv_S,
                file_name="MRT_relaxation_vector_S.csv",
                mime="text/csv"
            )
            
            # Capture to report
            try:
                from utils.report_builder import capture_button
                capture_button(df=df_S, title="MRT Relaxation Rate Vector S", source_page="Theory & Equations")
            except ImportError:
                pass
            
            st.markdown("---")
            st.markdown("**Diagonal Relaxation Matrix Diag(S) (19x19)**")
            st.caption("Diagonal matrix with relaxation rates on the diagonal: **Diag(S) = diag(S)**")
            
            S_diag = np.diag(S)
            df_S_diag = pd.DataFrame(S_diag,
                                    index=[f"Row {i+1}: {moment_names[i]}" for i in range(19)],
                                    columns=[f"a={i+1}" for i in range(19)])
            
            # Format: diagonal entries with decimals, off-diagonal (zeros) as integers
            # Apply column coloring
            def format_diag_S(val):
                return f"{val:.3f}" if abs(val) > 1e-10 else "0"
            
            styled_df = df_S_diag.style.apply(lambda df: get_column_styles(df, column_colors), axis=None).format(format_diag_S)
            st.dataframe(styled_df, height=600, width='stretch')
            
            csv_S_diag = df_S_diag.to_csv()
            st.download_button(
                label="📥 Download Diag(S) matrix (CSV)",
                data=csv_S_diag,
                file_name="MRT_diagonal_matrix_Diag_S.csv",
                mime="text/csv"
            )
            
            # Capture to report
            try:
                from utils.report_builder import capture_button
                capture_button(df=df_S_diag, title="MRT Diagonal Relaxation Matrix Diag(S) (19×19)", source_page="Theory & Equations")
            except ImportError:
                pass
        
        # Verification
        st.markdown("---")
        st.subheader("Verification")
        
        # Get stored error
        identity_error_max = st.session_state.get("mrt_identity_error_max", np.inf)
        
        # Recompute if not stored (backward compatibility)
        if identity_error_max == np.inf:
            identity_check = np.dot(M, M_inv)
            identity_error_max = np.max(np.abs(identity_check - np.eye(19)))
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric("Matrix Inversion Error (max)", f"{identity_error_max:.2e}", 
                     help="Maximum absolute error in M × M⁻¹ - I. Should be < 1e-8 for good inversion.")
        with col_v2:
            if identity_error_max < 1e-8:
                st.success("✅ Matrix inversion is accurate")
            elif identity_error_max < 1e-6:
                st.warning("⚠️ Matrix inversion has moderate error")
            else:
                st.error("❌ Matrix inversion has large error")
        
        # Display direction summary
        st.markdown("**Lattice Directions Used:**")
        dir_df = pd.DataFrame({

            "a": range(1, 20),
            "cx": st.session_state.mrt_dirx,
            "cy": st.session_state.mrt_diry,
            "cz": st.session_state.mrt_dirz
        })
        # Apply column coloring for directions table (4 columns)
        dir_column_colors = ['#f0f2f6', base_colors[0], base_colors[1], base_colors[2]]
        styled_dir_df = dir_df.style.apply(lambda df: get_column_styles(df, dir_column_colors), axis=None)
        col_dir1, col_dir2, col_dir3 = st.columns([0.2, 0.6, 0.2])
        with col_dir2:
            st.dataframe(styled_dir_df, width='stretch', hide_index=True)
