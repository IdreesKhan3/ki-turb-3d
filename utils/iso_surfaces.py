"""
Turbulence metrics and visualization methods
Computes Q-criterion, Q_S^S method, and related invariants for vortex visualization
"""

import numpy as np
from typing import Tuple


def compute_velocity_gradient_tensor(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Compute velocity gradient tensor A_ij = ∂u_i/∂x_j
    Uses periodic boundary conditions with central differences (matching Fortran implementation)
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components (ux, uy, uz)
        dx, dy, dz: Grid spacing (default 1.0)
        
    Returns:
        (nx, ny, nz, 3, 3) array where A[i,j,k,:,:] is the 3x3 gradient tensor at point (i,j,k)
    """
    ux = velocity[:, :, :, 0]
    uy = velocity[:, :, :, 1]
    uz = velocity[:, :, :, 2]
    
    # Compute all velocity gradients using periodic boundary conditions
    # Following Fortran: (u(ip) - u(im)) / (2*dx) with periodic wrap
    # Using np.roll for periodic boundaries: roll(-1) = i+1, roll(+1) = i-1
    
    # ∂ux/∂x, ∂ux/∂y, ∂ux/∂z
    dux_dx = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dx)
    dux_dy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dy)
    dux_dz = (np.roll(ux, -1, axis=2) - np.roll(ux, 1, axis=2)) / (2.0 * dz)
    
    # ∂uy/∂x, ∂uy/∂y, ∂uy/∂z
    duy_dx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dx)
    duy_dy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dy)
    duy_dz = (np.roll(uy, -1, axis=2) - np.roll(uy, 1, axis=2)) / (2.0 * dz)
    
    # ∂uz/∂x, ∂uz/∂y, ∂uz/∂z
    duz_dx = (np.roll(uz, -1, axis=0) - np.roll(uz, 1, axis=0)) / (2.0 * dx)
    duz_dy = (np.roll(uz, -1, axis=1) - np.roll(uz, 1, axis=1)) / (2.0 * dy)
    duz_dz = (np.roll(uz, -1, axis=2) - np.roll(uz, 1, axis=2)) / (2.0 * dz)
    
    # Build gradient tensor A_ij
    nx, ny, nz = velocity.shape[:3]
    A = np.zeros((nx, ny, nz, 3, 3))
    
    A[:, :, :, 0, 0] = dux_dx  # ∂ux/∂x
    A[:, :, :, 0, 1] = dux_dy  # ∂ux/∂y
    A[:, :, :, 0, 2] = dux_dz  # ∂ux/∂z
    A[:, :, :, 1, 0] = duy_dx  # ∂uy/∂x
    A[:, :, :, 1, 1] = duy_dy  # ∂uy/∂y
    A[:, :, :, 1, 2] = duy_dz  # ∂uy/∂z
    A[:, :, :, 2, 0] = duz_dx  # ∂uz/∂x
    A[:, :, :, 2, 1] = duz_dy  # ∂uz/∂y
    A[:, :, :, 2, 2] = duz_dz  # ∂uz/∂z
    
    return A


def compute_rotation_deformation_tensors(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rotation tensor Ω_ij and deformation tensor S_ij
    
    Ω_ij = (A_ij - A_ji) / 2  (antisymmetric part)
    S_ij = (A_ij + A_ji) / 2  (symmetric part)
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing
        
    Returns:
        (Omega, S) where both are (nx, ny, nz, 3, 3) arrays
    """
    A = compute_velocity_gradient_tensor(velocity, dx, dy, dz)
    
    # Transpose A to get A_ji: swap last two dimensions
    A_T = np.transpose(A, (0, 1, 2, 4, 3))
    
    # Rotation tensor (antisymmetric)
    Omega = 0.5 * (A - A_T)
    
    # Deformation tensor (symmetric)
    S = 0.5 * (A + A_T)
    
    return Omega, S


def compute_vorticity_vector(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Compute vorticity vector ω = ∇ × u
    Uses periodic boundary conditions with central differences (matching Fortran implementation)
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing
        
    Returns:
        (nx, ny, nz, 3) array of vorticity components (ωx, ωy, ωz)
    """
    ux = velocity[:, :, :, 0]
    uy = velocity[:, :, :, 1]
    uz = velocity[:, :, :, 2]
    
    # Compute gradients using periodic boundary conditions
    # Following Fortran: (u(ip) - u(im)) / (2*dx) with periodic wrap
    dux_dx = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dx)
    dux_dy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dy)
    dux_dz = (np.roll(ux, -1, axis=2) - np.roll(ux, 1, axis=2)) / (2.0 * dz)
    
    duy_dx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dx)
    duy_dy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dy)
    duy_dz = (np.roll(uy, -1, axis=2) - np.roll(uy, 1, axis=2)) / (2.0 * dz)
    
    duz_dx = (np.roll(uz, -1, axis=0) - np.roll(uz, 1, axis=0)) / (2.0 * dx)
    duz_dy = (np.roll(uz, -1, axis=1) - np.roll(uz, 1, axis=1)) / (2.0 * dy)
    duz_dz = (np.roll(uz, -1, axis=2) - np.roll(uz, 1, axis=2)) / (2.0 * dz)
    
    # Vorticity: ω = ∇ × u
    omega_x = duz_dy - duy_dz
    omega_y = dux_dz - duz_dx
    omega_z = duy_dx - dux_dy
    
    vorticity = np.zeros_like(velocity)
    vorticity[:, :, :, 0] = omega_x
    vorticity[:, :, :, 1] = omega_y
    vorticity[:, :, :, 2] = omega_z
    
    return vorticity


def compute_qs_s(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Compute Q_S^S method for vortex visualization
    
    Q_S^S = [(Q_W^3 + Q_S^3) + (Σ^2 - R_s^2)]^(1/3)
    
    Where:
    - Q_W = (1/2) Ω_ij Ω_ij  (rotation rate strength)
    - Q_S = -(1/2) S_ij S_ij  (deformation rate strength)
    - Σ = ω_i S_ij ω_j  (enstrophy production term)
    - R_s = -(1/3) S_ij S_jk S_ki  (strain rate production)
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing
        
    Returns:
        (nx, ny, nz) array of Q_S^S values
    """
    Omega, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    
    nx, ny, nz = velocity.shape[:3]
    
    # Vectorized computation using einsum for efficiency
    # Q_W = (1/2) Ω_ij Ω_ij
    Q_W = 0.5 * np.einsum('ijklm,ijklm->ijk', Omega, Omega)
    
    # Q_S = -(1/2) S_ij S_ij
    Q_S = -0.5 * np.einsum('ijklm,ijklm->ijk', S, S)
    
    # Σ = ω_i S_ij ω_j
    Sigma = np.einsum('ijkl,ijklm,ijkm->ijk', vorticity, S, vorticity)
    
    # R_s = -(1/3) S_ij S_jk S_ki = -(1/3) trace(S^3)
    S_cubed = np.einsum('ijklm,ijkmn,ijknl->ijk', S, S, S)
    R_s = -(1.0/3.0) * S_cubed
    
    # Q_S^S = [(Q_W^3 + Q_S^3) + (Σ^2 - R_s^2)]^(1/3)
    # Exact implementation per paper formula
    Q_W_cubed = Q_W ** 3
    Q_S_cubed = Q_S ** 3
    Sigma_squared = Sigma ** 2
    R_s_squared = R_s ** 2
    
    # Compute argument: [(Q_W^3 + Q_S^3) + (Σ^2 - R_s^2)]
    # Grouping matches paper: (Q_W^3 + Q_S^3) + (Σ^2 - R_s^2)
    arg = (Q_W_cubed + Q_S_cubed) + (Sigma_squared - R_s_squared)
    
    # Real cube root: preserve sign for physically correct result
    # In vortical regions, Q_S^S > 0; in strain-dominated regions, Q_S^S < 0
    qs_s = np.sign(arg) * np.power(np.abs(arg), 1.0/3.0)
    
    # Handle zero case to avoid numerical issues
    qs_s = np.where(np.abs(arg) < 1e-15, 0.0, qs_s)
    
    return qs_s


def compute_q_invariant(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Compute second invariant Q of velocity gradient tensor
    
    Q = -(1/2) A_ij A_ij = (1/4)(ω_i ω_i - 2 S_ij S_ij)
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing
        
    Returns:
        (nx, ny, nz) array of Q values
    """
    Omega, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    
    # Q = (1/4)(ω_i ω_i - 2 S_ij S_ij)
    omega_mag_sq = np.sum(vorticity * vorticity, axis=3)
    S_squared_sum = np.einsum('ijklm,ijklm->ijk', S, S)
    Q = 0.25 * (omega_mag_sq - 2.0 * S_squared_sum)
    
    return Q


def compute_r_invariant(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Compute third invariant R of velocity gradient tensor
    
    R = -(1/3) A_ij A_jk A_ki = -(1/3)(S_ij S_jk S_ki + (3/4) ω_i S_ij ω_j)
    
    Note: Since S_ij is symmetric, ω_i S_ij ω_j = ω_i ω_j S_ij
    
    Args:
        velocity: (nx, ny, nz, 3) array of velocity components
        dx, dy, dz: Grid spacing
        
    Returns:
        (nx, ny, nz) array of R values
    """
    Omega, S = compute_rotation_deformation_tensors(velocity, dx, dy, dz)
    vorticity = compute_vorticity_vector(velocity, dx, dy, dz)
    
    # S_ij S_jk S_ki (trace of S^3)
    S_cubed = np.einsum('ijklm,ijkmn,ijknl->ijk', S, S, S)
    
    # ω_i S_ij ω_j (equivalent to ω_i ω_j S_ij since S is symmetric)
    omega_omega_S = np.einsum('ijkl,ijklm,ijkm->ijk', vorticity, S, vorticity)
    
    R = -(1.0/3.0) * (S_cubed + 0.75 * omega_omega_S)
    
    return R

