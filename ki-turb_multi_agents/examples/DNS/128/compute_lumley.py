#!/usr/bin/env python3
"""Compute Lumley triangle (xi, eta) from isotropy_validation.dat and create plot."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("."))

# Read data
data = np.loadtxt("examples/DNS/128/isotropy_validation.dat", 
                   comments=None, dtype=str)

# Columns: timestep, idx, R11, R12, R22, R13, R33, R23, b11, mb22, mb33, status
timesteps = data[:, 0].astype(float)
b11 = data[:, 8].astype(float)       # b11
mb22 = data[:, 9].astype(float)      # -b22 (since b22 is typically negative)
mb33 = data[:, 10].astype(float)     # -b33
b12 = data[:, 3].astype(float)       # R12 (since 2k=1, b12 = R12)
b13 = data[:, 5].astype(float)
b23 = data[:, 7].astype(float)

# Reconstruct full b_ij
# b22 = -mb22, b33 = -mb33 when b22,b33 < 0
# But we need correct signs. Since trace(b)=0: b11 + b22 + b33 = 0
# So b22 + b33 = -b11. Given mb22 and mb33 are the magnitudes:
# b22 = -mb22 if b22 < 0 else mb22 (but trace forces at least one negative when b11>0)
# For data: when b11 > 0, both b22 and b33 are negative (two are negative, one positive)
# When b11 < 0, one of b22/b33 is positive
# Strategy: use b22 = -(b11 + b33), but we don't know b33 directly either.
# Simple approach: compute from R directly

R11 = data[:, 2].astype(float)
R22 = data[:, 4].astype(float)
R33 = data[:, 6].astype(float)
two_k = R11 + R22 + R33

b11_direct = R11 / two_k - 1/3
b22_direct = R22 / two_k - 1/3
b33_direct = R33 / two_k - 1/3
b12_direct = data[:, 3].astype(float) / two_k
b13_direct = data[:, 5].astype(float) / two_k
b23_direct = data[:, 7].astype(float) / two_k

n = len(b11_direct)
II = np.empty(n)
III = np.empty(n)

for q in range(n):
    B = np.array([[b11_direct[q], b12_direct[q], b13_direct[q]],
                  [b12_direct[q], b22_direct[q], b23_direct[q]],
                  [b13_direct[q], b23_direct[q], b33_direct[q]]])
    II[q] = -0.5 * np.trace(B @ B)   # matches core_physics convention
    III[q] = np.linalg.det(B)

eta = np.sqrt(np.maximum(-II / 3, 0))
xi = np.cbrt(III / 2)

# Print summary
print(f"Loaded {n} timesteps from isotropy_validation.dat")
print(f"Time range: {timesteps[0]:.0f} - {timesteps[-1]:.0f}")
print(f"xi range: [{xi.min():.6f}, {xi.max():.6f}]")
print(f"eta range: [{eta.min():.6f}, {eta.max():.6f}]")
print(f"First point: xi={xi[0]:.6f}, eta={eta[0]:.6f}")
print(f"Last point:  xi={xi[-1]:.6f}, eta={eta[-1]:.6f}")

# Save computed invariants
np.savetxt("examples/DNS/128/lumley_invariants.csv",
           np.column_stack([timesteps, xi, eta, II, III]),
           header="timestep,xi,eta,II,III",
           delimiter=",", fmt="%.10f")
print("Saved lumley_invariants.csv")
