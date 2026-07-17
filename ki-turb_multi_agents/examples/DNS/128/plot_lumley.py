#!/usr/bin/env python3
"""Compute Lumley triangle from isotropy_validation.dat and generate plot."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from visualizations.real_isotropy_vis import create_lumley_triangle_figure
from utils.plot_style import default_plot_style

data = np.loadtxt(HERE / "isotropy_validation.dat")

# Columns: t, idx, R11, R12, R22, R13, R33, R23, b11, |b22|, |b33|, PASS/FAIL_text
# b22 and b33 stored as absolute values; need sign.
# Since b11 > 0 and trace(b)=0, b22 and b33 are negative.

b11 = data[:, 7]
b22_abs = data[:, 8]
b33_abs = data[:, 9]

b22 = -b22_abs
b33 = -b33_abs

trace = b11 + b22 + b33
print(f"Trace range: {trace.min():.6f} to {trace.max():.6f}")

II = b11**2 + b22**2 + b33**2
III = b11**3 + b22**3 + b33**3

print(f"II range: {II.min():.6f} to {II.max():.6f}")
print(f"III range: {III.min():.6f} to {III.max():.6f}")

eta = np.sqrt(np.maximum(-II / 3.0, 0.0))
xi = np.cbrt(III / 2.0)

print(f"ξ range: {xi.min():.6f} to {xi.max():.6f}")
print(f"η range: {eta.min():.6f} to {eta.max():.6f}")
print(f"Start: ξ={xi[0]:.6f}, η={eta[0]:.6f}")
print(f"End:   ξ={xi[-1]:.6f}, η={eta[-1]:.6f}")

ps = default_plot_style()
ps["template"] = "plotly_dark"
ps["show_plot_title"] = True
ps["plot_title"] = "Lumley Triangle — DNS 128³"

fig = create_lumley_triangle_figure(xi, eta, ps)

out_path = HERE / "lumley_triangle.html"
fig.write_html(str(out_path))
print(f"Saved to {out_path}")

png_path = HERE / "lumley_triangle.png"
fig.write_image(str(png_path), width=800, height=700)
print(f"Saved to {png_path}")
