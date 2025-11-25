"""
IK-TURB 3D Logo Generator
Creates a professional physics-oriented fluid dynamics logo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

def create_vortex_field(x, y):
    """Create a vortex velocity field"""
    X, Y = np.meshgrid(x, y)
    
    # Center of vortex
    cx, cy = 0.0, 0.0
    
    # Distance from center
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Vortex velocity components (tangential flow)
    theta = np.arctan2(Y - cy, X - cx)
    strength = 2.0
    u = -strength * np.sin(theta) / (r + 0.1)
    v = strength * np.cos(theta) / (r + 0.1)
    
    # Add some background flow
    u += 0.3 * np.ones_like(X)
    
    return u, v

def create_energy_spectrum_curve():
    """Create an energy spectrum curve (Kolmogorov -5/3)"""
    k = np.logspace(-1, 1, 100)
    E = 0.5 * k**(-5/3)  # Kolmogorov spectrum
    return k, E

def create_logo(output_path="logo.png", format="png", size=(800, 400), dpi=300):
    """
    Generate IK-TURB 3D logo with fluid dynamics elements
    
    Parameters
    ----------
    output_path : str
        Output file path
    format : str
        Output format: 'png' or 'svg'
    size : tuple
        Logo size in pixels (width, height)
    dpi : int
        Resolution for PNG
    """
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(size[0]/dpi, size[1]/dpi), facecolor='none')
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    ax.axis('off')
    
    # Set limits
    xlim = (-2, 2)
    ylim = (-1, 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    
    # Color scheme: Fluid dynamics blue/teal
    primary_color = '#1E88E5'  # Blue
    secondary_color = '#00ACC1'  # Teal
    accent_color = '#004D40'  # Dark teal
    text_color = '#0D47A1'  # Dark blue
    
    # ============================================
    # 1. Vortex/Streamline Pattern
    # ============================================
    x = np.linspace(-1.5, 1.5, 50)
    y = np.linspace(-0.8, 0.8, 40)
    u, v = create_vortex_field(x, y)
    
    # Draw streamlines
    X, Y = np.meshgrid(x, y)
    
    # Create streamline-like curves (spiral patterns)
    n_spirals = 3
    for i in range(n_spirals):
        angle_offset = i * 2 * np.pi / n_spirals
        t = np.linspace(0, 4*np.pi, 200)
        r = 0.3 + 0.2 * t / (4*np.pi)
        x_spiral = r * np.cos(t + angle_offset) - 0.5
        y_spiral = r * np.sin(t + angle_offset)
        
        ax.plot(x_spiral, y_spiral, color=primary_color, 
               linewidth=2.5, alpha=0.6, zorder=2)
    
    # Add smaller vortices
    for i, (cx, cy) in enumerate([(0.8, 0.3), (-0.6, -0.4)]):
        t = np.linspace(0, 2*np.pi, 100)
        r = 0.15
        x_v = cx + r * np.cos(t)
        y_v = cy + r * np.sin(t)
        ax.plot(x_v, y_v, color=secondary_color, 
               linewidth=1.5, alpha=0.5, zorder=2)
    
    # ============================================
    # 2. 3D Grid Visualization (Isometric)
    # ============================================
    # Draw a simplified 3D lattice representation
    grid_color = accent_color
    grid_alpha = 0.2
    
    # Front face of cube
    cube_x = [1.2, 1.5, 1.5, 1.2, 1.2]
    cube_y = [-0.3, -0.1, 0.1, -0.1, -0.3]
    ax.plot(cube_x, cube_y, color=grid_color, linewidth=1.5, 
           alpha=grid_alpha, zorder=1)
    
    # Top face
    top_x = [1.2, 1.5, 1.6, 1.3, 1.2]
    top_y = [-0.1, 0.1, 0.2, 0.0, -0.1]
    ax.plot(top_x, top_y, color=grid_color, linewidth=1.5, 
           alpha=grid_alpha, zorder=1)
    
    # Right face
    right_x = [1.5, 1.6, 1.6, 1.5, 1.5]
    right_y = [0.1, 0.2, 0.0, -0.1, 0.1]
    ax.plot(right_x, right_y, color=grid_color, linewidth=1.5, 
           alpha=grid_alpha, zorder=1)
    
    # Grid lines inside
    for i in range(2):
        x_line = [1.2 + i*0.15, 1.3 + i*0.15]
        y_line = [-0.1 + i*0.1, 0.0 + i*0.1]
        ax.plot(x_line, y_line, color=grid_color, linewidth=0.8, 
               alpha=grid_alpha*0.7, zorder=1)
    
    # ============================================
    # 3. Main Text: "IK-TURB 3D"
    # ============================================
    # Primary text
    ax.text(-1.4, 0.15, 'IK-TURB', 
           fontsize=42, fontweight='bold', 
           color=text_color, family='sans-serif',
           ha='left', va='center', zorder=10)
    
    # "3D" text (smaller, positioned)
    ax.text(0.3, 0.15, '3D', 
           fontsize=32, fontweight='bold',
           color=primary_color, family='sans-serif',
           ha='left', va='center', zorder=10)
    
    # Subtitle
    ax.text(-1.4, -0.25, 'Turbulence Visualization & Analysis Suite',
           fontsize=12, fontweight='normal',
           color=accent_color, family='sans-serif',
           ha='left', va='center', zorder=10, style='italic')
    
    # ============================================
    # 4. Energy Cascade Field & Backscattering
    # ============================================
    # Add small flow arrows
    arrow_positions = [(-1.0, -0.6), (-0.3, -0.7), (0.4, -0.65)]
    for ax_x, ax_y in arrow_positions:
        arrow = mpatches.FancyArrowPatch(
            (ax_x, ax_y), (ax_x + 0.3, ax_y),
            arrowstyle='->', mutation_scale=15,
            color=secondary_color, alpha=0.4, linewidth=1.5, zorder=3
        )
        ax.add_patch(arrow)
    
    # Add small dots representing particles
    particle_positions = [(-0.8, 0.5), (0.2, 0.6), (1.0, 0.4)]
    for px, py in particle_positions:
        circle = Circle((px, py), 0.03, color=primary_color, 
                       alpha=0.6, zorder=4)
        ax.add_patch(circle)
    
    # ============================================
    # Save
    # ============================================
    plt.tight_layout(pad=0)
    
    if format.lower() == 'svg':
        fig.savefig(output_path, format='svg', transparent=True, 
                   bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig(output_path, format='png', dpi=dpi, transparent=True,
                   bbox_inches='tight', pad_inches=0.1)
    
    plt.close()
    print(f"✅ Logo saved to: {output_path}")

def create_icon(output_path="icon.png", size=256, dpi=300):
    """
    Create a square icon version (for favicon/app icon)
    """
    fig = plt.figure(figsize=(size/dpi, size/dpi), facecolor='none')
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    ax.axis('off')
    
    # Set limits for square
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    
    # Colors
    primary_color = '#1E88E5'
    secondary_color = '#00ACC1'
    text_color = '#0D47A1'
    
    # Create vortex pattern in center
    t = np.linspace(0, 3*np.pi, 150)
    r = 0.2 + 0.3 * t / (3*np.pi)
    x_vortex = r * np.cos(t)
    y_vortex = r * np.sin(t)
    
    ax.plot(x_vortex, y_vortex, color=primary_color, 
           linewidth=3, alpha=0.8, zorder=2)
    
    # Add "3D" text
    ax.text(0, -0.1, '3D', 
           fontsize=48, fontweight='bold',
           color=text_color, family='sans-serif',
           ha='center', va='center', zorder=10)
    
    # Small particles
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        px = 0.6 * np.cos(angle)
        py = 0.6 * np.sin(angle)
        circle = Circle((px, py), 0.04, color=secondary_color, 
                       alpha=0.7, zorder=3)
        ax.add_patch(circle)
    
    plt.tight_layout(pad=0)
    fig.savefig(output_path, format='png', dpi=dpi, transparent=True,
               bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Icon saved to: {output_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("IK-TURB 3D Logo Generator")
    print("=" * 60)
    print()
    
    # Generate main logo
    print("Generating main logo...")
    create_logo("logo.png", format="png", size=(1200, 400), dpi=300)
    create_logo("logo.svg", format="svg", size=(1200, 400))
    
    # Generate icon
    print("\nGenerating icon...")
    create_icon("icon.png", size=256, dpi=300)
    
    print("\n" + "=" * 60)
    print("✅ All logo files generated successfully!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - logo.png (main logo, high-res PNG)")
    print("  - logo.svg (main logo, vector format)")
    print("  - icon.png (square icon for favicon)")
    print("\n💡 You can now use these in your Streamlit app!")

