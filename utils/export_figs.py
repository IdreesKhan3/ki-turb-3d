"""
Shared export utilities for Plotly figures
Provides a consistent export panel interface across all pages
"""

import streamlit as st
from pathlib import Path
from plotly.graph_objects import Figure


# Export format definitions
_EXPORT_FORMATS = {
    "PNG (raster)": "png",
    "PDF (vector)": "pdf",
    "SVG (vector)": "svg",
    "EPS (vector)": "eps",
    "JPG/JPEG (raster)": "jpg",
    "WEBP (raster)": "webp",
    "TIFF (raster)": "tiff",
    "HTML (interactive)": "html",
}


def export_panel(fig: Figure, out_dir: Path, base_name: str):
    """
    Export figure to multiple research formats using kaleido.
    
    Provides a Streamlit UI panel for exporting Plotly figures to various formats.
    This is the shared export function used across all pages for consistency.
    
    Args:
        fig: Plotly figure object to export
        out_dir: Directory path where exported files will be saved
        base_name: Base filename (without extension) for exported files
        
    Example:
        >>> from utils.export_figs import export_panel
        >>> from pathlib import Path
        >>> export_panel(fig, Path("./outputs"), "my_figure")
    """
    with st.expander(f"📤 Export figure: {base_name}", expanded=False):
        fmts = st.multiselect(
            "Select export format(s)",
            list(_EXPORT_FORMATS.keys()),
            default=["PNG (raster)", "PDF (vector)", "SVG (vector)"],
            key=f"{base_name}_fmts"
        )
        
        c1, c2, c3 = st.columns(3)
        with c1:
            scale = st.slider("Scale (like DPI)", 1.0, 6.0, 3.0, 0.5, key=f"{base_name}_scale")
        with c2:
            width_px = st.number_input("Width px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_wpx")
        with c3:
            height_px = st.number_input("Height px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_hpx")
        
        if st.button("Export selected formats", key=f"{base_name}_doexport"):
            if not fmts:
                st.warning("Please select at least one format.")
                return
            
            errors = []
            for f_label in fmts:
                ext = _EXPORT_FORMATS[f_label]
                out = out_dir / f"{base_name}.{ext}"
                
                try:
                    if ext == "html":
                        fig.write_html(str(out))
                    else:
                        kwargs = {}
                        if width_px > 0:
                            kwargs["width"] = int(width_px)
                        if height_px > 0:
                            kwargs["height"] = int(height_px)
                        fig.write_image(str(out), scale=scale, **kwargs)
                except Exception as e:
                    errors.append((out.name, str(e)))
            
            if errors:
                st.error(
                    "Some exports failed. Ensure kaleido is installed:\n"
                    "pip install -U kaleido\n\n"
                    + "\n".join([f"- {n}: {msg}" for n, msg in errors])
                )
            else:
                st.success("All selected exports saved to dataset folder.")
