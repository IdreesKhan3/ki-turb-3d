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
    "JPG/JPEG (raster)": "jpg",
    "WEBP (raster)": "webp",
    "HTML (interactive)": "html",
}

# MIME type mapping for download buttons
_MIME_TYPES = {
    "png": "image/png",
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "html": "text/html",
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
            
            # export-only copy (do not alter Streamlit rendering)
            try:
                fig_export = fig.full_copy()
            except Exception:
                fig_export = fig  # fallback if full_copy not available

            # avoid PDF/SVG/PNG zero-line artifact at x=0/y=0
            fig_export.update_xaxes(zeroline=False)
            fig_export.update_yaxes(zeroline=False)
            
            # For PDF export, explicitly preserve margins to prevent Kaleido from adding extra padding

            if "PDF (vector)" in fmts:
                margin = fig_export.layout.margin
                if margin is not None:
                    # Get margin values safely
                    l_val = getattr(margin, 'l', 50) if margin else 50
                    r_val = getattr(margin, 'r', 20) if margin else 20
                    t_val = getattr(margin, 't', 30) if margin else 30
                    b_val = getattr(margin, 'b', 50) if margin else 50
                    # Explicitly set margins to prevent PDF padding issues
                    fig_export.update_layout(margin=dict(l=l_val, r=r_val, t=t_val, b=b_val))

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
                        fig_export.write_image(str(out), scale=scale, **kwargs)
                except Exception as e:
                    errors.append((out.name, str(e)))
            
            if errors:
                st.error(
                    "Some exports failed:\n\n"
                    + "\n".join([f"- {n}: {msg}" for n, msg in errors])
                    + "\n\nEnsure kaleido is installed: pip install -U kaleido"
                )
            else:
                st.success("All selected exports saved to dataset folder.")
        
        # Download buttons for each exported file
        if fmts:
            existing_files = []
            for f_label in fmts:
                ext = _EXPORT_FORMATS[f_label]
                file_path = out_dir / f"{base_name}.{ext}"
                if file_path.exists():
                    existing_files.append((ext, file_path))
            
            if existing_files:
                st.markdown("**Download exported files:**")
                cols = st.columns(min(len(existing_files), 4))
                for idx, (ext, file_path) in enumerate(existing_files):
                    with cols[idx % len(cols)]:
                        try:
                            mime_type = _MIME_TYPES.get(ext, "application/octet-stream")
                            st.download_button(
                                "Download",
                                data=file_path.read_bytes(),
                                file_name=f"{base_name}.{ext}",
                                mime=mime_type,
                                key=f"{base_name}_download_{ext}_{idx}"
                            )
                            st.caption(f".{ext}")
                        except Exception as e:
                            st.warning(f"Could not read {ext} file: {e}")
