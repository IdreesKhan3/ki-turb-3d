"""
Report Builder Utility
Generates PDF and HTML reports from selected panels, tables, and figures
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional, Any
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import base64
import io
import tempfile


def _convert_plotly_to_image_base64(fig: go.Figure) -> str:
    """
    Convert Plotly figure to base64-encoded PNG image for PDF embedding
    
    Returns:
        Base64-encoded image string (data URI format)
    """
    try:
        # Try using kaleido (recommended)
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e1:
        # Fallback: try write_image
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.write_image(tmp.name, width=1200, height=800, scale=2)
                with open(tmp.name, 'rb') as f:
                    img_bytes = f.read()
                Path(tmp.name).unlink()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return f"data:image/png;base64,{img_base64}"
        except Exception as e2:
            # Check if it's a Chrome/kaleido error
            error_msg = str(e1) + " " + str(e2)
            if "Chrome" in error_msg or "kaleido" in error_msg.lower():
                # Try to install Chrome automatically
                try:
                    import kaleido
                    kaleido.get_chrome_sync()
                    # Retry after installing Chrome
                    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    return f"data:image/png;base64,{img_base64}"
                except Exception:
                    pass
            # Last resort: return empty string
            return ""


def generate_html_report(
    title: str,
    sections: List[Dict[str, Any]],
    output_path: Path,
    include_toc: bool = True,
    for_pdf: bool = False
) -> str:
    """
    Generate HTML report from selected sections
    
    Args:
        title: Report title
        sections: List of section dicts with keys: 'title', 'type', 'content'
                  type can be: 'plot', 'table', 'text', 'image'
        output_path: Path to save HTML file
        include_toc: Whether to include table of contents
        
    Returns:
        Path to generated HTML file
    """
    html_parts = []
    
    # HTML header
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""" + title + """</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }
        h2 { color: #2ca02c; margin-top: 30px; }
        h3 { color: #9467bd; margin-top: 20px; }
        .section { margin: 30px 0; }
        .plot-container { margin: 20px 0; text-align: center; }
        .table-container { margin: 20px 0; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .toc { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #1f77b4; margin: 20px 0; }
        .toc ul { list-style-type: none; padding-left: 0; }
        .toc li { margin: 5px 0; }
        .toc a { text-decoration: none; color: #1f77b4; }
        .toc a:hover { text-decoration: underline; }
        .metadata { color: #666; font-size: 0.9em; margin: 10px 0; }
    </style>
    <!-- Plotly.js will be embedded inline in each plot for offline support -->
</head>
<body>
""")
    
    # Title and metadata
    html_parts.append(f"<h1>{title}</h1>")
    html_parts.append(f'<div class="metadata">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>')
    
    # Table of contents
    if include_toc and sections:
        html_parts.append('<div class="toc"><h3>Table of Contents</h3><ul>')
        for i, section in enumerate(sections, 1):
            section_id = f"section_{i}"
            html_parts.append(f'<li><a href="#{section_id}">{i}. {section.get("title", "Section")}</a></li>')
        html_parts.append('</ul></div>')
    
    # Sections
    for i, section in enumerate(sections, 1):
        section_id = f"section_{i}"
        section_title = section.get("title", f"Section {i}")
        section_type = section.get("type", "text")
        content = section.get("content")
        
        html_parts.append(f'<div class="section" id="{section_id}">')
        html_parts.append(f"<h2>{section_title}</h2>")
        
        if section_type == "plot" and isinstance(content, go.Figure):
            if for_pdf:
                # For PDF: convert Plotly figure to static image
                img_data = _convert_plotly_to_image_base64(content)
                if img_data:
                    html_parts.append(f'<div class="plot-container"><img src="{img_data}" alt="{section_title}" style="max-width: 100%;"></div>')
                else:
                    html_parts.append(f'<div class="plot-container"><p style="color: red;">⚠️ Could not render plot. Install kaleido: pip install kaleido</p></div>')
            else:
                # For HTML: embed interactive Plotly figure
                plot_html = content.to_html(include_plotlyjs='inline', div_id=f"plot_{i}")
                html_parts.append(f'<div class="plot-container">{plot_html}</div>')
        
        elif section_type == "table" and isinstance(content, pd.DataFrame):
            # Convert DataFrame to HTML table
            html_parts.append('<div class="table-container">')
            html_parts.append(content.to_html(classes='dataframe', table_id=f"table_{i}", escape=False))
            html_parts.append('</div>')
        
        elif section_type == "text":
            # Plain text/markdown
            html_parts.append(f'<div>{content}</div>')
        
        elif section_type == "image" and isinstance(content, (str, Path)):
            # Image file
            img_path = Path(content)
            if img_path.exists():
                html_parts.append(f'<div class="plot-container"><img src="{img_path}" alt="{section_title}" style="max-width: 100%;"></div>')
        
        html_parts.append('</div>')
    
    # Footer
    html_parts.append("""
</body>
</html>
""")
    
    # Write HTML file
    html_content = "\n".join(html_parts)
    output_path.write_text(html_content, encoding="utf-8")
    
    return str(output_path)


def generate_pdf_report(
    title: str,
    sections: List[Dict[str, Any]],
    output_path: Path,
    include_toc: bool = True
) -> str:
    """
    Generate PDF report from selected sections
    
    Note: Requires weasyprint or reportlab. Falls back to HTML if PDF generation fails.
    
    Args:
        title: Report title
        sections: List of section dicts
        output_path: Path to save PDF file
        include_toc: Whether to include table of contents
        
    Returns:
        Path to generated file (PDF or HTML fallback)
    """
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        # Generate HTML first (with static images for PDF)
        html_path = output_path.with_suffix('.html')
        html_file = generate_html_report(title, sections, html_path, include_toc, for_pdf=True)
        
        # Convert HTML to PDF
        font_config = FontConfiguration()
        html_doc = HTML(filename=html_file)
        css = CSS(string='@page { size: A4; margin: 2cm; }')
        html_doc.write_pdf(output_path, stylesheets=[css], font_config=font_config)
        
        return str(output_path)
    
    except ImportError:
        # Fallback: generate HTML and suggest weasyprint installation
        html_path = output_path.with_suffix('.html')
        html_file = generate_html_report(title, sections, html_path, include_toc)
        st.info(
            "ℹ️ PDF generation requires 'weasyprint' (not installed). "
            "HTML report generated successfully instead.\n\n"
            "To generate PDFs in the future, install: `pip install weasyprint`\n"
            "Or convert the HTML file to PDF manually using your browser's print function."
        )
        return html_file
    
    except Exception as e:
        # Fallback to HTML on any error
        html_path = output_path.with_suffix('.html')
        html_file = generate_html_report(title, sections, html_path, include_toc)
        st.warning(f"PDF generation failed: {e}. Generated HTML report instead.")
        return html_file


def capture_plotly_figure(fig: go.Figure) -> Dict[str, Any]:
    """Capture a Plotly figure for report inclusion"""
    return {
        "type": "plot",
        "content": fig,
        "title": fig.layout.title.text if fig.layout.title else "Plot"
    }


def capture_dataframe(df: pd.DataFrame, title: str = "Table") -> Dict[str, Any]:
    """Capture a DataFrame for report inclusion"""
    return {
        "type": "table",
        "content": df,
        "title": title
    }


def capture_text(text: str, title: str = "Text") -> Dict[str, Any]:
    """Capture text content for report inclusion"""
    return {
        "type": "text",
        "content": text,
        "title": title
    }


def add_figure_to_report(fig: go.Figure, title: str, source_page: str = "Unknown"):
    """
    Add a Plotly figure to the report builder
    
    Args:
        fig: Plotly figure to capture
        title: Title for the section
        source_page: Name of the page where figure was captured
    """
    if 'report_sections' not in st.session_state:
        st.session_state.report_sections = []
    
    # Store the figure directly - Streamlit can handle Plotly figures in session state
    try:
        st.session_state.report_sections.append({
            "title": title,
            "type": "plot",
            "figure": fig,
            "source_page": source_page
        })
    except Exception as e:
        st.error(f"Error adding figure to report: {e}")
        raise


def add_table_to_report(df: pd.DataFrame, title: str, source_page: str = "Unknown"):
    """
    Add a DataFrame/table to the report builder
    
    Args:
        df: DataFrame to capture
        title: Title for the section
        source_page: Name of the page where table was captured
    """
    if 'report_sections' not in st.session_state:
        st.session_state.report_sections = []
    
    st.session_state.report_sections.append({
        "title": title,
        "type": "table",
        "dataframe": df,
        "source_page": source_page
    })


def add_text_to_report(text: str, title: str, source_page: str = "Unknown"):
    """
    Add text content to the report builder
    
    Args:
        text: Text content to capture
        title: Title for the section
        source_page: Name of the page where text was captured
    """
    if 'report_sections' not in st.session_state:
        st.session_state.report_sections = []
    
    st.session_state.report_sections.append({
        "title": title,
        "type": "text",
        "content": text,
        "source_page": source_page
    })


def capture_button(fig: Optional[go.Figure] = None, df: Optional[pd.DataFrame] = None, 
                   title: Optional[str] = None, source_page: str = "Unknown"):
    """
    Display a button to capture current figure or table for report
    
    Args:
        fig: Plotly figure to capture (optional)
        df: DataFrame to capture (optional)
        title: Custom title (optional, will auto-generate if not provided)
        source_page: Name of the current page
        
    Returns:
        True if button was clicked and content was captured
    """
    if fig is None and df is None:
        return False
    
    if fig is not None:
        default_title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "Plot"
        capture_title = title or default_title
        
        # Use a more unique key that includes the title to avoid conflicts
        button_key = f"capture_fig_{source_page}_{hash(str(capture_title))}"
        
        if st.button("📋 Capture Figure for Report", key=button_key):
            try:
                add_figure_to_report(fig, capture_title, source_page)
                st.success(f"✅ Captured: {capture_title}")
                st.toast(f"Added to report: {capture_title}", icon="✅")
                # Store a flag to show the section was just added
                st.session_state[f"_just_captured_{button_key}"] = True
                # Force a rerun to update the Report Builder page if it's open
                return True
            except Exception as e:
                st.error(f"Failed to capture figure: {e}")
                return False
    
    if df is not None:
        capture_title = title or "Data Table"
        
        # Use a more unique key
        button_key = f"capture_table_{source_page}_{hash(str(capture_title))}"
        
        if st.button("📋 Capture Table for Report", key=button_key):
            try:
                add_table_to_report(df, capture_title, source_page)
                st.success(f"✅ Captured: {capture_title}")
                st.toast(f"Added to report: {capture_title}", icon="✅")
                # Store a flag to show the section was just added
                st.session_state[f"_just_captured_{button_key}"] = True
                return True
            except Exception as e:
                st.error(f"Failed to capture table: {e}")
                return False
    
    return False

