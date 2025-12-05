"""
Report Builder Page
One-click export of selected panels, tables, and figures into PDF/HTML reports
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from utils.report_builder import generate_html_report, generate_pdf_report
st.set_page_config(page_icon="⚫")

def main():
    inject_theme_css()
    
    st.title("📄 Report Builder")
    st.markdown("Select panels, tables, and figures to include in your PDF/HTML report")
    
    # Get data directories
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return
    
    data_dir = Path(data_dirs[0])
    
    # Initialize report sections in session state
    if 'report_sections' not in st.session_state:
        st.session_state.report_sections = []
    
    # Show notification if sections were just added
    num_sections = len(st.session_state.report_sections)
    if 'report_section_count' not in st.session_state:
        st.session_state.report_section_count = 0
    
    if num_sections > st.session_state.report_section_count:
        new_count = num_sections - st.session_state.report_section_count
        st.success(f"✅ {new_count} new section(s) added to report!")
        st.session_state.report_section_count = num_sections
    
    st.sidebar.header("📋 Report Configuration")
    
    # Report metadata
    report_title = st.sidebar.text_input(
        "Report Title",
        value=f"Turbulence Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
        help="Title for your report"
    )
    
    author = st.sidebar.text_input(
        "Author",
        value="",
        help="Your name or organization"
    )
    
    include_toc = st.sidebar.checkbox("Include Table of Contents", value=True)
    
    st.sidebar.markdown("---")
    
    # Section management
    st.sidebar.subheader("Report Sections")
    st.sidebar.caption(f"Current sections: {len(st.session_state.report_sections)}")
    
    if st.sidebar.button("➕ Add Section", width='stretch'):
        st.session_state.report_sections.append({
            "title": f"Section {len(st.session_state.report_sections) + 1}",
            "type": "text",
            "content": "",
            "source_page": "Manual"
        })
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Report Sections")
        
        # Show count of sections
        num_sections = len(st.session_state.report_sections)
        if num_sections > 0:
            st.caption(f"Total sections: {num_sections}")
            # Debug: show section types
            section_types = [s.get('type', 'unknown') for s in st.session_state.report_sections]
            st.caption(f"Types: {', '.join(section_types)}")
        
        # Refresh button to reload sections
        if st.button("🔄 Refresh", help="Refresh to see newly captured content", key="refresh_sections"):
            st.rerun()
        
        if not st.session_state.report_sections:
            st.info("👈 Use the sidebar to add sections to your report, or navigate to other pages to capture figures and tables.")
        else:
            for idx, section in enumerate(st.session_state.report_sections):
                # Expand newly added sections by default
                is_new = idx >= (st.session_state.get('last_seen_section_count', 0))
                with st.expander(f"Section {idx+1}: {section['title']}", expanded=is_new):
                    col_title, col_type = st.columns([3, 1])
                    
                    with col_title:
                        section['title'] = st.text_input(
                            "Section Title",
                            value=section['title'],
                            key=f"section_title_{idx}"
                        )
                    
                    with col_type:
                        section['type'] = st.selectbox(
                            "Type",
                            ["text", "plot", "table", "image"],
                            index=["text", "plot", "table", "image"].index(section.get("type", "text")),
                            key=f"section_type_{idx}"
                        )
                    
                    if section['type'] == "text":
                        section['content'] = st.text_area(
                            "Content",
                            value=section.get('content', ''),
                            height=150,
                            key=f"section_content_{idx}"
                        )
                    elif section['type'] == "plot":
                        if 'figure' in section and section['figure'] is not None:
                            st.plotly_chart(section['figure'], width='stretch', key=f"plot_chart_{idx}")
                        else:
                            st.warning("⚠️ Figure not available. Please recapture from the source page.")
                    elif section['type'] == "table":
                        if 'dataframe' in section and section['dataframe'] is not None:
                            st.dataframe(section['dataframe'], width='stretch')
                        else:
                            st.warning("⚠️ Table not available. Please recapture from the source page.")
                    
                    st.caption(f"Source: {section.get('source_page', 'Manual')}")
                    
                    if st.button("🗑️ Remove", key=f"remove_{idx}"):
                        st.session_state.report_sections.pop(idx)
                        st.rerun()
    
    with col2:
        st.subheader("⚙️ Generate Report")
        
        report_format = st.radio(
            "Report Format",
            ["HTML (Interactive)", "PDF (Print-ready)"],
            help="HTML includes interactive Plotly figures. PDF is print-ready but static."
        )
        
        st.markdown("---")
        
        if st.button("📄 Generate Report", type="primary", width='stretch'):
            if not st.session_state.report_sections:
                st.error("Please add at least one section to the report.")
                return
            
            # Prepare sections for report
            report_sections = []
            for section in st.session_state.report_sections:
                if section['type'] == 'text' and section.get('content'):
                    report_sections.append({
                        'title': section['title'],
                        'type': 'text',
                        'content': section['content']
                    })
                elif section['type'] == 'plot' and 'figure' in section:
                    report_sections.append({
                        'title': section['title'],
                        'type': 'plot',
                        'content': section['figure']
                    })
                elif section['type'] == 'table' and 'dataframe' in section:
                    report_sections.append({
                        'title': section['title'],
                        'type': 'table',
                        'content': section['dataframe']
                    })
            
            if not report_sections:
                st.error("No valid content found in sections. Please add content.")
                return
            
            # Generate report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if report_format == "PDF (Print-ready)":
                output_file = data_dir / f"report_{timestamp}.pdf"
                try:
                    report_path = generate_pdf_report(
                        report_title,
                        report_sections,
                        output_file,
                        include_toc
                    )
                    file_format = "PDF" if report_path.endswith('.pdf') else "HTML"
                    st.success(f"✅ {file_format} report generated: {Path(report_path).name}")
                    
                    # Provide download button
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "📥 Download Report",
                            f.read(),
                            file_name=Path(report_path).name,
                            mime="application/pdf" if report_path.endswith('.pdf') else "text/html"
                        )
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
            else:
                output_file = data_dir / f"report_{timestamp}.html"
                report_path = generate_html_report(
                    report_title,
                    report_sections,
                    output_file,
                    include_toc
                )
                st.success(f"✅ Report generated: {Path(report_path).name}")
                
                # Provide download button
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📥 Download Report",
                        f.read(),
                        file_name=Path(report_path).name,
                        mime="text/html"
                    )
    
    st.markdown("---")
    st.subheader("📖 How to Use")
    
    st.markdown("""
    **Step 1: Add Sections**
    - Click "➕ Add Section" in the sidebar
    - Edit section title and type
    - Add content (text, or capture from other pages)
    
    **Step 2: Capture Content from Pages**
    - Navigate to analysis pages (Energy Spectra, Flatness, etc.)
    - Use "📋 Capture for Report" buttons to add figures/tables
    - Captured content appears in your report sections
    
    **Step 3: Generate Report**
    - Choose format (HTML or PDF)
    - Click "📄 Generate Report"
    - Download the generated file
    
    **Tips:**
    - HTML reports include interactive Plotly figures
    - PDF reports are static but print-ready
    - Install `weasyprint` for PDF generation: `pip install weasyprint`
    """)
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Sections", width='stretch'):
            st.session_state.report_sections = []
            st.rerun()
    
    with col2:
        if st.button("📋 Add Text Section", width='stretch'):
            st.session_state.report_sections.append({
                "title": "New Text Section",
                "type": "text",
                "content": "",
                "source_page": "Manual"
            })
            st.rerun()
    
    with col3:
        if st.button("📊 Add Table Placeholder", width='stretch'):
            st.session_state.report_sections.append({
                "title": "New Table Section",
                "type": "table",
                "content": None,
                "source_page": "Manual"
            })
            st.rerun()


if __name__ == "__main__":
    main()

