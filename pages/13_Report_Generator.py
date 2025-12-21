"""
Report Builder Page
High-quality scientific report generation with structural controls and LaTeX support
"""

import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from utils.report_builder import generate_html_report, generate_pdf_report
st.set_page_config(page_icon="📄", layout="wide")

# ==========================================================
# Helper Functions
# ==========================================================
def move_section(index, direction):
    """Moves a section up (-1) or down (+1) in the list."""
    sections = st.session_state.report_sections
    new_index = index + direction
    if 0 <= new_index < len(sections):
        sections[index], sections[new_index] = sections[new_index], sections[index]

def delete_section(index):
    """Removes a section from the report."""
    st.session_state.report_sections.pop(index)

def insert_section(index, type="text"):
    """Inserts a new section at a specific index."""
    new_section = {
        "title": "New Section",
        "type": type,
        "content": "" if type == "text" else None,
        "caption": "",
        "source_page": "Manual",
        "header_level": "H2"
    }
    
    if type == "manual_table":
        new_section["content"] = pd.DataFrame(
            {"Column 1": ["", ""], "Column 2": ["", ""]}
        )
        new_section["type"] = "table"
    
    st.session_state.report_sections.insert(index + 1, new_section)

def save_config(data_dir):
    """Saves the current report structure to JSON."""
    config_path = data_dir / "report_config.json"

    serializable_sections = []
    for sec in st.session_state.report_sections:
        item = sec.copy()

        # Persist tables
        if item.get("type") == "table":
            df = item.get("dataframe")
            if df is None:
                df = item.get("content")
            if isinstance(df, pd.DataFrame):
                item["table_json"] = df.to_json(orient="split")
            # remove non-serializable objects
            item["dataframe"] = None
            if isinstance(item.get("content"), pd.DataFrame):
                item["content"] = None

        # Keep current behavior for plots (do not persist)
        if item.get("type") == "plot":
            item["figure"] = None

        serializable_sections.append(item)

    with open(config_path, "w") as f:
        json.dump(serializable_sections, f, indent=4)

    st.toast(f"💾 Report configuration saved to {config_path.name}")

def load_config(data_dir):
    """Loads a report structure from JSON."""
    config_path = data_dir / "report_config.json"
    if not config_path.exists():
        st.error("No saved configuration found.")
        return

    with open(config_path, "r") as f:
        data = json.load(f)

    # Backward compatibility + table restore
    for section in data:
        if "header_level" not in section:
            section["header_level"] = "H2" if section.get("type") == "text" else "Normal"

        if section.get("type") == "table" and section.get("table_json"):
            try:
                df = pd.read_json(section["table_json"], orient="split")
                section["dataframe"] = df
                section["content"] = df
            except Exception:
                section["dataframe"] = None

    st.session_state.report_sections = data
    st.toast("📂 Configuration loaded (Note: You may need to recapture plots)")

# ==========================================================
# Main
# ==========================================================
def main():
    inject_theme_css()
    
    st.title("📄 Scientific Report Builder")
    st.info("Tip: Use the 'Insert' buttons between blocks to arrange figures and tables precisely within your text flow.")
    
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
        st.success(f"{new_count} new section(s) added to report!")
        st.session_state.report_section_count = num_sections
    
    st.sidebar.header("Report Metadata")
    
    report_title = st.sidebar.text_input(
        "Title",
        value=f"Turbulence Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
        help="Title for your report"
    )
    
    author = st.sidebar.text_input(
        "Author",
        value="",
        help="Your name or organization"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("⚙️ Actions")
    
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("💾 Save Config"):
            save_config(data_dir)
    with col_s2:
        if st.button("📂 Load Config"):
            load_config(data_dir)
            st.rerun()
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Report Sections")
    st.sidebar.caption(f"Current sections: {len(st.session_state.report_sections)}")
    
    include_toc = st.sidebar.checkbox("Include Table of Contents", value=True)
    
    # =========================
    # Report Section Management
    # =========================
    if not st.session_state.report_sections:
        st.markdown("### Start your report")
        c1, c2 = st.columns(2)
        if c1.button("➕ Add Title/Text", key="start_text", width='stretch'):
            st.session_state.report_sections.append({
                "title": "Introduction",
                "type": "text",
                "content": "",
                "caption": "",
                "header_level": "H1",
                "source_page": "Manual"
            })
            st.rerun()
        if c2.button("➕ Create Table", key="start_table", width='stretch'):
            insert_section(-1, "manual_table")
            st.rerun()
    
    for idx, section in enumerate(st.session_state.report_sections):
        with st.container(border=True):
            c_drag, c_title, c_type, c_del = st.columns([1, 6, 2, 1])
            
            with c_drag:
                if idx > 0 and st.button("⬆", key=f"u_{idx}"):
                    move_section(idx, -1)
                    st.rerun()
                if idx < len(st.session_state.report_sections) - 1 and st.button("⬇", key=f"d_{idx}"):
                    move_section(idx, 1)
                    st.rerun()
            
            with c_title:
                if section['type'] == 'text':
                    col_h, col_t = st.columns([2, 5])
                    header_levels = ["H1", "H2", "H3", "H4", "Normal"]
                    current_level = section.get('header_level', 'H2')
                    level_index = header_levels.index(current_level) if current_level in header_levels else 1
                    section['header_level'] = col_h.selectbox(
                        "Level", header_levels,
                        key=f"h_lvl_{idx}",
                        index=level_index,
                        label_visibility="collapsed"
                    )
                    section['title'] = col_t.text_input(
                        "Header Text", value=section['title'], key=f"t_{idx}", label_visibility="collapsed"
                    )
                else:
                    section['title'] = st.text_input(
                        "Caption Title", value=section['title'], key=f"t_{idx}", label_visibility="collapsed"
                    )
            
            with c_type:
                st.caption(f"Type: {section['type'].upper()}")
            
            with c_del:
                if st.button("Delete", key=f"del_{idx}"):
                    delete_section(idx)
                    st.rerun()
            
            if section['type'] == "text":
                section['content'] = st.text_area(
                    "Markdown Content", value=section.get('content', ''), height=150, key=f"txt_{idx}"
                )
                with st.expander("👁️ Preview"):
                    h_map = {"H1": "# ", "H2": "## ", "H3": "### ", "H4": "#### ", "Normal": ""}
                    prefix = h_map.get(section.get('header_level', 'Normal'), "")
                    if prefix:
                        st.markdown(f"{prefix}{section['title']}")
                    st.markdown(section['content'])
            
            elif section['type'] == "table":
                st.caption("Edit Table Data:")
                if 'dataframe' in section and section['dataframe'] is not None:
                    section['dataframe'] = st.data_editor(
                        section['dataframe'],
                        key=f"tbl_ed_{idx}",
                        num_rows="dynamic",
                        use_container_width=True,
                        column_config={
                            "_index": st.column_config.NumberColumn("Row", help="Row number")
                        }
                    )
                    # Add column button below the table
                    if st.button("➕ Add Column", key=f"add_col_{idx}"):
                        new_col_name = f"Column {len(section['dataframe'].columns) + 1}"
                        section['dataframe'][new_col_name] = ""
                        st.rerun()
                elif 'content' in section and isinstance(section['content'], pd.DataFrame):
                    section['content'] = st.data_editor(
                        section['content'],
                        key=f"tbl_man_{idx}",
                        num_rows="dynamic",
                        use_container_width=True,
                        column_config={
                            "_index": st.column_config.NumberColumn("Row", help="Row number")
                        }
                    )
                    section['dataframe'] = section['content']
                    # Add column button below the table
                    if st.button("➕ Add Column", key=f"add_col_man_{idx}"):
                        new_col_name = f"Column {len(section['content'].columns) + 1}"
                        section['content'][new_col_name] = ""
                        st.rerun()
                else:
                    st.error("Table data lost.")
                
                section['caption'] = st.text_input("Caption", value=section.get('caption', ''), key=f"tc_{idx}")
            
            elif section['type'] == "plot":
                if section.get('figure'):
                    st.plotly_chart(section['figure'], width='stretch', key=f"plt_{idx}")
                else:
                    st.warning("Plot data missing or placeholder.")
                section['caption'] = st.text_input("Caption", value=section.get('caption', ''), key=f"pc_{idx}")
        
        st.markdown("<div style='text-align: center; margin: 10px 0; opacity: 0.5;'>⬇️ <i>Insert Here</i> ⬇️</div>", unsafe_allow_html=True)
        
        c_i1, c_i2, c_i3 = st.columns(3)
        if c_i1.button("📄 Text", key=f"add_txt_{idx}"):
            insert_section(idx, "text")
            st.rerun()
        if c_i2.button("Manual Table", key=f"add_tbl_{idx}"):
            insert_section(idx, "manual_table")
            st.rerun()
        if c_i3.button("Placeholder Plot", key=f"add_plt_{idx}"):
            st.info("To add specific plots, go to Analysis pages and click 'Capture'. They will appear at the end, then use ⬆ to move them here.")
    
    st.markdown("---")
    
    # =========================
    # Export Report
    # =========================
    col1, col2 = st.columns([3, 1])
    
    with col1:
    
        st.subheader("Export Report")
        
        export_fmt = st.radio("Format", ["PDF (Static)", "HTML (Interactive)"], horizontal=True)
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Generate File", type="primary", width='stretch'):
            if not st.session_state.report_sections:
                st.error("Please add at least one section to the report.")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            final_sections = []
            for s in st.session_state.report_sections:
                if s['type'] == 'text' and s.get('content'):
                    final_sections.append({
                        'title': s['title'],
                        'type': 'text',
                        'content': s['content'],
                        'caption': s.get('caption', ''),
                        'header_level': s.get('header_level', 'H2')
                    })
                elif s['type'] == 'plot' and 'figure' in s:
                    final_sections.append({
                        'title': s['title'],
                        'type': 'plot',
                        'content': s['figure'],
                        'caption': s.get('caption', ''),
                        'header_level': s.get('header_level', 'Normal')
                    })
                elif s['type'] == 'table':
                    df = s.get('dataframe')
                    if df is None:
                        df = s.get('content')
                    if df is not None:
                        final_sections.append({
                            'title': s['title'],
                            'type': 'table',
                            'content': df,
                            'caption': s.get('caption', ''),
                            'header_level': s.get('header_level', 'Normal')
                        })
            
            if not final_sections:
                st.error("No valid content found in sections. Please add content.")
                return
            
            with st.spinner("Rendering scientific report..."):
                if export_fmt.startswith("HTML"):
                    output_file = data_dir / f"report_{timestamp}.html"
                    report_path = generate_html_report(
                        report_title,
                        final_sections,
                        output_file,
                        include_toc,
                        author=author
                    )
                else:
                    output_file = data_dir / f"report_{timestamp}.pdf"
                    report_path = generate_pdf_report(
                        report_title,
                        final_sections,
                        output_file,
                        include_toc,
                        author=author
                    )
                
                st.success(f"Report Generated: {Path(report_path).name}")
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📥 Download Report",
                        f.read(),
                        file_name=Path(report_path).name,
                        mime="application/pdf" if report_path.endswith('.pdf') else "text/html"
                    )
    


if __name__ == "__main__":
    main()

