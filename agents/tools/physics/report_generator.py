"""
Report Generator agent tools: preview_report, add_report_section,
remove_report_section, reorder_report_section, edit_report_section, generate_report.

Wires the Report Builder page (Page 12) into the agentic schema.
Full parity with manual page: view (preview), add, remove, reorder, edit, generate.
"""

import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.io as pio


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "preview_report",
            "description": "Show the FULL compiled report in chat: HTML with figures, tables, text, sections—everything rendered. Use when user wants to SEE the report ('show report', 'complete compiled report', 'full report', 'how it looks', 'report with figures', 'what's in my report', 'report structure', 'list sections').",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Report title. Default: 'Turbulence Analysis Report - date'.",
                    },
                    "author": {
                        "type": "string",
                        "description": "Author name for metadata.",
                    },
                    "include_toc": {
                        "type": "boolean",
                        "description": "Include table of contents. Default true.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "add_report_section",
            "description": "Add a section to the scientific report. section_type: 'plot' (uses last figure) | 'text' | 'table'. For plot: pass caption= with a natural 2–4 sentence description of what the figure shows. For text: pass content= with the FULL actual explanation—never placeholders like '[Detailed explanation of...]'. Write complete prose. Add each figure once; use text sections to explain or reference figures by number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_type": {
                        "type": "string",
                        "enum": ["plot", "text", "table"],
                        "description": "Type: 'plot' uses figure_queue/last_figure; 'text' uses content (or last_generated_code if content empty); 'table' uses table_data or last_table_summary_rows.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Section title (e.g. 'Energy Spectra', 'Introduction', 'Results Table').",
                    },
                    "content": {
                        "type": "string",
                        "description": "Required for section_type='text'. Markdown. Ignored for plot/table.",
                    },
                    "table_data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Required for section_type='table'. List of dicts: each dict = row, keys = column names. E.g. [{\"Col1\": \"a\", \"Col2\": 1}, {\"Col1\": \"b\", \"Col2\": 2}].",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption below the section.",
                    },
                    "header_level": {
                        "type": "string",
                        "enum": ["H1", "H2", "H3", "H4", "Normal"],
                        "description": "Header level. Default H2 for text, Normal for plot/table.",
                    },
                },
                "required": ["section_type", "title"],
            },
        },
        {
            "name": "remove_report_section",
            "description": "Remove a section from the report by index (1-based). Use when user says 'delete section N', 'remove section N', 'remove the Nth section'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based section index. E.g. 1 = first section.",
                    },
                },
                "required": ["index"],
            },
        },
        {
            "name": "reorder_report_section",
            "description": "Move a section to a new position. Use when user says 'move section N up/down', 'reorder section N to position M', 'swap sections N and M'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_index": {
                        "type": "integer",
                        "description": "1-based index of section to move.",
                    },
                    "to_index": {
                        "type": "integer",
                        "description": "1-based target index (after move, section will be at this position).",
                    },
                },
                "required": ["from_index", "to_index"],
            },
        },
        {
            "name": "edit_report_section",
            "description": "Edit an existing section. Only provided fields are updated. Use when user says 'change section N title to X', 'edit section N', 'update caption of section N'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based section index to edit.",
                    },
                    "title": {
                        "type": "string",
                        "description": "New title. Omit to keep current.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (for text sections). Omit to keep current.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "New caption. Omit to keep current.",
                    },
                    "header_level": {
                        "type": "string",
                        "enum": ["H1", "H2", "H3", "H4", "Normal"],
                        "description": "New header level. Omit to keep current.",
                    },
                },
                "required": ["index"],
            },
        },
        {
            "name": "generate_report",
            "description": "Generate the scientific report as HTML or PDF file. Uses report_sections from session. Use when user says 'generate report', 'export report', 'create pdf report', 'save report', 'build report'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["html", "pdf"],
                        "description": "Output format: html (interactive) or pdf (static).",
                    },
                    "title": {
                        "type": "string",
                        "description": "Report title. Default: 'Turbulence Analysis Report - date'.",
                    },
                    "author": {
                        "type": "string",
                        "description": "Author name for metadata.",
                    },
                    "include_toc": {
                        "type": "boolean",
                        "description": "Include table of contents. Default true.",
                    },
                    "data_dir": {
                        "type": "string",
                        "description": "Directory to save report file. Uses session data_directory if not specified.",
                    },
                },
                "required": [],
            },
        },
    ]


def _serialize_section_for_export(sec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert agent report section to format expected by generate_html_report."""
    stype = sec.get("type")
    if stype == "text":
        content = sec.get("content", "")
        # Content must be string; pandas DataFrames raise in boolean context
        if content is None or not isinstance(content, str) or not content.strip():
            return None
        return {
            "title": sec.get("title", "Section"),
            "type": "text",
            "content": content,
            "caption": sec.get("caption", ""),
            "header_level": sec.get("header_level", "H2"),
        }
    if stype == "plot":
        fig = sec.get("figure")
        if fig is None:
            return None
        # generate_html_report expects 'content' to be the figure for plot type
        return {
            "title": sec.get("title", "Plot"),
            "type": "plot",
            "content": fig,
            "caption": sec.get("caption", ""),
            "header_level": sec.get("header_level", "Normal"),
        }
    if stype == "table":
        # Use explicit None checks; avoid boolean evaluation of DataFrame
        df = sec.get("dataframe")
        if df is None:
            df = sec.get("content")
        if df is None:
            return None
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return None
        return {
            "title": sec.get("title", "Table"),
            "type": "table",
            "content": df,
            "caption": sec.get("caption", ""),
            "header_level": sec.get("header_level", "Normal"),
        }
    return None


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> str:
    session_context = session_context or {}

    if name == "preview_report":
        from utils.report_builder import generate_html_report

        sections = session_context.get("report_sections") or []
        if not sections:
            return "Error: Report is empty. Add sections first (add_report_section), or capture figures from analysis pages."

        final_sections = []
        for sec in sections:
            conv = _serialize_section_for_export(sec)
            if conv is not None:
                final_sections.append(conv)

        if not final_sections:
            return "Error: No valid content in report sections. Plots may have been lost. Add sections again."

        report_title = args.get("title") or f"Turbulence Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
        author = args.get("author", "")
        include_toc = args.get("include_toc", True)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            # Use for_pdf=True so plots are static base64 images (not 3MB+ Plotly.js).
            # Streamlit's html component has a size limit; inline Plotly would exceed it.
            generate_html_report(
                report_title, final_sections, tmp_path,
                include_toc=include_toc, for_pdf=True, author=author
            )
            html_content = tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "message": "Report preview (compiled form). Review and confirm before generating PDF.",
            "artifact_type": "report_html",
            "artifact_content": html_content,
            "artifact_title": "Report Preview",
        }

    if name == "remove_report_section":
        sections = session_context.get("report_sections") or []
        idx = args.get("index")
        if idx is None:
            return "Error: Provide index (1-based)."
        if not (1 <= idx <= len(sections)):
            return f"Error: Index {idx} out of range. Report has {len(sections)} section(s)."
        removed = sections.pop(idx - 1)
        return f"Removed section {idx} ({removed.get('title', '?')}). Report now has {len(sections)} section(s)."

    if name == "reorder_report_section":
        sections = session_context.get("report_sections") or []
        from_idx = args.get("from_index")
        to_idx = args.get("to_index")
        if from_idx is None or to_idx is None:
            return "Error: Provide from_index and to_index (1-based)."
        n = len(sections)
        if not (1 <= from_idx <= n and 1 <= to_idx <= n):
            return f"Error: Indices must be 1 to {n}."
        if from_idx == to_idx:
            return "No change: section already at that position."
        item = sections.pop(from_idx - 1)
        sections.insert(to_idx - 1, item)
        return f"Moved section from position {from_idx} to {to_idx}. Report now has {n} section(s)."

    if name == "edit_report_section":
        sections = session_context.get("report_sections") or []
        idx = args.get("index")
        if idx is None:
            return "Error: Provide index (1-based)."
        if not (1 <= idx <= len(sections)):
            return f"Error: Index {idx} out of range. Report has {len(sections)} section(s)."
        sec = sections[idx - 1]
        if "title" in args and args["title"] is not None:
            sec["title"] = str(args["title"])
        if "content" in args and args["content"] is not None:
            sec["content"] = str(args["content"])
        if "caption" in args and args["caption"] is not None:
            sec["caption"] = str(args["caption"])
        if "header_level" in args and args["header_level"] is not None:
            sec["header_level"] = str(args["header_level"])
        return f"Updated section {idx} ({sec.get('title', '?')})."

    if name == "add_report_section":
        section_type = args.get("section_type", "plot")
        title = args.get("title", "Section")
        caption = args.get("caption", "")
        header_level = args.get("header_level", "H2" if section_type == "text" else "Normal")

        sections = session_context.setdefault("report_sections", [])

        # Deduplicate: reject exact duplicate title+type to prevent TOC/section doubling
        title_norm = (title or "").strip()
        for existing in sections:
            if (existing.get("title", "").strip() == title_norm and
                    existing.get("type") == section_type):
                n = len(sections)
                return (
                    f"Section '{title}' already added to report (skipped duplicate). "
                    f"Report now has {n} section(s). Proceed to next step or preview_report."
                )

        if section_type == "table":
            table_data = args.get("table_data")
            if table_data is None:
                # Use last table from any summary tool (get_*_summary sets last_table_summary_rows)
                table_data = session_context.get("last_table_summary_rows")
            if table_data is None:
                return "Error: For table sections, provide table_data as list of dicts (each dict = row, keys = column names). Or run a summary tool (get_real_isotropy_summary, get_flatness_summary, etc.) first, then add_report_section."
            # DataFrame: convert to list of dicts
            if isinstance(table_data, pd.DataFrame):
                table_data = table_data.to_dict("records")
            # Tool result dict: extract summary_rows, table_data, rows, or data
            elif isinstance(table_data, dict):
                extracted = (
                    table_data.get("summary_rows")
                    or table_data.get("table_data")
                    or table_data.get("rows")
                    or table_data.get("data")
                )
                if extracted is not None:
                    table_data = extracted
                elif table_data and not any(
                    k in table_data for k in ("status", "message", "artifact_type", "artifact_content")
                ):
                    # Single-row dict: wrap in list
                    table_data = [table_data]
            if not isinstance(table_data, list):
                return "Error: For table sections, provide table_data as list of dicts (each dict = row, keys = column names). Or run a summary tool first, then add_report_section."
            try:
                df = pd.DataFrame(table_data)
            except Exception as e:
                return f"Error: Invalid table_data: {e}"
            sections.append({
                "title": title,
                "type": "table",
                "dataframe": df,
                "content": df,
                "caption": caption,
                "header_level": header_level,
                "source_page": "Autonomous Lab",
            })
            return f"Added table section '{title}' to report. Report now has {len(sections)} section(s)."

        if section_type == "plot":
            queue = session_context.get("figure_queue") or []
            fig = queue.pop(0) if queue else session_context.get("last_figure")
            if fig is None:
                json_str = session_context.get("last_figure_json")
                if json_str:
                    try:
                        fig = pio.from_json(json_str)
                    except Exception:
                        pass
            if fig is None:
                return "Error: No figure available to add. Produce a plot first (e.g. plot_spectrum, plot_pdf), then add it to the report."
            sections.append({
                "title": title,
                "type": "plot",
                "figure": fig,
                "caption": caption,
                "header_level": header_level,
                "source_page": "Autonomous Lab",
            })
            n_plots = sum(1 for s in sections if s.get("type") == "plot")
            msg = f"Added plot section '{title}' to report. Report now has {len(sections)} section(s)."
            if n_plots > 2:
                msg += " REMINDER: Use section_type='text' to reference figures (e.g. 'Figure 1 shows...'). Do NOT add the same figure again."
            return msg

        if section_type == "text":
            content = args.get("content", "")
            used_stored_code = False
            if content is None or not isinstance(content, str) or not content.strip():
                # Fallback: use last generated code (generate_code stores in session_context)
                content = session_context.pop("last_generated_code", None)
                used_stored_code = bool(content)
            if content is None or not isinstance(content, str) or not content.strip():
                return "Error: For text sections, provide 'content' (markdown string). Or run generate_code first—its output is stored for add_report_section."
            # Wrap code in markdown code block to prevent underscores/subscripts from being parsed as italics
            if used_stored_code and content.strip().startswith(("import ", "from ", "def ", "class ", "#")):
                content = "```python\n" + content.strip() + "\n```"
            sections.append({
                "title": title,
                "type": "text",
                "content": content,
                "caption": caption,
                "header_level": header_level,
                "source_page": "Autonomous Lab",
            })
            return f"Added text section '{title}' to report. Report now has {len(sections)} section(s)."

        return f"Error: Unknown section_type '{section_type}'. Use 'plot', 'text', or 'table'."

    if name == "generate_report":
        from utils.report_builder import generate_html_report, generate_pdf_report

        sections = session_context.get("report_sections") or []
        if not sections:
            return "Error: Report has no sections. Add sections first (add_report_section with plot or text), or capture figures from analysis pages."

        # Convert to export format (handle figure vs content for plot)
        final_sections = []
        for sec in sections:
            conv = _serialize_section_for_export(sec)
            if conv is not None:
                final_sections.append(conv)

        if not final_sections:
            return "Error: No valid content in report sections. Plots may have been lost. Add sections again."

        data_dir = args.get("data_dir") or session_context.get("data_directory") or session_context.get("data_directories", [None])[0]
        if not data_dir:
            return "Error: No data directory. Load data first or specify data_dir."

        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = (project_root / str(data_dir).lstrip("/")).resolve()
        if not data_path.exists():
            alt = project_root / "examples" / str(data_dir).lstrip("/")
            if alt.exists():
                data_path = alt
            else:
                return f"Error: Data directory not found: {data_dir}"

        report_title = args.get("title") or f"Turbulence Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
        author = args.get("author", "")
        include_toc = args.get("include_toc", True)
        fmt = args.get("format", "html")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fmt == "pdf":
            output_file = data_path / f"report_{timestamp}.pdf"
            report_path = generate_pdf_report(
                report_title, final_sections, output_file,
                include_toc=include_toc, author=author
            )
        else:
            output_file = data_path / f"report_{timestamp}.html"
            report_path = generate_html_report(
                report_title, final_sections, output_file,
                include_toc=include_toc, author=author
            )

        return f"Report generated: {Path(report_path).name} at {report_path}"

    return f"Error: Unknown report tool '{name}'"
