"""
Generation tools: LLM-based content and code generation.

Uses the configured LLM provider (Ollama/Gemini) for:
- generate_content: papers, abstracts, patents, manuals, reports (LaTeX, markdown, raw)
- generate_code: Python, shell, JavaScript, etc.
- compile_latex: compile LaTeX .tex file to PDF
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._shared import resolve_path


GENERATION_TOOL_NAMES = frozenset({"generate_content", "generate_code", "compile_latex"})


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for generation tools."""
    return [
        {
            "name": "generate_content",
            "description": "Generate long-form text using the LLM. Use for papers, abstracts, patents, manuals, reports, book chapters. Supports LaTeX, markdown, or raw text output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content_type": {
                        "type": "string",
                        "description": "Type: paper, abstract, patent, manual, report, book_chapter, thesis_section, literature_review, cover_letter",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Main topic or subject of the content",
                    },
                    "outline": {
                        "type": "string",
                        "description": "Optional outline or structure (e.g. 'Introduction, Methods, Results, Discussion')",
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: raw, markdown, latex (default: markdown)",
                    },
                    "constraints": {
                        "type": "string",
                        "description": "Optional constraints (word limit, citation style, target audience)",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context (prior research, file content, user notes)",
                    },
                },
            },
        },
        {
            "name": "generate_code",
            "description": "Generate code using the LLM. Use for scripts, functions, modules in Python, shell, JavaScript, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Language: python, shell, bash, javascript, typescript, etc.",
                    },
                    "task": {
                        "type": "string",
                        "description": "Clear description of what the code should do",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context (existing code, file paths, dependencies)",
                    },
                    "constraints": {
                        "type": "string",
                        "description": "Optional constraints (style, libraries to use/avoid)",
                    },
                },
            },
        },
        {
            "name": "compile_latex",
            "description": "Compile a LaTeX .tex file to PDF. Use pdflatex (or xelatex if pdflatex fails). Runs twice for cross-references. Call after write_file when user asks to compile LaTeX to PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the .tex file (relative to project root, e.g. exports/paper.tex)",
                    },
                },
            },
        },
    ]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Execute a generation tool. Returns generated text."""
    session_context = session_context or {}

    provider_name = session_context.get("llm_provider_name", "ollama")
    try:
        from ...shared.llm_provider import get_llm_provider

        llm = get_llm_provider(provider_name)
    except Exception as e:
        return f"Error: Could not initialize LLM ({provider_name}): {e}"

    if name == "generate_content":
        content_type = args.get("content_type", "paper")
        topic = args.get("topic", "")
        outline = args.get("outline", "")
        output_format = args.get("output_format", "markdown")
        constraints = args.get("constraints", "")
        context = args.get("context", "")

        if not topic:
            return "Error: topic is required for generate_content"

        system_prompt = f"""You are an expert academic and technical writer. Generate high-quality {content_type} content.

OUTPUT FORMAT: {output_format.upper()}
- raw: Plain text, no special formatting
- markdown: Use markdown (headers, lists, bold, code blocks)
- latex: Use LaTeX syntax (\\section, \\textbf, \\cite, etc.)

Include citations where appropriate (use [Author et al., Year] or \\cite{{key}} for LaTeX).
Be thorough, accurate, and well-structured."""

        prompt_parts = [f"Generate a {content_type} on: {topic}"]
        if outline:
            prompt_parts.append(f"\nStructure/outline: {outline}")
        if constraints:
            prompt_parts.append(f"\nConstraints: {constraints}")
        if context:
            prompt_parts.append(f"\nContext: {context}")

        prompt = "\n".join(prompt_parts)

        try:
            response = llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=0.7,
            )
            return response or "No content generated."
        except Exception as e:
            return f"Error generating content: {e}"

    if name == "generate_code":
        language = args.get("language", "python")
        task = args.get("task", "")
        context = args.get("context", "")
        constraints = args.get("constraints", "")

        if not task:
            return "Error: task is required for generate_code"

        system_prompt = f"""You are an expert programmer. Generate clean, correct {language} code.

Requirements:
- Write complete, runnable code
- Include comments for non-obvious logic
- Follow best practices for {language}
- Output ONLY the code (no markdown fences unless the user needs a full file with imports)"""

        prompt_parts = [f"Task: {task}"]
        if context:
            prompt_parts.append(f"\nContext: {context}")
        if constraints:
            prompt_parts.append(f"\nConstraints: {constraints}")

        prompt = "\n".join(prompt_parts)

        try:
            response = llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )
            code = response or "No code generated."
            # Store for add_report_section when code goes in report (same pattern as last_table_summary_rows)
            session_context["last_generated_code"] = code
            return code
        except Exception as e:
            return f"Error generating code: {e}"

    if name == "compile_latex":
        filepath = args.get("filepath", "")
        if not filepath:
            return "Error: filepath is required for compile_latex"
        if not filepath.strip().lower().endswith(".tex"):
            return "Error: filepath must be a .tex file"

        try:
            tex_path = resolve_path(filepath, project_root)
            if not tex_path.exists():
                return f"Error: File not found: {filepath}"
            work_dir = tex_path.parent
            tex_name = tex_path.name

            for compiler in ["pdflatex", "xelatex"]:
                try:
                    # Run twice for cross-references and bibliography
                    for _ in range(2):
                        res = subprocess.run(
                            [compiler, "-interaction=nonstopmode", "-halt-on-error", tex_name],
                            cwd=str(work_dir),
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if res.returncode != 0:
                            return f"Error ({compiler}): {res.stderr or res.stdout or 'Compilation failed'}"
                    pdf_path = work_dir / tex_path.stem
                    pdf_path = pdf_path.with_suffix(".pdf")
                    if pdf_path.exists():
                        return f"Success: PDF created at {pdf_path.relative_to(project_root)}"
                    return f"Error: {compiler} ran but PDF not found at {pdf_path}"
                except FileNotFoundError:
                    continue
            return "Error: Neither pdflatex nor xelatex found. Install TeX Live or MiKTeX."
        except ValueError as e:
            return f"Error: {e}"
        except subprocess.TimeoutExpired:
            return "Error: LaTeX compilation timed out (120s)."
        except Exception as e:
            return f"Error compiling LaTeX: {e}"

    return f"Error: Unknown generation tool '{name}'"
