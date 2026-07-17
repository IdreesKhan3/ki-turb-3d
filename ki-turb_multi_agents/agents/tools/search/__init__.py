"""
Search tools: regex, web search, semantic search, code analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


SEARCH_TOOL_NAMES = frozenset({
    "regex_search", "replace_regex",
    "web_search", "search_research_papers", "download_file", "browse_web",
    "semantic_search", "find_symbol_definitions", "find_symbol_references",
    "find_class_implementations",
})


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for search tools."""
    return [
        {
            "name": "regex_search",
            "description": "Search for regex pattern in files. Returns matches with context. Use for pattern matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search"},
                    "file_pattern": {"type": "string", "description": "File glob (default *.py)"},
                    "context_lines": {"type": "integer", "description": "Lines before/after match (default 2)"},
                    "max_results": {"type": "integer", "description": "Max matches to return (default 50)"},
                },
            },
        },
        {
            "name": "replace_regex",
            "description": "Replace regex matches in a file. Use \\1, \\2 for capture groups. Path relative to project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"},
                    "pattern": {"type": "string", "description": "Regex pattern to match"},
                    "replacement": {"type": "string", "description": "Replacement (use \\1 for group 1)"},
                },
            },
        },
        {
            "name": "web_search",
            "description": (
                "Search the live web for documentation, error fixes, theory, and how-tos. "
                "Use when local knowledge is insufficient, a tool/compile step failed with an "
                "unfamiliar error, or the user asks to look something up. Follow promising "
                "URLs with browse_web. Cite sources in the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results to return (default 5)"},
                },
            },
        },
        {
            "name": "search_research_papers",
            "description": (
                "Search arXiv for scientific papers (turbulence, LBM, OpenLB methods, spectra, "
                "structure functions, etc.). Prefer this for literature/theory questions, then "
                "browse_web on abstract/PDF URLs when you need details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default 10)"},
                },
            },
        },
        {
            "name": "download_file",
            "description": "Download a file from a URL. Saves to project by default. Requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to download from"},
                    "save_path": {"type": "string", "description": "Path to save file (optional, relative to project)"},
                },
            },
        },
        {
            "name": "browse_web",
            "description": (
                "Fetch a URL and extract readable page text. Use after web_search or "
                "search_research_papers to learn from docs, forum posts, manuals, or papers "
                "and apply that knowledge to the current problem."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to browse"}},
            },
        },
        {
            "name": "semantic_search",
            "description": "Find code by meaning/concept. Natural language query (e.g. 'where is spectrum computed', 'authentication logic'). Use for deep code exploration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query describing what to find"},
                    "top_k": {"type": "integer", "description": "Number of results (default 10)"},
                    "file_pattern": {"type": "string", "description": "File glob (default *.py)"},
                },
            },
        },
        {
            "name": "find_symbol_definitions",
            "description": "Find where a symbol (function, class, variable) is defined. AST-based code analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string", "description": "Name of the symbol to find"},
                    "file_pattern": {"type": "string", "description": "File glob (default *.py)"},
                },
            },
        },
        {
            "name": "find_symbol_references",
            "description": "Find where a symbol is used/referenced in the codebase. AST-based code analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string", "description": "Name of the symbol to find references for"},
                    "file_pattern": {"type": "string", "description": "File glob (default *.py)"},
                },
            },
        },
        {
            "name": "find_class_implementations",
            "description": "Find classes that inherit from a base class. AST-based code analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_class": {"type": "string", "description": "Name of the base class"},
                    "file_pattern": {"type": "string", "description": "File glob (default *.py)"},
                },
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    """Execute a search tool. Returns result string."""
    if name == "regex_search":
        from .regex_search import regex_search as do_regex_search
        pattern = args.get("pattern", "")
        if not pattern:
            return "Error: pattern required"
        result = do_regex_search(
            project_root,
            pattern,
            file_pattern=args.get("file_pattern", "*.py"),
            context_lines=int(args.get("context_lines", 2)),
            max_results=int(args.get("max_results", 50)),
        )
        if not result.get("ok"):
            return result.get("message", "Regex search failed")
        matches = result.get("matches", [])
        out = [f"Found {len(matches)} match(es)"]
        for m in matches[:20]:
            out.append(f"  {m['file']}:{m['line']} | {m['matched_text']}")
        return "\n".join(out)

    if name == "replace_regex":
        from .regex_search import replace_regex as do_replace_regex
        filepath = args.get("filepath", "")
        pattern = args.get("pattern", "")
        replacement = args.get("replacement", "")
        if not filepath or not pattern:
            return "Error: filepath and pattern required"
        result = do_replace_regex(project_root, filepath, pattern, replacement)
        if not result.get("ok"):
            return result.get("message", "Replace failed")
        return result.get("message", f"Replaced {result.get('replacements', 0)} occurrence(s)")

    if name == "web_search":
        query = args.get("query", "")
        if not query:
            return "Error: query required"
        num_results = int(args.get("num_results", 5))
        try:
            from .web_search import WebSearchTools
            tools = WebSearchTools()
            results = tools.web_search(query, num_results=num_results)
            return json.dumps(results) if isinstance(results, dict) else str(results)
        except Exception as e:
            return f"Web search error: {e}"

    if name == "search_research_papers":
        query = args.get("query", "")
        if not query:
            return "Error: query required"
        max_results = int(args.get("max_results", 10))
        try:
            from .web_search import WebSearchTools
            tools = WebSearchTools()
            results = tools.search_research_papers(query, max_results=max_results)
            return json.dumps(results) if isinstance(results, dict) else str(results)
        except Exception as e:
            return f"Research papers search error: {e}"

    if name == "download_file":
        url = args.get("url", "")
        if not url:
            return "Error: url required"
        save_path = args.get("save_path")
        if save_path and not Path(save_path).is_absolute():
            save_path = str(project_root / save_path)
        try:
            from .web_search import WebSearchTools
            tools = WebSearchTools()
            results = tools.download_file(url, save_path=save_path)
            return json.dumps(results) if isinstance(results, dict) else str(results)
        except Exception as e:
            return f"Download error: {e}"

    if name == "browse_web":
        url = args.get("url", "")
        if not url:
            return "Error: url required"
        try:
            from .web_search import WebSearchTools
            tools = WebSearchTools()
            results = tools.browse_web(url)
            return json.dumps(results) if isinstance(results, dict) else str(results)
        except Exception as e:
            return f"Browse web error: {e}"

    if name == "semantic_search":
        query = args.get("query", "")
        if not query:
            return "Error: query required"
        top_k = int(args.get("top_k", 10))
        file_pattern = args.get("file_pattern", "*.py")
        try:
            from .semantic_search import semantic_search as do_semantic_search
            result = do_semantic_search(str(project_root), query, top_k=top_k, file_pattern=file_pattern)
            if not result.get("ok"):
                return result.get("message", "Semantic search failed")
            outputs = result.get("outputs", {})
            results = outputs.get("results", [])
            lines = [result.get("message", ""), ""]
            for r in results:
                lines.append(f"  {r.get('file', '')} (score: {r.get('score', 0):.3f})")
                if r.get("preview"):
                    prev = r["preview"][:180].replace("\n", " ").strip()
                    lines.append(f"    {prev}{'...' if len(r.get('preview', '')) > 180 else ''}")
            return "\n".join(lines)
        except Exception as e:
            return f"Semantic search error: {e}"

    if name == "find_symbol_definitions":
        symbol_name = args.get("symbol_name", "")
        if not symbol_name:
            return "Error: symbol_name required"
        file_pattern = args.get("file_pattern", "*.py")
        try:
            from .code_analysis import find_symbol_definitions as do_find_defs
            result = do_find_defs(str(project_root), symbol_name, file_pattern)
            if not result.get("ok"):
                return result.get("message", "Find definitions failed")
            outputs = result.get("outputs", {})
            defs = outputs.get("definitions", [])
            lines = [result.get("message", ""), ""]
            for d in defs[:30]:
                lines.append(f"  {d.get('file', '')}:{d.get('line', '')} | {d.get('type', '')} {d.get('name', '')}")
                if d.get("signature"):
                    lines.append(f"    {d['signature']}")
            return "\n".join(lines)
        except Exception as e:
            return f"Find symbol definitions error: {e}"

    if name == "find_symbol_references":
        symbol_name = args.get("symbol_name", "")
        if not symbol_name:
            return "Error: symbol_name required"
        file_pattern = args.get("file_pattern", "*.py")
        try:
            from .code_analysis import find_symbol_references as do_find_refs
            result = do_find_refs(str(project_root), symbol_name, file_pattern)
            if not result.get("ok"):
                return result.get("message", "Find references failed")
            outputs = result.get("outputs", {})
            refs = outputs.get("references", [])
            lines = [result.get("message", ""), ""]
            for r in refs[:50]:
                lines.append(f"  {r.get('file', '')}:{r.get('line', '')} | {r.get('context', '')}")
            return "\n".join(lines)
        except Exception as e:
            return f"Find symbol references error: {e}"

    if name == "find_class_implementations":
        base_class = args.get("base_class", "")
        if not base_class:
            return "Error: base_class required"
        file_pattern = args.get("file_pattern", "*.py")
        try:
            from .code_analysis import find_class_implementations as do_find_impls
            result = do_find_impls(str(project_root), base_class, file_pattern)
            if not result.get("ok"):
                return result.get("message", "Find implementations failed")
            outputs = result.get("outputs", {})
            impls = outputs.get("implementations", [])
            lines = [result.get("message", ""), ""]
            for i in impls:
                lines.append(f"  {i.get('class', '')} in {i.get('file', '')}:{i.get('line', '')} (extends {i.get('base', '')})")
            return "\n".join(lines)
        except Exception as e:
            return f"Find class implementations error: {e}"

    return f"Error: Unknown search tool '{name}'"
