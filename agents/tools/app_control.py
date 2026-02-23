"""
App-level control tools: theme, data selection, selection mode.
Allows agents to control main app sidebar settings (Theme, Data Selection).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ._session_loader import load_data_into_session
from utils.theme_config import get_theme_list


APP_CONTROL_TOOL_NAMES = frozenset({
    "set_app_theme",
    "load_data",
    "set_selection_mode",
    "set_hdf5_format",
})


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Tool definitions for app control."""
    themes = get_theme_list()
    return [
        {
            "name": "set_app_theme",
            "description": "Set the app theme (applies to all pages). Use when user asks to switch theme, use dark mode, light mode, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "enum": themes,
                        "description": f"Theme name. Options: {', '.join(themes)}",
                    },
                },
                "required": ["theme"],
            },
        },
        {
            "name": "load_data",
            "description": "Load simulation data directory/directories into the app session. Use when user asks to switch dataset, load DNS/128, compare DNS/512 and LES/128, etc. Sets the session data path used by all subsequent plot/compute tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {
                        "type": "string",
                        "description": "Single directory path (e.g. examples/DNS/128, DNS/512). Use for single simulation.",
                    },
                    "data_directories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple directories for comparison (e.g. [\"examples/DNS/512\", \"examples/LES/128\"]). Use for multi-simulation comparison.",
                    },
                },
            },
        },
        {
            "name": "set_selection_mode",
            "description": "Set data selection mode: single simulation or multiple simulations (comparison). Use when user asks to compare simulations, switch to single mode, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["single", "multiple"],
                        "description": "single = one directory; multiple = compare multiple directories",
                    },
                },
                "required": ["mode"],
            },
        },
        {
            "name": "set_hdf5_format",
            "description": "Set HDF5 velocity file layout: Fortran (transpose) for Fortran-written HDF5 (OpenACC solver), or Default (no transpose) for Python-written or standard layout. Use when user asks to load/use HDF5 with fortran option, default HDF5, switch HDF5 format, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["fortran", "default"],
                        "description": "fortran = apply transpose for Fortran-written HDF5; default = no transpose (Python/standard layout)",
                    },
                },
                "required": ["format"],
            },
        },
    ]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
):
    """Execute app control tool. Requires Streamlit session state."""
    import streamlit as st

    if name == "set_app_theme":
        theme = args.get("theme")
        if not theme:
            return "Error: theme is required."
        themes = get_theme_list()
        if theme not in themes:
            return f"Error: Unknown theme '{theme}'. Available: {', '.join(themes)}"
        st.session_state.theme = theme
        if "plot_style" in st.session_state:
            del st.session_state["plot_style"]
        return {"status": "success", "message": f"Theme set to: {theme}"}

    if name == "load_data":
        data_dir = args.get("data_dir", "").strip()
        data_dirs = args.get("data_directories") or []

        if data_dirs:
            paths = [d.strip() for d in data_dirs if d and str(d).strip()]
            multi = True
        elif data_dir:
            paths = [data_dir]
            multi = False
        else:
            return "Error: Provide data_dir (single path) or data_directories (list of paths)."

        st.session_state.multi_directory_mode = multi
        success, msg = load_data_into_session(project_root, paths, multi, st.session_state)
        if success:
            return {"status": "success", "message": msg}
        return f"Error: {msg}"

    if name == "set_selection_mode":
        mode = args.get("mode")

        if mode == "single":
            st.session_state.multi_directory_mode = False
            return {"status": "success", "message": "Selection mode: Single Simulation"}
        if mode == "multiple":
            st.session_state.multi_directory_mode = True
            return {"status": "success", "message": "Selection mode: Multiple Simulations (Comparison)"}

        return f"Error: mode must be 'single' or 'multiple'. Got: {mode}"

    if name == "set_hdf5_format":
        fmt = args.get("format")
        if fmt == "fortran":
            st.session_state.hdf5_fortran_order = True
            st.cache_data.clear()
            return {"status": "success", "message": "HDF5 format: Fortran (transpose) — for Fortran-written velocity files"}
        if fmt == "default":
            st.session_state.hdf5_fortran_order = False
            st.cache_data.clear()
            return {"status": "success", "message": "HDF5 format: Default (no transpose) — for Python/standard layout"}
        return f"Error: format must be 'fortran' or 'default'. Got: {fmt}"

    return f"Error: Unknown app control tool '{name}'"
