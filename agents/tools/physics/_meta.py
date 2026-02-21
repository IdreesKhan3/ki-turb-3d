"""
Metadata helpers for physics tools.
Resolves source_file dynamically from __file__ so paths stay correct after refactors.
"""

from pathlib import Path
from typing import Dict


def get_artifact_source_meta(module_file: str, project_root: Path, tool_name: str) -> Dict[str, str]:
    """
    Return source_file and tool_name for figure artifacts.
    Path is resolved from the module's __file__—no hardcoding.
    """
    try:
        abs_path = Path(module_file).resolve()
        proj = Path(project_root).resolve()
        rel = abs_path.relative_to(proj)
        return {"source_file": str(rel).replace("\\", "/"), "tool_name": tool_name}
    except ValueError:
        return {"tool_name": tool_name}
