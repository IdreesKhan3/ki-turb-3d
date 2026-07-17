"""Detect platform-engineering requests (pages, plots, solvers, VTK, HPC, …)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import WorkflowPlan, WorkflowStep

_ENG_VERB = re.compile(
    r"\b("
    r"inspect|plan|implement|add|remove|delete|refactor|wire|connect|integrate|"
    r"extend|create|build|modify|update|change|introduce|hook|port"
    r")\b",
    re.I,
)
# Platform product targets only — not generic "file"/"module" (too broad for user scripts).
_ENG_TARGET = re.compile(
    r"\b("
    r"page|pages|streamlit|page_schema|"
    r"plotting\s+tools?|plot\s+tools?|visualizer\s+tools?|"
    r"backend|solver|solvers|"
    r"openlb|palabos|openfoam|ansys|vtk|paraview|hpc|gpu|slurm|"
    r"integration|integrations|schema|registry|capability|"
    r"codebase|repository|remote runner|cluster|"
    r"agent(?:s)?(?:\s+prompt|\s+tool|\s+workflow)?"
    r")\b",
    re.I,
)
_DOC_EXTS = r"py|md|txt|yml|yaml|json|toml|cfg|ini|tex|bib|cls|sty|sh"
# Concrete project-relative paths the user named (examples/foo.py, exports/a.tex, …).
_NAMED_PATH = re.compile(
    rf"(?P<path>"
    rf"(?:examples|pages|agents|tests|integrations|postprocessing|knowledge|schemas|exports|paper|present)"
    rf"/(?:[\w.-]+/)*[\w.-]+\.(?:{_DOC_EXTS})"
    rf"|"
    rf"[\w.-]+(?:/[\w.-]+)+\.(?:{_DOC_EXTS})"
    rf")",
    re.I,
)
# "create a test.py file in examples" / "in exports … paper.tex"
_FILE_IN_DIR = re.compile(
    rf"\b(?P<file>[\w.-]+\.(?:{_DOC_EXTS}))\b(?:\s+file)?\s+in\s+"
    rf"(?P<dir>examples|tests|pages|agents|integrations|exports|paper)\b",
    re.I,
)
_DIR_THEN_FILE = re.compile(
    rf"\bin\s+(?P<dir>examples|tests|pages|agents|integrations|exports|paper)\b[^.\n]{{0,80}}?\b"
    rf"(?P<file>[\w.-]+\.(?:{_DOC_EXTS}))\b",
    re.I,
)
# "modify test.py" / "update paper.tex" without a directory prefix
_BARE_EDIT_FILE = re.compile(
    rf"\b(?:create|modify|update|edit|write|patch|change|improve|upgrade)\b[^.\n]{{0,80}}?\b"
    rf"(?P<file>[\w.-]+\.(?:py|tex|md|bib|sh))\b",
    re.I,
)
# Cues that the user is editing a manuscript/document source.
_DOC_EDIT_CUE = re.compile(
    r"(?i)\b("
    r"(?:\.?tex|latex)\s+(?:paper|file|manuscript|document)|"
    r"(?:paper|manuscript|document)\s+(?:\.?tex|latex)|"
    r"(?:current|existing|this|that)\s+(?:\.?tex|latex|paper|manuscript)\b|"
    r"(?:modify|update|change|edit|improve|upgrade)\b[^.\n]{0,80}?\b(?:\.?tex|latex)\b|"
    r"figure\b[^.\n]{0,80}?\b(?:\.?tex|latex|paper|manuscript)\b|"
    r"(?:\.?tex|latex|paper|manuscript)\b[^.\n]{0,80}?\bfigure\b"
    r")"
)
_PLATFORM_REQUEST = re.compile(
    r"\b("
    r"streamlit page|page_schema|tool registry|capability pack|"
    r"vtk|paraview|slurm|remote runner|cfd ?backend|"
    r"new (?:analysis )?page|wire (?:the )?page"
    r")\b",
    re.I,
)
_PLAN_ONLY = re.compile(
    r"\b("
    r"plan only|make a plan|what files|which files|inspect(?:\s+and)?\s+plan|"
    r"do not (?:edit|change|modify|implement)|without (?:editing|changing|implementing)"
    r")\b",
    re.I,
)
_CONTINUE = re.compile(
    r"\b("
    r"continue|proceed|do step(?:\s+\d+)?|execute(?:\s+the)?\s+plan|"
    r"approve(?:\s+and)?\s+execute|next step|run step(?:\s+\d+)?"
    r")\b",
    re.I,
)
_STEP_NUM = re.compile(r"\b(?:step|do step)\s+(\d+)\b", re.I)


@dataclass
class EngineeringIntent:
    plan_only: bool = False
    continue_execution: bool = False
    step_index: Optional[int] = None  # 0-based when set
    request: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "plan_only": self.plan_only,
            "continue_execution": self.continue_execution,
            "step_index": self.step_index,
        }


def extract_named_paths(text: str) -> List[str]:
    """Return unique project-relative paths mentioned in the user request."""
    found: List[str] = []
    seen = set()

    def _add(path: str) -> None:
        path = (path or "").lstrip("./")
        if not path:
            return
        key = path.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    raw = text or ""
    for match in _NAMED_PATH.finditer(raw):
        _add(match.group("path"))
    for match in _FILE_IN_DIR.finditer(raw):
        _add(f"{match.group('dir')}/{match.group('file')}")
    for match in _DIR_THEN_FILE.finditer(raw):
        _add(f"{match.group('dir')}/{match.group('file')}")
    # Bare "modify test.py" → examples/test.py when examples/ is mentioned, else keep bare.
    if "examples" in raw.lower():
        for match in _BARE_EDIT_FILE.finditer(raw):
            name = match.group("file")
            if "/" not in name:
                _add(f"examples/{name}")
    elif not found:
        for match in _BARE_EDIT_FILE.finditer(raw):
            name = match.group("file")
            if "/" not in name:
                _add(name)
    return found


def is_document_or_latex_edit_request(text: str) -> bool:
    """True when the user asks to edit an existing manuscript or document source."""
    raw = (text or "").strip()
    if not raw:
        return False
    if _PLATFORM_REQUEST.search(raw):
        return False
    paths = extract_named_paths(raw)
    if any(p.lower().endswith((".tex", ".bib", ".cls", ".sty")) for p in paths):
        return True
    if _DOC_EDIT_CUE.search(raw) and (
        _ENG_VERB.search(raw)
        or re.search(r"(?i)\b(figure|image|plot|dpi|resolution|improve|upgrade)\b", raw)
    ):
        return True
    return False


def is_simple_file_edit_request(text: str) -> bool:
    """
    True when the user asks to create/modify a concrete file/script and is NOT
    asking to change the KI-TURB product (pages, registries, solvers, …).
    Those belong to free-form steward, not the gated engineering subgraph.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if _PLATFORM_REQUEST.search(raw):
        return False
    if is_document_or_latex_edit_request(raw):
        return True
    paths = extract_named_paths(raw)
    if not paths:
        # "this file" / "that file" with a create/modify verb still prefers steward.
        if re.search(r"\b(?:this|that|the)\s+file\b", raw, re.I) and _ENG_VERB.search(raw):
            return True
        return False
    # User-script / example / export paths are almost never platform engineering.
    if any(p.lower().startswith(("examples/", "exports/", "paper/")) for p in paths):
        return True
    # Named source file + edit verb without platform product phrasing → steward.
    if _ENG_VERB.search(raw) and not _ENG_TARGET.search(raw):
        return True
    if _ENG_VERB.search(raw) and not _PLATFORM_REQUEST.search(raw):
        # "modify … plot … save fig" in a named script is still a file edit.
        if any(p.lower().endswith((".py", ".tex", ".md", ".sh")) for p in paths):
            return True
    return False


def is_engineering_request(text: str, session_summary: Optional[Dict[str, Any]] = None) -> bool:
    session_summary = session_summary or {}
    raw = (text or "").strip()
    if not raw:
        return False
    if _CONTINUE.search(raw) and session_summary.get("engineering_plan"):
        return True
    # Concrete user-file edits must not enter the platform engineering subgraph.
    if is_simple_file_edit_request(raw):
        return False
    if not _ENG_VERB.search(raw):
        return False
    if not _ENG_TARGET.search(raw):
        return False
    # Pure analysis/plot display without engineering verbs already excluded by verb check.
    # Avoid stealing obvious simulation lifecycle phrasing without code-change targets.
    lower = raw.lower()
    if re.search(r"\b(compile|start|run|supervise|fetch)\b.*\b(openlb|simulation|hit)\b", lower):
        if not re.search(r"\b(page|plot|tool|backend|integrat|code|file|module|hpc|vtk)\b", lower):
            return False
    return True


def parse_engineering_intent(
    text: str,
    session_summary: Optional[Dict[str, Any]] = None,
) -> Optional[EngineeringIntent]:
    if not is_engineering_request(text, session_summary):
        return None
    session_summary = session_summary or {}
    raw = (text or "").strip()
    continue_execution = bool(_CONTINUE.search(raw) and session_summary.get("engineering_plan"))
    step_index = None
    m = _STEP_NUM.search(raw)
    if m:
        step_index = max(0, int(m.group(1)) - 1)
        continue_execution = True
    plan_only = bool(_PLAN_ONLY.search(raw)) and not continue_execution
    if continue_execution:
        plan_only = False
    return EngineeringIntent(
        plan_only=plan_only,
        continue_execution=continue_execution,
        step_index=step_index,
        request=raw,
    )


def engineering_workflow_plan(request: str, intent: EngineeringIntent) -> WorkflowPlan:
    mode = "continue" if intent.continue_execution else ("plan_only" if intent.plan_only else "plan_and_optionally_execute")
    instruction = (
        f"Engineering mode={mode}. User request: {request}. "
        "Discover capability packs and repository evidence, draft an EngineeringPlan "
        "(create/modify/do_not_touch/verify/steps), seek approval, then execute only "
        "approved steps with mandatory verification."
    )
    return WorkflowPlan(
        kind="engineering_workflow",
        steps=[WorkflowStep(role="engineer", instruction=instruction)],
        rationale=f"Engineering workflow ({mode})",
    )


__all__ = [
    "EngineeringIntent",
    "extract_named_paths",
    "is_document_or_latex_edit_request",
    "is_engineering_request",
    "is_simple_file_edit_request",
    "parse_engineering_intent",
    "engineering_workflow_plan",
]
