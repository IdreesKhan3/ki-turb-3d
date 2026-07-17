"""General remote document / literature request detection (field-agnostic)."""
from __future__ import annotations

import re

_ACTION = re.compile(
    r"(?i)\b("
    r"read|open|show|display|view|render|screenshot|screen[ -]?shot|"
    r"extract|inspect|fetch|download|find|locate|preview|look\s*up|lookup|"
    r"get|pull|browse"
    r")\b"
)
# Literature / document objects (strong remote-doc signal).
_DOC_ARTIFACT = re.compile(
    r"(?i)\b("
    r"paper|article|book|thesis|dissertation|preprint|manuscript|manual|"
    r"handbook|textbook|chapter|publication|document|report|whitepaper|"
    r"pdf|appendix|user\s+guide|documentation"
    r")\b"
)
# Visual nouns alone are too weak — local plot requests also say "show the figure".
_VISUAL_ARTIFACT = re.compile(
    r"(?i)\b("
    r"figure|fig\.?|diagram|schematic|illustration|image|plate|map|table"
    r")\b"
)
_LOCATOR = re.compile(
    r"(?ix)"
    r"(?:"
    r"https?://\S+"
    r"|doi:\S+"
    r"|\bdoi\b"
    r"|\barxiv\b"
    r"|\bpublmed\b"
    r"|\bieeexplore\b"
    r"|\bspringer\b"
    r"|\bscience\s*direct\b"
    r"|\bonline\b"
    r"|\bon\s+the\s+web\b"
    r"|\bfrom\s+the\s+web\b"
    r"|\bweb\s+page\b"
    r"|\bwebsite\b"
    r")"
)
_LOCAL_PATH = re.compile(
    r"""(?ix)
    (?:[a-z]:[\\/]|[/\\]|\.{1,2}[\\/])?
    [^\s"'<>|]+?
    \.(?:pdf|docx?|xlsx?|xls|html?|png|jpe?g|gif|webp|bmp|tiff?)
    (?=$|[\s"'.,;:)\]])
    """
)


def is_remote_document_request(text: str) -> bool:
    """
    True for online papers/books/docs from any field.

    Not triggered by local workflow phrasing like "show the figure" / "plot spectra"
    unless there is a web/DOI locator or a clear literature document noun.
    """
    text = (text or "").strip()
    if not text:
        return False
    if _LOCAL_PATH.search(text) and not re.search(r"(?i)https?://", text):
        return False

    has_action = bool(_ACTION.search(text))
    has_doc = bool(_DOC_ARTIFACT.search(text))
    has_visual = bool(_VISUAL_ARTIFACT.search(text))
    has_locator = bool(_LOCATOR.search(text))

    # Explicit web/DOI/literature locator + fetch/read/show intent.
    if has_locator and (has_action or has_doc or has_visual):
        return True
    # Local-looking "show figure" is not a remote document request.
    if has_action and has_doc:
        return True
    return False


def remote_document_plan_instruction(user_text: str) -> str:
    return (
        "Remote document request (any field). Do NOT call KI-TURB plot_* / compute_* tools.\n"
        f"User request: {user_text}\n"
        "1) Locate the source with web_search and/or search_research_papers "
        "(use whatever repository fits: arXiv, publishers, books, manuals, etc.).\n"
        "2) browse_web the best landing/HTML/PDF URL.\n"
        "3) download_file the PDF or the specific figure image into project tmp/.\n"
        "4) If a PDF page screenshot is needed, use read_document on the local file "
        "with the requested page number so the image is returned as a chat artifact.\n"
        "Cite the URL. Return local path(s) of saved media."
    )


__all__ = ["is_remote_document_request", "remote_document_plan_instruction"]
