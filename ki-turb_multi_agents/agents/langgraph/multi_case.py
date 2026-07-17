"""Light structural cues for multi-case / comparison simulation requests."""
from __future__ import annotations

import re

# Structural only — not OpenLB/collision keyword soup.
_MULTI_CASE = re.compile(
    r"(?i)\b("
    r"two\s+(?:separate\s+)?(?:cases?|runs?|simulations?)|"
    r"both\s+(?:cases?|runs?|simulations?)|"
    r"case\s*[a-z]\b|"
    r"identical\s+except|"
    r"compare\s+(?:the\s+)?(?:two|both)|"
    r"vs\.?|versus"
    r")\b"
)


def is_multi_case_request(text: str) -> bool:
    """True when the user asked for more than one case/run in one request."""
    return bool(_MULTI_CASE.search(text or ""))


__all__ = ["is_multi_case_request"]
