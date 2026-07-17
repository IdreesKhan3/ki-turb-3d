"""Result of validating a CFD case against physics constraints.

A :class:`ValidationReport` collects individual :class:`ConstraintCheck` results.
Only failing checks with ``severity == "error"`` flip ``passed`` to ``False``;
warnings and info are advisory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConstraintCheck(BaseModel):
    name: str
    passed: bool
    severity: str = "error"          # info | warning | error
    message: str = ""
    value: Optional[Any] = None
    limit: Optional[Any] = None


class ValidationReport(BaseModel):
    passed: bool = True
    checks: List[ConstraintCheck] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add(self, check: ConstraintCheck) -> "ValidationReport":
        self.checks.append(check)
        if check.severity == "error" and not check.passed:
            self.passed = False
        return self

    def errors(self) -> List[ConstraintCheck]:
        return [c for c in self.checks if c.severity == "error" and not c.passed]
