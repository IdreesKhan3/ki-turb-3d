"""
Tool Result Contract - Standardized format for all tool outputs
Provides a single source of truth for tool execution results
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    """Represents a user-visible artifact from tool execution"""
    type: str  # "code" | "text" | "json" | "table" (extend as needed)
    title: str = ""
    language: str = ""
    text: str = ""
    data: Any = None


@dataclass
class Completion:
    """Indicates whether the action completed the user's request"""
    is_terminal: bool = False  # action completed the user's request
    needs_user_input: bool = False  # stop loop and ask user (e.g., confirmation)
    reason: str = ""


@dataclass
class ToolResult:
    """Standardized tool execution result"""
    ok: bool
    tool: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)
    message: str = ""
    completion: Completion = field(default_factory=Completion)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "ok": self.ok,
            "tool": self.tool,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "artifacts": [a.__dict__ for a in self.artifacts],
            "message": self.message,
            "completion": self.completion.__dict__,
            "error": self.error,
        }
