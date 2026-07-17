"""Durable engineering lesson store (JSONL)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Lesson:
    task: str
    capability: str = ""
    symptoms: str = ""
    fix: str = ""
    files: List[str] = field(default_factory=list)
    verify: List[str] = field(default_factory=list)
    reuse_when: str = ""
    outcome: str = "unknown"
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lesson":
        return cls(
            task=str(data.get("task") or ""),
            capability=str(data.get("capability") or ""),
            symptoms=str(data.get("symptoms") or ""),
            fix=str(data.get("fix") or ""),
            files=list(data.get("files") or []),
            verify=list(data.get("verify") or []),
            reuse_when=str(data.get("reuse_when") or ""),
            outcome=str(data.get("outcome") or "unknown"),
            timestamp=float(data.get("timestamp") or time.time()),
        )


class LessonStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, lesson: Lesson) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> List[Lesson]:
        if not self.path.is_file():
            return []
        lessons: List[Lesson] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lessons.append(Lesson.from_dict(json.loads(line)))
            except Exception:
                continue
        return lessons

    def retrieve(
        self,
        query: str,
        *,
        capability: str = "",
        k: int = 5,
    ) -> List[Lesson]:
        tokens = {t for t in re.findall(r"[a-z0-9_]+", (query or "").lower()) if len(t) > 2}
        scored: List[tuple[int, Lesson]] = []
        for lesson in self.read_all():
            if capability and lesson.capability and lesson.capability != capability:
                # Soft filter: still allow if query overlaps strongly.
                pass
            hay = " ".join(
                [
                    lesson.task,
                    lesson.capability,
                    lesson.symptoms,
                    lesson.fix,
                    lesson.reuse_when,
                    " ".join(lesson.files),
                ]
            ).lower()
            score = sum(1 for t in tokens if t in hay)
            if capability and lesson.capability == capability:
                score += 2
            if score > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda item: (-item[0], -item[1].timestamp))
        return [lesson for _, lesson in scored[: max(0, k)]]


def default_store(project_root: Path | str) -> LessonStore:
    root = Path(project_root).resolve()
    return LessonStore(root / "knowledge" / "lessons" / "lessons.jsonl")


def record_lesson(
    project_root: Path | str,
    *,
    task: str,
    capability: str = "",
    symptoms: str = "",
    fix: str = "",
    files: Optional[List[str]] = None,
    verify: Optional[List[str]] = None,
    reuse_when: str = "",
    outcome: str = "unknown",
) -> Lesson:
    lesson = Lesson(
        task=task,
        capability=capability,
        symptoms=symptoms,
        fix=fix,
        files=list(files or []),
        verify=list(verify or []),
        reuse_when=reuse_when or task,
        outcome=outcome,
    )
    default_store(project_root).append(lesson)
    return lesson


def retrieve_lessons(
    project_root: Path | str,
    query: str,
    *,
    capability: str = "",
    k: int = 5,
) -> List[Lesson]:
    return default_store(project_root).retrieve(query, capability=capability, k=k)


def format_lessons(lessons: List[Lesson]) -> str:
    if not lessons:
        return ""
    blocks = []
    for i, lesson in enumerate(lessons, 1):
        blocks.append(
            f"{i}. outcome={lesson.outcome} capability={lesson.capability}\n"
            f"   task: {lesson.task}\n"
            f"   symptoms: {lesson.symptoms}\n"
            f"   fix: {lesson.fix}\n"
            f"   reuse_when: {lesson.reuse_when}\n"
            f"   files: {', '.join(lesson.files)}"
        )
    return "\n".join(blocks)


__all__ = [
    "Lesson",
    "LessonStore",
    "default_store",
    "record_lesson",
    "retrieve_lessons",
    "format_lessons",
]
