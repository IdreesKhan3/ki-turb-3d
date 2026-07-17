"""Score engineering plan-only golden tasks without mutating the repo."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from agents.knowledge.capability_loader import match_capabilities
from agents.langgraph.engineering_intent import parse_engineering_intent
from agents.langgraph.engineering_services import build_deterministic_plan, _repo_discoveries
from agents.langgraph.router import RequestRouter


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        return list(data.get("tasks") or [])
    # Minimal fallback: not expected in CI if PyYAML is present.
    return []


def score_task(task: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    request = str(task.get("request") or "")
    router = RequestRouter(planner_agent=None, project_root=project_root)
    plan = router.plan(request, {})
    intent = parse_engineering_intent(request, {})
    caps = match_capabilities(request, project_root)
    discoveries = _repo_discoveries(project_root, caps)
    eng_plan = build_deterministic_plan(
        request,
        capabilities=caps,
        discoveries=discoveries,
        plan_only=bool(task.get("plan_only", True)),
    )
    mentioned = set(eng_plan.create) | set(eng_plan.modify)
    for step in eng_plan.steps:
        mentioned.update(step.create)
        mentioned.update(step.modify)
    for item in eng_plan.discoveries:
        mentioned.add(item.file)

    require = [str(x) for x in (task.get("require_files") or [])]
    forbid = [str(x) for x in (task.get("forbid_files") or [])]
    missing = [p for p in require if not any(p in m or m.endswith(p) for m in mentioned)]
    forbidden_hit = [p for p in forbid if p in mentioned]

    ok = (
        plan.kind == "engineering_workflow"
        and intent is not None
        and not missing
        and not forbidden_hit
        and bool(eng_plan.steps)
        and bool(eng_plan.verify or any(step.verify for step in eng_plan.steps))
    )
    return {
        "id": task.get("id"),
        "ok": ok,
        "plan_kind": plan.kind,
        "capabilities": caps,
        "missing_required": missing,
        "forbidden_hit": forbidden_hit,
        "create": eng_plan.create,
        "modify": eng_plan.modify,
    }


def run_all(project_root: Path | None = None) -> List[Dict[str, Any]]:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    tasks_path = Path(__file__).resolve().parent / "tasks.yaml"
    return [score_task(task, root) for task in _load_tasks(tasks_path)]


__all__ = ["score_task", "run_all"]
