"""CI smoke tests for engineering golden tasks."""
from __future__ import annotations

from pathlib import Path

from evals.engineering.runner import run_all

ROOT = Path(__file__).resolve().parents[2]


def test_engineering_golden_tasks_pass():
    results = run_all(ROOT)
    assert results, "no engineering eval tasks loaded"
    # Cheapest two tasks must pass in CI.
    by_id = {item["id"]: item for item in results}
    for task_id in ("plan_new_page", "plan_new_plot"):
        assert task_id in by_id, task_id
        assert by_id[task_id]["ok"], by_id[task_id]
    # Full suite should also be green when tasks.yaml is complete.
    failed = [item for item in results if not item["ok"]]
    assert not failed, failed
