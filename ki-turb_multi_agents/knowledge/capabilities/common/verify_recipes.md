# Verify recipes

Allowlisted verify patterns for the engineer role:

- `pytest <path> -q`
- `python -m pytest <path> -q`
- `python -c "import …"`
- `python -m compileall <path>`
- `run_import_check` tool with a dotted module or `.py` path
- `run_pytest` tool with project-relative test paths

Do **not** use `cat` / `head` / `ls` as verify commands. Prefer `read_file` during
the edit step, or `python -m compileall <file>` / `python -c …` for verify.
(Inspect-shell leftovers are auto-translated, but avoid relying on that.)

Suggested defaults:

- Single script edits → `python -m compileall <path> -q` or `run_import_check`
- Registry permission changes → `tests/agents/test_tool_registry_permissions.py`
- Engineering routing → `tests/test_engineering_routing.py`
- Engineering smoke evals → `tests/evals/test_engineering_smoke.py`
