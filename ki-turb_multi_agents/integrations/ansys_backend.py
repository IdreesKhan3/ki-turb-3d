"""ANSYS Fluent backend adapter.

Generates a Fluent journal file from a :class:`~schemas.cfd_case.CFDCase` and
runs Fluent in batch mode. The Fluent launcher path is taken from
``KITURB_ANSYS_EXECUTABLE`` or passed explicitly. The number of solver
iterations comes from the case runtime, and the mesh must be supplied via the
case geometry ``mesh_file``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from schemas import CFDCase, SimulationJob

from .base import LocalCommandBackend


class AnsysBackend(LocalCommandBackend):
    name = "ansys"
    env_var = "KITURB_ANSYS_EXECUTABLE"

    def _write_case_inputs(self, case: CFDCase, case_dir: Path) -> List[Path]:
        journal = case_dir / "case.jou"
        journal.write_text(_render_fluent_journal(case), encoding="utf-8")
        return [journal]

    def _build_argv(self, job: SimulationJob, executable: str) -> List[str]:
        return [executable, "3ddp", "-g", "-i", "case.jou"]


def _render_fluent_journal(case: CFDCase) -> str:
    iterations = case.runtime.max_steps if case.runtime.max_steps is not None else 1000
    mesh_file = case.geometry.mesh_file
    lines: List[str] = []
    if mesh_file:
        lines.append(f'/file/read-case "{mesh_file}"')
    if case.solver.reynolds_number is not None:
        lines.append(f"; Reynolds number: {case.solver.reynolds_number}")
    lines.extend(
        [
            "/solve/initialize/initialize-flow",
            f"/solve/iterate {iterations}",
            '/file/write-data "output/result.dat"',
            "/exit yes",
        ]
    )
    return "\n".join(lines) + "\n"
