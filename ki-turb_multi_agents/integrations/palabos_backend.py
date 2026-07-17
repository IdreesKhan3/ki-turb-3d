"""Palabos backend adapter.

Palabos is a C++ lattice-Boltzmann library whose cases read an XML input file.
This adapter serializes a :class:`~schemas.cfd_case.CFDCase` into that XML and
launches a prebuilt Palabos executable, whose path is taken from
``KITURB_PALABOS_EXECUTABLE`` or passed explicitly. The executable is invoked
with the case directory as its working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from xml.sax.saxutils import escape

from schemas import CFDCase, SimulationJob

from .base import LocalCommandBackend


class PalabosBackend(LocalCommandBackend):
    name = "palabos"
    env_var = "KITURB_PALABOS_EXECUTABLE"

    def _write_case_inputs(self, case: CFDCase, case_dir: Path) -> List[Path]:
        config = case_dir / "param.xml"
        config.write_text(_render_palabos_xml(case), encoding="utf-8")
        return [config]

    def _build_argv(self, job: SimulationJob, executable: str) -> List[str]:
        return [executable, "param.xml"]


def _render_palabos_xml(case: CFDCase) -> str:
    nx, ny, nz = case.mesh.resolution
    sx, sy, sz = case.geometry.size
    solver = case.solver
    runtime = case.runtime
    lines = [
        '<?xml version="1.0" ?>',
        "<geometry>",
        f"    <name> {escape(case.name)} </name>",
        f"    <nx> {nx} </nx> <ny> {ny} </ny> <nz> {nz} </nz>",
        f"    <lx> {sx} </lx> <ly> {sy} </ly> <lz> {sz} </lz>",
        "</geometry>",
        "<fluid>",
        f"    <reynolds> {solver.reynolds_number if solver.reynolds_number is not None else ''} </reynolds>",
        f"    <viscosity> {solver.viscosity if solver.viscosity is not None else ''} </viscosity>",
        "</fluid>",
        "<numerics>",
        f"    <maxSteps> {runtime.max_steps if runtime.max_steps is not None else ''} </maxSteps>",
        f"    <statIter> {runtime.output_interval} </statIter>",
        "</numerics>",
    ]
    return "\n".join(lines) + "\n"
