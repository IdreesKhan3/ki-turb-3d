"""Field-agnostic remote document routing."""
from __future__ import annotations

from pathlib import Path

from agents.langgraph.router import RequestRouter
from agents.remote_document import is_remote_document_request


def test_remote_document_is_field_agnostic():
    cases = [
        "read this paper and screenshot figure 2 of the circuit schematic",
        "download the Nature article on CRISPR and show figure 1",
        "look up the OpenLB user guide online and extract the boundary condition diagram",
        "fetch https://example.org/book.pdf and show page 12",
        "find the textbook chapter on Kolmogorov spectra and screenshot the figure",
    ]
    for text in cases:
        assert is_remote_document_request(text), text
        plan = RequestRouter(planner_agent=None, project_root=Path(".")).deterministic_plan(text)
        assert plan is not None, text
        assert plan.rationale == "schema:remote document", (text, plan.rationale)
        assert plan.steps[0].role == "analyst"
        assert plan.steps[0].tool is None
        assert "web_search" in plan.steps[0].instruction
        assert "Do NOT call KI-TURB plot_" in plan.steps[0].instruction


def test_local_pdf_is_not_hard_gated_remote():
    text = "read document reports/foo.pdf page 3"
    assert is_remote_document_request(text) is False
    # Local docs go to LLM/free-form — not the remote-document hard gate.
    plan = RequestRouter(planner_agent=None, project_root=Path(".")).deterministic_plan(text)
    assert plan is None


def test_turbulence_stats_plot_not_stolen_by_remote_docs():
    text = "plot dissipation rate from openlb turbulence stats"
    assert is_remote_document_request(text) is False
    # Keyword plot+openlb is NOT an auto analyze hard gate (LLM decides).
    plan = RequestRouter(planner_agent=None, project_root=Path(".")).deterministic_plan(text)
    assert plan is None


def test_show_figure_alone_is_not_remote_document():
    """Local plot phrasing must not steal solver/analysis tools."""
    assert is_remote_document_request("show the figure here in chat") is False
    assert is_remote_document_request(
        "run the simulation then plot energy spectra on one figure and show it"
    ) is False


def test_solver_lifecycle_beats_remote_document_heuristic():
    """Lifecycle hard gate is backend-agnostic and outranks remote-doc guesses."""
    text = (
        "Compile and run OpenLB FHIT N=16^3 tau=0.506 with MRT, "
        "then plot energy spectra and show the figure here."
    )
    assert is_remote_document_request(text) is False
    plan = RequestRouter(planner_agent=None, project_root=Path(".")).deterministic_plan(text)
    assert plan is not None
    assert plan.rationale != "schema:remote document"
    tools = [step.tool for step in plan.steps]
    assert "build_simulation_case" in tools or "compile_simulation" in tools
