"""Smoke tests for web search / browse / arXiv tools used by agents in-action."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.langgraph.intent_plans import plan_from_intent
from agents.langgraph.request_intent import classify_request
from agents.tools.search.web_search import WebSearchTools


@pytest.mark.network
def test_web_search_returns_results():
    tools = WebSearchTools()
    result = tools.web_search("OpenLB lattice Boltzmann", num_results=5)
    assert result.get("success") is True, result
    assert result.get("total", 0) >= 1, result
    assert result["results"][0].get("title")
    assert result["results"][0].get("link")


@pytest.mark.network
def test_arxiv_search_returns_papers():
    tools = WebSearchTools()
    result = tools.search_research_papers(
        "homogeneous isotropic turbulence structure functions",
        max_results=3,
    )
    assert result.get("success") is True, result
    assert result.get("total", 0) >= 1
    assert result["papers"][0].get("title")


@pytest.mark.network
def test_browse_web_extracts_content():
    tools = WebSearchTools()
    result = tools.browse_web("https://www.openlb.net/")
    assert result.get("success") is True, result
    assert "openlb" in (result.get("content") or "").lower() or "lattice" in (
        result.get("title") or ""
    ).lower()


@pytest.mark.network
def test_execute_tool_web_search_json(tmp_path: Path):
    from agents.tools.search import execute_tool

    raw = execute_tool("web_search", {"query": "lattice Boltzmann method", "num_results": 3}, tmp_path)
    data = json.loads(raw)
    assert data.get("success") is True
    assert data.get("total", 0) >= 1


def test_research_intent_routes_to_web_plan():
    intent = classify_request("search the web for OpenLB MRT collision docs")
    assert intent is not None
    assert intent.action == "research"
    plan = plan_from_intent(intent)
    assert plan is not None
    assert plan.steps[0].role == "analyst"
    assert "web_search" in plan.steps[0].instruction


def test_explain_without_analysis_routes_to_research():
    intent = classify_request("what is the Kolmogorov -5/3 spectrum")
    assert intent is not None
    assert intent.action == "research"


def test_research_then_run_compound_task():
    text = (
        "first search web online and learn what are the best parameters for openlb "
        "to run and validate dhit simulation. and then based on those parameters run the simulation"
    )
    intent = classify_request(text)
    assert intent is not None
    assert intent.action == "research_then_run"
    assert intent.case_params.get("hit_mode") == "decaying"
    plan = plan_from_intent(intent)
    assert plan is not None
    assert plan.rationale.startswith("schema:research_then_run")
    assert plan.steps[0].role == "analyst"
    assert plan.steps[0].tool is None
    assert "web_search" in plan.steps[0].instruction
    assert plan.steps[1].role == "simulation"
    assert plan.steps[1].tool is None
    assert "build_simulation_case" in plan.steps[1].instruction
    tools = [s.tool for s in plan.steps]
    assert "compile_simulation" in tools
    assert "start_simulation" in tools


def test_web_search_not_misrouted_through_simulation_module():
    """Regression: simulation role may *use* web_search, but executor is search/."""
    from pathlib import Path
    import json
    from agents.tools import execute_tool, _module_key_for_tool

    assert _module_key_for_tool("web_search") == "search"
    assert _module_key_for_tool("browse_web") == "search"
    root = Path(__file__).resolve().parents[1]
    raw = execute_tool(
        "web_search",
        {"query": "OpenLB", "num_results": 2},
        root,
        allowed_tool_names={"web_search"},
    )
    assert "Unknown tool" not in str(raw)
    data = json.loads(raw)
    assert data.get("success") is True


