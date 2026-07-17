from agents.langgraph.fhit_routing import (
    ACTIVE_SIMULATION_JOB_ID,
    existing_openlb_data_plan,
    fhit_simulation_pipeline_plan,
    is_hit_simulation_request,
    is_load_existing_openlb_request,
    requested_post_run_analyses,
    resolve_simulation_stage,
)
from agents.langgraph.openlb_hit_build import (
    build_simulation_parameter_catalog,
    normalize_build_args,
    parse_openlb_build_args,
    simulation_build_step_instruction,
)
from agents.tools.simulation.hit_calibration import build_args_to_openlb_config


def test_fhit_request_detects_fhit_and_run_keywords():
    assert is_hit_simulation_request("Run FHIT 64^3 Smagorinsky with spectral forcing")
    assert is_hit_simulation_request("simulate homogeneous isotropic turbulence on openlb")
    assert not is_hit_simulation_request("plot energy spectra for DNS/512")


def test_openlb_smoke_test_without_hit_keyword_routes_to_simulation():
    query = (
        "run a small smoke test of openlb simulation where you compile run simulation, "
        "fetch data and compute energy spectra and visualize and show there figure here. "
        "use smagorinsky model grid size 8^3"
    )
    assert is_hit_simulation_request(query)
    plan = fhit_simulation_pipeline_plan(query)
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    assert tools[0] == "build_simulation_case"
    assert "compile_simulation" in tools
    assert "supervise_simulation" in tools
    assert "fetch_simulation_outputs" in tools
    assert tools[-2:] == ["compute_spectra", "plot_spectrum"]
    assert plan.steps[0].tool_args["resolution"] == [8, 8, 8]
    assert plan.steps[0].tool_args["scheme"] == "Smagorinsky"


def test_router_prefers_openlb_pipeline_over_spectra_page():
    from agents.langgraph.router import RequestRouter

    query = (
        "run a small smoke test of openlb simulation where you compile run simulation, "
        "fetch data and compute energy spectra and visualize. use smagorinsky 8^3"
    )
    plan = RequestRouter(planner_agent=None).deterministic_plan(query)
    assert plan is not None
    assert plan.steps[0].tool == "build_simulation_case"
    assert plan.steps[0].role == "simulation"


def test_build_catalog_lists_core_physics_parameters():
    catalog = build_simulation_parameter_catalog()
    for key in (
        "resolution",
        "reynolds_number",
        "viscosity",
        "relaxation_time",
        "scheme",
        "forcing_type",
        "mach_number",
        "box_length",
        "density",
        "char_velocity",
    ):
        assert key in catalog


def test_build_instruction_maps_user_request_without_hardcoded_defaults():
    text = (
        "OpenLB FHIT 32x64x128 D3Q27, Re=2500, nu=0.002, tau=0.55, "
        "MRT collision, spectral forcing k=1-3"
    )
    instruction = simulation_build_step_instruction(text)
    assert "build_simulation_case exactly once" in instruction
    assert "32x64x128" in instruction
    assert "D3Q27" in instruction
    assert "2500" in instruction
    assert "do not invent" in instruction.lower()
    assert "never `fhit` or `dhit`" in instruction


def test_parse_fhit_build_args_reads_cube_power_notation():
    args = parse_openlb_build_args("Run FHIT 64 grid Smagorinsky with spectral forcing")
    assert args["backend"] == "openlb"
    assert args["flow"] == "hit"
    assert args["hit_mode"] == "forced"
    assert args["resolution"] == [64, 64, 64]
    assert args["scheme"] == "Smagorinsky"
    assert args["forcing_type"] == "spectral_low_k"
    assert args["turbulence_regime"] == "les"
    assert "64" in args["name"]
    assert args["ic_wavenumber_max"] <= 31
    derived = build_args_to_openlb_config(args).derive_scaling()
    assert derived.actual_mach < 0.1


def test_parse_fhit_build_args_defaults_to_forced_mode():
    args = parse_openlb_build_args("run openlb hit simulation")
    assert args["hit_mode"] == "forced"


def test_parse_fhit_build_args_supports_dhit():
    args = parse_openlb_build_args("compile and run DHIT on openlb 32^3")
    assert args["hit_mode"] == "decaying"
    assert args["resolution"] == [32, 32, 32]


def test_normalize_build_args_maps_fhit_alias():
    args = normalize_build_args({"hit_mode": "fhit", "backend": "openlb", "name": "x"})
    assert args["hit_mode"] == "forced"


def test_fhit_pipeline_pins_build_with_parsed_args():
    query = (
        "openlb compile FHIT 16^3 Smagorinsky then compile then run simulation then fetch data "
        "then compute energy spectra"
    )
    plan = fhit_simulation_pipeline_plan(query)
    assert plan is not None
    assert plan.steps[0].tool == "build_simulation_case"
    assert plan.steps[0].role == "simulation"
    assert plan.steps[0].tool_args["resolution"] == [16, 16, 16]
    assert plan.steps[0].tool_args["hit_mode"] == "forced"
    assert plan.steps[1].tool == "compile_simulation"
    assert plan.steps[1].tool_args["job_id"] == ACTIVE_SIMULATION_JOB_ID
    assert [step.tool for step in plan.steps[-2:]] == ["compute_spectra", "plot_spectrum"]


def test_parse_fhit_build_args_reads_iteration_count():
    args = parse_openlb_build_args(
        "run FHIT 16^3 Smagorinsky spectral forcing for 1000 iterations"
    )
    assert args["max_steps"] == 1000
    assert args["resolution"] == [16, 16, 16]


def test_parse_fhit_build_args_reads_iterations_before_and_after_each_clause():
    args = parse_openlb_build_args(
        "run fhit simulation with mrt collsion in open lb iterations 1000 "
        "and save data after each 100 iterations test case fhit"
    )
    assert args["max_steps"] == 1000
    assert args["output_interval"] == 100
    assert args["scheme"] == "MRT"


def test_supervise_step_uses_poll_node_not_blocking_tool():
    from agents.langgraph.app_graph import AppGraphNodes
    from agents.langgraph.fhit_routing import fhit_simulation_pipeline_plan

    plan = fhit_simulation_pipeline_plan("run fhit mrt openlb for 1000 iterations")
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    supervise_index = tools.index("supervise_simulation")
    state = {"plan": plan.model_dump(mode="json"), "task_index": supervise_index}
    assert AppGraphNodes._is_supervise_step(state)


    plan = fhit_simulation_pipeline_plan("compile and run FHIT on openlb")
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    assert tools == [
        "build_simulation_case",
        "compile_simulation",
        "start_simulation",
        "supervise_simulation",
        "fetch_simulation_outputs",
        "postprocess_simulation_outputs",
        "load_dataset_manifest",
    ]


def test_meta_openlb_storage_question_does_not_start_simulation():
    query = "where did you save the openlb fetched data?"
    assert not is_hit_simulation_request(query)
    from agents.langgraph.router import RequestRouter

    plan = RequestRouter(planner_agent=None).deterministic_plan(query)
    assert plan is not None
    assert plan.steps[0].role == "steward"
    assert plan.steps[0].tool is None
    assert "Do NOT build" in plan.steps[0].instruction
    assert fhit_simulation_pipeline_plan(query) is None


def test_load_existing_openlb_data_does_not_start_simulation():
    queries = [
        "load already saved agents data from openlb simulations",
        "load the already saved openlb simulation data",
        "load existing openlb results and plot spectra",
        "use the saved openlb job data",
        "load job_5fa8049d84b4",
    ]
    for query in queries:
        assert is_load_existing_openlb_request(query), query
        assert not is_hit_simulation_request(query), query
        assert fhit_simulation_pipeline_plan(query) is None, query
        plan = existing_openlb_data_plan(query, project_root=None)
        assert plan is not None, query
        tools = [step.tool for step in plan.steps]
        assert tools[0] == "load_dataset_manifest", query
        assert "start_simulation" not in tools
        assert "build_simulation_case" not in tools
        assert "compile_simulation" not in tools


def test_router_load_existing_openlb_beats_run_pipeline():
    from agents.langgraph.fhit_routing import LATEST_SIMULATION_JOB_ID
    from agents.langgraph.router import RequestRouter
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    plan = RequestRouter(planner_agent=None, project_root=root).deterministic_plan(
        "load already saved agents data from openlb simulations"
    )
    assert plan is not None
    assert plan.steps[0].tool == "load_dataset_manifest"
    assert "start_simulation" not in [s.tool for s in plan.steps]
    job_id = (plan.steps[0].tool_args or {}).get("job_id")
    # Resolved latest job when present; otherwise the deferred sentinel.
    assert job_id and (
        str(job_id).startswith("job_") or str(job_id) == LATEST_SIMULATION_JOB_ID
    )


def test_multi_case_request_skips_single_lifecycle_hard_gate():
    """Multi-case compares must not collapse to one schema:run lifecycle."""
    from agents.langgraph.multi_case import is_multi_case_request
    from agents.langgraph.router import RequestRouter
    from pathlib import Path

    text = (
        "Compile and run two OpenLB FHIT cases (forced HIT), identical except collision: "
        "N=16^3, tau=0.506. Case A: MRT. Case B: BGK. For each: build → compile → start → "
        "wait → fetch → postprocess. Then compute_spectra for both, plot_spectrum on one "
        "figure with both curves, show it here, and report both job_ids."
    )
    assert is_multi_case_request(text)
    router = RequestRouter(planner_agent=None, project_root=Path("."))
    assert router.deterministic_plan(text) is None
    plan = router.plan(text, {})
    assert plan.rationale.startswith("Free-form simulation multi-case")
    assert plan.steps[0].role == "simulation"
    assert plan.steps[0].tool is None
    assert "MULTI-CASE" in plan.steps[0].instruction
    assert "silently substitute" in plan.steps[0].instruction


def test_single_case_run_still_hard_gated():
    from agents.langgraph.router import RequestRouter
    from pathlib import Path

    text = "compile and run OpenLB FHIT N=16^3 tau=0.506 MRT then plot energy spectra"
    plan = RequestRouter(planner_agent=None, project_root=Path(".")).deterministic_plan(text)
    assert plan is not None
    assert plan.rationale.startswith("schema:run")
    assert plan.steps[0].tool == "build_simulation_case"
    tools = [s.tool for s in plan.steps]
    assert tools.count("start_simulation") == 1
    assert "compute_spectra" in tools


def test_load_existing_with_spectra_chains_analysis_only():
    query = "load existing openlb results and plot energy spectra"
    plan = existing_openlb_data_plan(query)
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    assert tools[0] == "load_dataset_manifest"
    assert "start_simulation" not in tools
    assert "plot_spectrum" in tools


def test_openlb_mention_alone_does_not_start_simulation():
    assert not is_hit_simulation_request("what is openlb?")
    assert not is_hit_simulation_request("where is the openlb manifest?")
    assert not is_hit_simulation_request("load already saved agents data from openlb simulations")


def test_compile_only_stops_after_compile():
    queries = [
        "compile FHIT 64^3 MRT on openlb and stop",
        "set FHIT parameters 32^3 BGK and compile only",
        "openlb HIT 16^3 Smagorinsky just compile",
        "build and compile openlb FHIT 8^3 MRT without running",
        "compile openlb hit 64^3 mrt",
    ]
    for query in queries:
        assert is_hit_simulation_request(query), query
        assert resolve_simulation_stage(query) == "compile", query
        plan = fhit_simulation_pipeline_plan(query)
        assert plan is not None, query
        tools = [step.tool for step in plan.steps]
        assert tools == ["build_simulation_case", "compile_simulation"], query
        assert "start_simulation" not in tools


def test_compile_and_run_then_stop_does_not_force_analysis():
    query = "compile and run FHIT on openlb 16^3 MRT then stop"
    assert resolve_simulation_stage(query) == "run"
    plan = fhit_simulation_pipeline_plan(query)
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    assert tools == [
        "build_simulation_case",
        "compile_simulation",
        "start_simulation",
        "supervise_simulation",
        "fetch_simulation_outputs",
        "postprocess_simulation_outputs",
        "load_dataset_manifest",
    ]
    assert "compute_spectra" not in tools
    assert "plot_spectral_isotropy" not in tools


def test_run_then_spectral_isotropy_one_shot():
    query = (
        "run openlb FHIT 16^3 MRT, then compute and plot spectral isotropy"
    )
    plan = fhit_simulation_pipeline_plan(query)
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    assert "supervise_simulation" in tools
    assert "load_dataset_manifest" in tools
    assert tools[-2:] == ["compute_spectral_isotropy", "plot_spectral_isotropy"]
    assert "compute_spectra" not in tools


def test_run_then_multiple_analyses_one_shot():
    query = (
        "compile and run openlb FHIT 8^3 Smagorinsky then compute energy spectra "
        "and real isotropy and plot them"
    )
    plan = fhit_simulation_pipeline_plan(query)
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    assert "compute_spectra" in tools
    assert "plot_spectrum" in tools
    assert "compute_isotropy" in tools
    assert "plot_real_isotropy" in tools
    ids = [a.analysis_id for a in requested_post_run_analyses(query)]
    assert "energy_spectra" in ids
    assert "energy_fractions" in ids or "real_isotropy" in ids


def test_spectral_isotropy_does_not_imply_energy_spectra():
    ids = [a.analysis_id for a in requested_post_run_analyses(
        "run fhit then plot spectral isotropy"
    )]
    assert "spectral_isotropy" in ids
    assert "energy_spectra" not in ids


def test_component_spectra_does_not_imply_energy_spectra():
    ids = [a.analysis_id for a in requested_post_run_analyses(
        "run fhit then plot component spectra"
    )]
    assert "component_spectra" in ids
    assert "energy_spectra" not in ids


def test_bare_isotropy_routes_to_real_isotropy_after_run():
    query = "run openlb FHIT 16^3 then plot isotropy"
    ids = [a.analysis_id for a in requested_post_run_analyses(query)]
    assert "energy_fractions" in ids or "real_isotropy" in ids
    plan = fhit_simulation_pipeline_plan(query)
    tools = [step.tool for step in plan.steps]
    assert "plot_real_isotropy" in tools


def test_router_compile_only_is_deterministic():
    from agents.langgraph.router import RequestRouter

    plan = RequestRouter(planner_agent=None).deterministic_plan(
        "compile FHIT 32^3 MRT on openlb and stop"
    )
    assert plan is not None
    assert [s.tool for s in plan.steps] == [
        "build_simulation_case",
        "compile_simulation",
    ]

def test_openlb_set_grid_without_run_verb_is_simulation_not_dns_archive():
    query = (
        "from openlb set grid 32^3, Re 1600 iterations 5000 and save data after each "
        "500 iterations and then compute and plot its dissipation rate and visualize it"
    )
    assert is_hit_simulation_request(query)
    assert resolve_simulation_stage(query) == "run"
    plan = fhit_simulation_pipeline_plan(query)
    assert plan is not None
    tools = [step.tool for step in plan.steps]
    assert tools[0] == "build_simulation_case"
    assert "start_simulation" in tools
    assert "load_dataset_manifest" in tools
    assert tools[-1] == "plot_turbulence_stats"
    assert "load_data" not in tools
    args = plan.steps[0].tool_args
    assert args["resolution"] == [32, 32, 32]
    assert args["reynolds_number"] == 1600.0
    assert args["max_steps"] == 5000
    assert args["output_interval"] == 500
    plot_args = plan.steps[-1].tool_args
    assert any(t.get("y_col") == "eps_real" for t in plot_args.get("traces", []))


def test_router_openlb_dissipation_does_not_invent_examples_dns():
    from agents.langgraph.router import RequestRouter

    query = (
        "from openlb set grid 32^3, Re 1600 iterations 5000 and save data after each "
        "500 iterations and then compute and plot its dissipation rate and visualize it"
    )
    plan = RequestRouter(planner_agent=None).deterministic_plan(query)
    assert plan is not None
    assert plan.steps[0].tool == "build_simulation_case"
    assert all(
        (step.tool_args or {}).get("data_dir") != "examples/DNS/32"
        for step in plan.steps
    )
