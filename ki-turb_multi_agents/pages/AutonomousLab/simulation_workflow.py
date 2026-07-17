"""Sidebar panel that drives the CFD simulation workflow step by step."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agents.tools import simulation
from agents.tools._session_loader import load_manifest_into_session
from agents.tools.simulation import _store


def _run(label: str, tool: str, project_root: Path, **args) -> None:
    with st.spinner(f"{label}..."):
        result = simulation.execute_tool(tool, args, project_root)
    st.session_state.sim_workflow_log = result


def render_simulation_workflow(project_root: Path) -> None:
    st.markdown("### CFD Simulation")
    with st.form("sim_workflow_build"):
        name = st.text_input("Case name", value="hit_64")
        hit_mode = st.selectbox("HIT mode", ["forced", "decaying"], index=0)
        resolution = st.number_input("Resolution (cube N³)", min_value=4, max_value=512, value=64, step=1)
        scheme = st.selectbox("Collision model", ["Smagorinsky", "BGK", "DNS", "RLB"])
        forcing_type = st.selectbox(
            "Forcing type",
            ["spectral_low_k", "linear", "low_wavenumber", "none"],
            index=0 if hit_mode == "forced" else 3,
        )
        reynolds = st.number_input("Reynolds number", min_value=10.0, value=1000.0)
        mach = st.number_input("Mach number", min_value=0.01, max_value=0.09, value=0.05, step=0.01)
        tau = st.number_input("Relaxation time (tau)", min_value=0.51, max_value=1.5, value=0.53, step=0.01)
        max_steps = st.number_input("Max steps", min_value=100, value=10000, step=1000)
        output_interval = st.number_input("Output interval", min_value=1, value=1000, step=500)
        build = st.form_submit_button("Build & validate case")

    if build:
        args = dict(
            backend="openlb",
            flow="hit",
            name=name,
            hit_mode=hit_mode,
            resolution=[int(resolution)] * 3,
            scheme=scheme,
            forcing_type=forcing_type if hit_mode == "forced" else "none",
            reynolds_number=float(reynolds),
            mach_number=float(mach),
            relaxation_time=float(tau),
            max_steps=int(max_steps),
            output_interval=int(output_interval),
        )
        _run("Building", "build_simulation_case", project_root, **args)
        result = st.session_state.get("sim_workflow_log", "")
        for line in result.splitlines():
            if line.startswith("job_id:"):
                st.session_state.sim_workflow_job = line.split(":", 1)[1].strip()

    job_id = st.session_state.get("sim_workflow_job")
    if job_id and not (_store.job_dir(project_root, job_id) / "job.json").is_file():
        st.session_state.pop("sim_workflow_job", None)
        job_id = None
    if job_id:
        st.caption(f"Active job: `{job_id}`")
        c1, c2, c3 = st.columns(3)
        if c1.button("Compile", use_container_width=True):
            _run("Compiling", "compile_simulation", project_root, job_id=job_id)
        if c2.button("Run", use_container_width=True):
            _run("Starting", "start_simulation", project_root, job_id=job_id)
        if c3.button("Status", use_container_width=True):
            _run("Checking", "check_simulation_status", project_root, job_id=job_id)

        c4, c5, c6 = st.columns(3)
        if c4.button("Fetch", use_container_width=True):
            _run("Fetching", "fetch_simulation_outputs", project_root, job_id=job_id)
        if c5.button("Post-process", use_container_width=True):
            _run("Post-processing", "postprocess_simulation_outputs", project_root, job_id=job_id)
        if c6.button("Load", use_container_width=True):
            manifest_path = str(_store.job_dir(project_root, job_id) / _store.MANIFEST_FILENAME)
            ok, msg = load_manifest_into_session(project_root, manifest_path, st.session_state)
            st.session_state.sim_workflow_log = msg

    log = st.session_state.get("sim_workflow_log")
    if log:
        st.code(log)
