# KI-TURB 3D — Turbulence Analysis & Multi-Agent Lab

**Version 3.0.0**

**KI-TURB 3D** is a Streamlit suite for analyzing and visualizing turbulence data from Lattice Boltzmann (LBM) and Navier–Stokes (NS) simulations — especially incompressible Forced Homogeneous Isotropic Turbulence (FHIT / HIT).

It has **two equal usage modes** that share the same pages, data formats, and analyses:


| Mode       | How you work                                                      | AI          |
| ---------- | ----------------------------------------------------------------- | ----------- |
| **Manual** | Pages **01–13**: load data, plot, export — fully local            | None        |
| **Agent**  | Page **00 Autonomous Lab**: natural-language multi-agent workflow | LLM + tools |


Manual mode never calls agents or the web. Agent mode can drive the same analyses, OpenLB HIT cases, files/docs, and (via the Engineer) plan-gated product changes — with human confirmation for destructive or expensive actions.

---



## What’s new in v3

- **Seven-role LangGraph team**, including a dedicated **Engineer** for pages/tools/integrations work
- **LLM-first planning** with a thin hard gate only for explicit simulation lifecycle (load / compile / run / status)
- **OpenLB HIT lifecycle** end-to-end: build → compile → run → supervise → fetch → postprocess → load into analysis pages
- **Job inventory & follow-ups**: list saved jobs, resolve files under the active `simulations/<job_id>/` tree
- **Failure recovery** (handoff / locate / explain) instead of failing closed on the first tool error
- **Compact live activity UI** while agents stream
- Solver adapters beyond OpenLB (`palabos`, `ansys`, …) are wired as backends; OpenLB HIT is the production-deep path today

---



## Features (analysis)

- Energy spectra E(k) with Kolmogorov scaling checks  
- Structure functions S_p(r) and ESS  
- Real-space & spectral isotropy (Lumley triangle, component spectra)  
- PDFs (velocity, vorticity, dissipation, enstrophy)  
- Flatness / intermittency F(r)  
- Time-series turbulence stats (TKE, \varepsilon, Re, energy balance, …)  
- Interactive 3D volume viewer  
- Multi-simulation comparison and research-grade export (PNG/PDF/SVG/JPG/WEBP/HTML)  
- Report generator across pages

---



## Dual control: Manual pages ↔ Agents

- **Manual:** sidebar → data directory → pages 01–13 → style/export. No LLM required.  
- **Agent:** Autonomous Lab plans a short workflow; specialists call the same compute/plot loaders that feed pages 01–13.  
- After a simulation job is fetched/postprocessed, agents `load_dataset_manifest` so Overview / Spectra / Isotropy / … see the new products — the same path Manual mode uses when you pick a data folder.  
- Steward can also set theme / selection / HDF5 options (`set_app_theme`, `set_selection_mode`, `set_hdf5_format`).

---



## Multi-agent system (v3)


| Role             | Responsibility                                                               |
| ---------------- | ---------------------------------------------------------------------------- |
| **Orchestrator** | Plan synthesis, answers, web-backed explanations                             |
| **Steward**      | Files, manifests, docs, Git, settings, verify helpers                        |
| **Simulation**   | Solver lifecycle tools (OpenLB HIT primary)                                  |
| **Analyst**      | Turbulence `compute_`* tools + research browse                               |
| **Visualizer**   | Registered `plot_`* / report tools                                           |
| **Reviewer**     | Independent scientific review (research tools)                               |
| **Engineer**     | Product engineering: inspect → plan → edit pages/tools/solvers → self-verify |


**Routing (high level):** explicit single-case load/compile/run/status/inquire → deterministic lifecycle plan; multi-case / compare runs → LLM planner (or free-form Simulation lead if no planner); everything else → LLM planner (or free-form Steward). Domain nouns alone do **not** force analyze/run pipelines. A completion self-check runs before finalize.

### Important tools by role

**Simulation — OpenLB HIT (and shared job APIs)**  
`build_simulation_case` · `compile_simulation` · `start_simulation` · `supervise_simulation` · `check_simulation_status` · `cancel_simulation` · `fetch_simulation_outputs` · `postprocess_simulation_outputs` · `load_dataset_manifest` / `read_dataset_manifest` · `list_simulation_jobs`

**Analyst — compute**  
`compute_spectra` · `compute_spectral_isotropy` · `compute_isotropy` · `compute_flatness` · `compute_structure_functions` · `compute_pdfs` · `compute_volume_field` · `compute_overview_validation` · `load_analysis_products` · `export_data`

**Visualizer — plots / reports**  
`plot_spectrum` · `plot_spectral_isotropy` / `plot_component_spectra` · `plot_real_isotropy` · `plot_lumley_triangle` · `plot_flatness` · `plot_structure_functions` · `plot_pdf` · `plot_turbulence_stats` · `plot_volume_3d` · report section tools · `export_figure`

**Steward / Engineer — repo & verify**  
`list_directory` · `find_file` · `read_file` / `read_document` · `write_file` / `modify_file` / `delete_file` · `search_codebase` · `git_operation` · `run_pytest` · `run_import_check` · `run_verify_command`  
*(Steward also:* `load_data`*,* `list_simulation_jobs`*, allowlisted* `run_shell_command`*.)*

**Research (Orchestrator / Reviewer / several roles)**  
`web_search` · `search_research_papers` · `browse_web`

Destructive edits, downloads, and simulation start/fetch/postprocess require **explicit user confirmation**. Shell is allowlisted (inspection/Git); use `delete_file` for deletions.

### OpenLB HIT from agents

Typical compile-only or full run flows:

1. `build_simulation_case` (HIT / FHIT / DHIT params, MRT/BGK/…, grid, …)
2. `compile_simulation` → artifacts under `simulations/<job_id>/executable/`
3. Optional: `start_simulation` → `supervise_simulation` → `fetch_simulation_outputs` → `postprocess_simulation_outputs` → `load_dataset_manifest`
4. Analyst / Visualizer tools (or Manual pages) on the loaded products

Jobs live under `simulations/job_*`. Use `list_simulation_jobs` to inventory saved runs; follow-up file opens resolve against the active job tree.

---



## Requirements

- **Python** 3.10+ (3.12 recommended)  
- **OS** Windows 10/11, Linux, or macOS  
- **RAM** 8 GB min; 16 GB recommended for 512³ volume / multi-run compare  
- **Browser** modern Chromium / Firefox / Safari / Edge

---



## Installation

**Linux/macOS:**

```bash
git lfs install && git clone https://github.com/IdreesKhan3/ki-turb-3d.git && cd ki-turb-3d
python3 -m venv myenv && source myenv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
streamlit run app.py
```

**Windows (PowerShell):**

```powershell
git lfs install; git clone https://github.com/IdreesKhan3/ki-turb-3d.git; cd ki-turb-3d
python -m venv myenv; .\myenv\Scripts\Activate.ps1
pip install --upgrade pip; pip install -r requirements.txt
streamlit run app.py
```

Repo uses **Git LFS** for example data (~620 MB). Without LFS, placeholders remain until `git lfs pull`. Launchers: `run_app.sh` / `run_app.bat`. App: `http://localhost:8501`.

---



## Usage

1. Activate venv → `streamlit run app.py` (or launcher).
2. **Manual:** load a data directory in the sidebar → pages 01–13.
3. **Agent:** open **00 Autonomous Lab** → pick LLM → chat (confirm when prompted).
4. Customize plots in the style sidebar; export PNG/PDF/SVG/JPG/WEBP/HTML.



### Pages


| Page                          | Role                                           |
| ----------------------------- | ---------------------------------------------- |
| **00 Autonomous Lab**         | Agent chat (skip in pure Manual mode)          |
| **01 Overview**               | Case / metadata                                |
| **02 Theory & Equations**     | NS/LBM theory, D3Q19, MRT                      |
| **03 Multi Method Support**   | LBM vs NS (info)                               |
| **04 Real Isotropy**          | Real-space isotropy / Lumley                   |
| **05 Spectral Isotropy**      | Spectral isotropy / components                 |
| **06 Energy Spectra**         | E(k), Kolmogorov checks                        |
| **07 Flatness Factors**       | F(r)                                           |
| **08 Structure Functions**    | S_p(r), ESS                                    |
| **09 PDFs**                   | Velocity / vorticity / dissipation / enstrophy |
| **10 Other Turbulence Stats** | TKE, \varepsilon, Re, balance                  |
| **11 3D Volume Viewer**       | Interactive volumes                            |
| **12 Report Generator**       | Capture figures & tables                       |
| **13 Citation**               | How to cite                                    |




### LLM backends (Agent mode only)


| Backend                   | Env                                       | Notes                                        |
| ------------------------- | ----------------------------------------- | -------------------------------------------- |
| **DeepSeek** (UI default) | `DEEPSEEK_API_KEY`                        | Chat/tool text leaves the machine            |
| **Google Gemini**         | `GOOGLE_API_KEY`                          | Same                                         |
| **Ollama** (local)        | `OLLAMA_MODEL` (e.g. `qwen2.5-coder:32b`) | LLM local; web search still external if used |


Plots and raw volumes stay local. For maximum privacy, use **Manual** mode.

### Data formats

See [examples/DATA_ORGANIZATION.md](examples/DATA_ORGANIZATION.md).


| Analysis            | Typical files                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------- |
| Spectra             | `spectrum*.dat`, `norm*.dat`                                                                |
| Structure functions | `structure_functions_*.txt`, `structure_funcs*_t*.bin`                                      |
| Flatness            | `flatness_data*_t*.txt`                                                                     |
| Real isotropy       | `eps_real_validation*.csv`, `reynolds_stress_validation*.csv`, `turbulence_validation*.csv` |
| Spectral isotropy   | `isotropy_coeff_*.dat`                                                                      |
| Stats               | `turbulence_stats*.csv`                                                                     |
| Volumes             | `*.vti`, `*.h5` / `*.hdf5`                                                                  |
| Params              | `simulation.input` (LBM), `simulation.json` (NS)                                            |


Assumes periodic BC; forced HIT from LBM and NS (DNS/LES).

---



## Project structure

```
ki-turb-3d/
├── app.py
├── pages/                     # Streamlit pages 00–13 (+ AutonomousLab helpers)
├── agents/
│   ├── langgraph/             # Router, graphs, recovery, turn memory, intent plans
│   ├── tools/                 # Role-scoped tools (physics, simulation, core, …)
│   ├── knowledge/             # Capability / lesson loaders
│   └── runtime/               # Tool registry & permissions
├── integrations/              # CFD backends (OpenLB; Palabos/Ansys hooks)
├── schemas/                   # Case / job / manifest models
├── postprocessing/            # HIT product pipelines
├── simulations/               # Local job dirs (job_*) — runtime artifacts
├── knowledge/capabilities/    # Engineering capability maps
├── examples/                  # Example DNS/LES datasets (Git LFS)
├── data_readers/ · utils/ · visualizations/
└── requirements.txt
```

---



## Dependencies

Listed in `requirements.txt`.

**Core:** streamlit, numpy, pandas, plotly, matplotlib, scipy, kaleido, weasyprint, h5py, pyvista  
**Agent mode:** langchain / langgraph, requests, google-generativeai, DeepSeek access, beautifulsoup4 / search providers, multimodal chat input  

---



## Safety & privacy

- Confirm before delete/rename/overwrite, downloads, and costly simulation steps  
- Paths constrained to the project; shell allowlisted  
- Telemetry: disable Streamlit usage stats in sidebar Privacy Settings — see [PRIVACY_POLICY.md](PRIVACY_POLICY.md)  
- Manual mode: fully offline after install; Agent mode needs an LLM (and optional web search)

---



## Troubleshooting


| Issue                       | Fix                                              |
| --------------------------- | ------------------------------------------------ |
| Python not found            | Install 3.10+; ensure PATH                       |
| Broken venv                 | Recreate `myenv`; reinstall `requirements.txt`   |
| Port 8501 busy              | Kill the process or `--server.port 8502`         |
| Import / Streamlit missing  | Activate venv; `pip install -r requirements.txt` |
| Example data tiny / missing | `git lfs install && git lfs pull`                |


**Support:** GitHub Issues.

---



## Citation

Cite the **exact released version** you used.

**BibTeX:**

```bibtex
@software{ki_turb_3d,
  title   = {KI-TURB 3D: Turbulence Analysis and Visualization Suite},
  author  = {Khan, Muhammad Idrees and Yao, Huadong},
  year    = {2026},
  version = {3.0.0},
  url     = {https://github.com/IdreesKhan3/ki-turb-3d},
  license = {MIT}
}
```

**APA:** Khan, M. I., & Yao, H. (2026). *KI-TURB 3D: Turbulence Analysis and Visualization Suite* (Version 3.0.0) [Computer software]. [https://github.com/IdreesKhan3/ki-turb-3d](https://github.com/IdreesKhan3/ki-turb-3d)

---



## License

MIT License — see `LICENSE`.

## Acknowledgments

Developed with assistance from AI-based tools and KI-TURB agents itself. Scientific logic and results are curated by the author. Report bugs via GitHub Issues.

**Maintainer:** Muhammad Idrees Khan