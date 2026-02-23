# KI-TURB 3D: Turbulence Analysis & Visualization Suite

**KI-TURB 3D** is a comprehensive web-based application for analyzing and visualizing turbulence data from Lattice Boltzmann Method (LBM) and Navier-Stokes (NS) simulations. The tool is universal for incompressible Forced Homogeneous Isotropic Turbulence (FHIT) from both LBM and NS.

KI-TURB 3D operates in **two usage modes**: **Manual** (fully private—navigate pages 01–13, load data, plot directly; no AI, no web search) and **Agent** (use the Autonomous Lab with a domain-aware 5-agent system: Orchestrator, Steward, Analyst, Visualizer, Reviewer; choose Cloud LLM or Local LLM). Both modes use the same data, same pages, same analyses—choose the interface that fits your workflow.

## Features

- **Integrated AI Assistant (Domain-Aware 5-Agent System)**: Plan-driven Orchestrator, Steward, Analyst, Visualizer, and Reviewer with controlled execution. Supports open-domain conversation and tool-assisted tasks: search codebase, execute code, web/literature search, reviewed file operations (confirmation for destructive changes). Domain-aware for turbulence analysis (spectra, isotropy, flatness, structure functions, etc.).
- **Energy Spectra Analysis**: Compute and visualize 3D kinetic energy spectra E(k) with Kolmogorov scaling validation
- **Structure Functions**: Longitudinal structure functions S_p(r) with Extended Self-Similarity (ESS) analysis and scaling exponents
- **Isotropy Validation**: Real-space and spectral isotropy analysis with Lumley triangle visualization
- **Probability Density Functions**: PDFs for velocity, vorticity, dissipation, and enstrophy
- **Flatness Factors**: Intermittency analysis via flatness factors F(r)
- **Time Series Statistics**: Including kinetic energy, dissipation rate, Reynolds numbers, and energy balance tracking and more.
- **3D Visualization**: Interactive 3D volume viewer with ParaView-like controls for velocity fields
- **Multi-Simulation Comparison**: Side-by-side comparison of multiple DNS/LES runs
- **Research-Grade Export**: Export plots to PNG, PDF, SVG, JPG, WEBP, and HTML formats
- **Customizable Plotting**: Full control over plot styling, fonts, colors, grids, and themes
- **Report Generation**: The report generator has access to all the pages to capture figures and tables

## Requirements

- **Python**: 3.10 or higher (Python 3.12 recommended)
- **Operating System**: Windows 10/11, Linux, or macOS
- **Memory**: Minimum 4 GB RAM (8 GB recommended for large datasets)
- **Browser**: Modern web browser (Chrome, Firefox, Edge, Safari)

## Usage Modes

### 🔒 Fully Private Mode (Manual)

Use pages **01–13** directly (Overview, Theory, Energy Spectra, etc.). Fully local—all processing runs on your machine; no AI, agents, or external connections.

### 🤖 Agent Mode (Autonomous Lab)

Use the **00 Autonomous Lab** page to chat with the 5-agent system. Same analyses as Manual mode, invoked via natural language. Choose one of two LLM backends:

| Backend | LLM | Privacy |
|---------|-----|---------|
| **Cloud** | **Google Gemini** | Chat messages and tool results (e.g. `read_file`, overview summaries) are sent to Google. Plots and raw data stay local. When agents use web search, queries go to DuckDuckGo. |
| **Local** | **Ollama** (Mistral, Qwen Coder, etc.) | LLM runs on your machine. When agents use web search, queries go to DuckDuckGo; otherwise processing stays local. |

**Note:** Agents can use web search (DuckDuckGo) when helpful—with either backend, web search sends queries externally. For maximum privacy, use Manual mode.

### AI Interaction Modes (Agent Mode Only)

When using the Autonomous Lab, the 5-agent system operates in:

- **Conversational Mode**: Conceptual explanations, turbulence theory, interpretation of results, and general questions
- **Agent Mode**: Tool-assisted execution for tasks such as locating files, inspecting data, editing code with review, and structured problem solving
- **Writing Mode**: Generation of structured scientific text, documentation, summaries, and reports

The agent operates within a controlled execution loop with explicit action validation, repetition prevention, and human-in-the-loop confirmation for any destructive or irreversible operations.

### Safety

- Destructive operations (delete, rename, overwrite) require explicit user confirmation.
- File operations are validated for scope and intent before execution.
- All actions can be reviewed and canceled by the user.

### Additional Privacy Controls

Telemetry Toggle (disable Streamlit's anonymous usage statistics via Privacy Settings in the sidebar). See [PRIVACY_POLICY.md](PRIVACY_POLICY.md) for complete details.

---

## Installation

**Quick Start (Linux/macOS):**
```bash
git lfs install && git clone https://github.com/IdreesKhan3/ki-turb-3d.git && cd ki-turb-3d
python3 -m venv myenv && source myenv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
streamlit run app.py
```

**Quick Start (Windows PowerShell):**
```powershell
git lfs install; git clone https://github.com/IdreesKhan3/ki-turb-3d.git; cd ki-turb-3d
python -m venv myenv; .\myenv\Scripts\Activate.ps1
pip install --upgrade pip; pip install -r requirements.txt
streamlit run app.py
```

### 1. Clone the Repository

This repository uses **Git LFS** (Large File Storage) for example datasets (~620 MB).

**With Git LFS (Recommended):**
```bash
git lfs install
git clone https://github.com/IdreesKhan3/ki-turb-3d.git
cd ki-turb-3d
```

**Without Git LFS:** Clone as above; example data will be placeholder files. Later: `git lfs install` and `git lfs pull` to fetch large files. The app runs without example data; provide your own data to run analyses.

### 2. Install Python

**Prerequisites:** Windows 10+; Linux or macOS; internet for initial setup. Administrator privileges may be required for Python installation.

| OS | Command / Steps |
|----|-----------------|
| **Windows** | **Option A:** `winget install Python.Python.3.12` **Option B:** Download from [python.org](https://www.python.org/downloads/), run installer, check "Add Python to PATH" |
| **Linux** | `sudo apt update && sudo apt install python3 python3-pip python3-venv` (Debian/Ubuntu) or equivalent for your distribution |
| **macOS** | `brew install python3` or download from [python.org](https://www.python.org/downloads/) |

Verify: `python --version` or `python3 --version` (should show 3.10+).

### 3. Create Virtual Environment and Install Dependencies

**Windows (PowerShell):**
```powershell
cd path\to\ki-turb-3d
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
cd path\to\ki-turb-3d
python -m venv myenv
myenv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
cd /path/to/ki-turb-3d
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Installs: streamlit, numpy, pandas, plotly, matplotlib, scipy, kaleido, weasyprint, pyvista, h5py, and optional requests, google-generativeai, beautifulsoup4.

### 4. Run the Application

**Windows:** Double-click `run_app.bat` or run `streamlit run app.py`  
**Linux/macOS:** `chmod +x run_app.sh && ./run_app.sh` or `streamlit run app.py`

Dashboard opens at `http://localhost:8501`. Stop with `Ctrl+C`.

---

## Usage

1. **Start**: Activate venv (if not already), run `streamlit run app.py` or the launcher script (`run_app.bat` / `run_app.sh`). Navigate to `http://localhost:8501` if it doesn't open.
2. **Load Data**: Sidebar → select simulation data directory (e.g. `examples/DNS/512`).
3. **Navigate Pages**: Use the sidebar to access analysis pages (see **Pages** below).
4. **Customize Plots**: Plot style sidebar for fonts, colors, grids, export.
5. **Export**: PNG, PDF, SVG, JPG, WEBP, HTML.

### Pages

| Page | Description |
|------|-------------|
| **00 Autonomous Lab** | Agent mode: 5-agent chat (Cloud or Local LLM). Manual mode: skip this page, use 01–13 directly. |
| **01 Overview** | Simulation parameters and metadata |
| **02 Theory & Equations** | Mathematical formulations, D3Q19 stencil, MRT matrix generator |
| **03 Multi Method Support** | LBM vs NS comparison (informational) |
| **04 Real Isotropy** | Real-space isotropy validation with Lumley triangle |
| **05 Spectral Isotropy** | Spectral isotropy analysis with component spectra |
| **06 Energy Spectra** | E(k) analysis with Kolmogorov scaling, time evolution |
| **07 Flatness Factors** | Intermittency analysis via flatness F(r) |
| **08 Structure Functions** | S_p(r) and ESS analysis |
| **09 PDFs** | Probability density functions (velocity, vorticity, dissipation, enstrophy) |
| **10 Other Turbulence Stats** | Kinetic energy, dissipation, Reynolds numbers, length scales, energy balance |
| **11 3D Volume Viewer** | Interactive volume rendering |
| **12 Report Generator** | Capture figures and tables, build scientific reports |
| **13 Citation** | Citation information (informational) |

### LLM Backend (Agent Mode Only)

- **Cloud (Gemini)**: Set `GOOGLE_API_KEY`. Faster and more efficient for agents. See "Integrated AI Assistant – Backend Configuration" below.
- **Local (Ollama)**: Set `OLLAMA_MODEL` (e.g., `mistral:7b`, `qwen2.5-coder:32b`). LLM runs locally on your machine. Manual mode works without any LLM.

### Data Format Requirements

For data loading and supported file formats, see [examples/DATA_ORGANIZATION.md](examples/DATA_ORGANIZATION.md).

| Analysis | Files |
|----------|-------|
| Energy Spectra | `spectrum*.dat`, `norm*.dat` (k, E(k) format) |
| Structure Functions | `structure_functions_*.txt`, `structure_funcs*_t*.bin` |
| Flatness | `flatness_data*_t*.txt` |
| Real Isotropy | `eps_real_validation*.csv`, `reynolds_stress_validation*.csv`, `turbulence_validation*.csv` |
| Spectral Isotropy | `isotropy_coeff_*.dat` |
| Other Turbulence Statistics | `turbulence_stats*.csv`, `eps_real_validation*.csv` |
| Velocity Fields | `*.vti` (VTK ImageData), `*.h5`/`*.hdf5` (HDF5) |
| Parameters | `simulation.input` (LBM), `simulation.json` (NS) |

**Note:** Assumes periodic boundary conditions; supports forced HIT from LBM and Navier-Stokes (DNS/LES).

---

## Project Structure

```
ki-turb-3d/
├── app.py                     # Application entry point
├── pages/                     # Streamlit analysis modules
│   ├── 00_Autonomous_Lab.py   # Agent chat
│   ├── 01_Overview.py
│   ├── 02_Theory_Equations.py
│   ├── 03_Multi_Method_Support.py
│   ├── 04_Real_Isotropy.py
│   ├── 05_Spectral_Isotropy.py
│   ├── 06_Energy_Spectra.py
│   ├── 07_Flatness.py
│   ├── 08_Structure_Functions.py
│   ├── 09_PDFs.py
│   ├── 10_Other_Turbulence_Stats.py
│   ├── 11_3D_Volume_Viewer.py
│   ├── 12_Report_Generator.py
│   └── 13_Citation.py
├── agents/                    # Domain-aware 5-agent system (Orchestrator, Steward, Analyst, Visualizer, Reviewer)
├── data_readers/              # Input format abstraction (csv, hdf5, spectrum, etc.)
├── utils/                     # Shared utilities (plotting, reporting, IO)
├── visualizations/            # Domain-specific visual tools (d3q19_lattice, etc.)
├── examples/                  # Example datasets (DNS/, LES/)
└── requirements.txt
```

---

## Dependencies

All dependencies are listed in `requirements.txt`. Install with `pip install -r requirements.txt` (see Installation).

**Core:** streamlit, numpy, pandas, plotly, matplotlib, scipy, kaleido, weasyprint, h5py, pyvista  
**For agent mode:** requests (Ollama), google-generativeai (Gemini), beautifulsoup4 (web search), st-chat-input-multimodal (voice/multimodal)

---

## Integrated AI Assistant – Backend Configuration (Agent Mode)

Configure an LLM backend only when using the Autonomous Lab (Agent mode). Manual mode requires no LLM.

| Option | Backend | Setup |
|--------|---------|-------|
| **Cloud** | **Google Gemini** | Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey). **Windows:** `$env:GOOGLE_API_KEY="your-key"` or add to `run_app.bat`. **Linux/macOS:** `export GOOGLE_API_KEY="your-key"` or add to `run_app.sh`. Typically faster and more efficient. |
| **Local** | **Ollama** (Mistral, Qwen Coder, etc.) | Install from [ollama.com](https://ollama.com). Start: `ollama serve`. Pull model: `ollama pull mistral:7b` or `ollama pull qwen2.5-coder:32b`. Set `OLLAMA_MODEL` in `run_app.bat`/`run_app.sh`. App auto-detects Ollama at `http://localhost:11434`. No API keys. LLM runs locally; when agents use web search, queries go to DuckDuckGo. |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Python not found | Add Python to PATH; restart terminal |
| Virtual environment issues | Delete `myenv`, recreate: `python -m venv myenv`; ensure Python 3.10+ |
| Port 8501 in use | **Windows:** `Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` **Linux/macOS:** `lsof -ti:8501 \| xargs kill -9` Or: `streamlit run app.py --server.port 8502` |
| Streamlit not found / Import errors | Activate venv; `pip install -r requirements.txt` |
| File reading errors | Check data formats and permissions |

**Support:** GitHub Issues for bugs or unexpected behavior.

---

## Notes

- Add `myenv` to `.gitignore`
- **Manual mode** runs fully offline after setup; no internet required
- **Agent mode** requires an LLM backend (Gemini or Ollama); web search (when used) sends queries to DuckDuckGo
- All data processing is local; dashboard is localhost-only by default

---

## Citation

If you use **KI-TURB 3D** in your research, please cite the specific released version used.

**BibTeX:**
```bibtex
@software{ki_turb_3d,
  title   = {KI-TURB 3D: Turbulence Analysis and Visualization Suite},
  author  = {Muhammad Idrees Khan},
  year    = {2025},
  version = {2.0.0},
  url     = {https://github.com/IdreesKhan3/ki-turb-3d},
  license = {MIT}
}
```

**APA:** Khan, M. I. (2025). KI-TURB 3D: Turbulence Analysis and Visualization Suite (Version 2.0.0) [Computer software]. https://github.com/IdreesKhan3/ki-turb-3d

---

## License

MIT License. See `LICENSE` file.

## Acknowledgments

This project was developed with the assistance of AI-based tools. All scientific logic, analysis, and results have been carefully reviewed and curated by the author. Please report bugs or unexpected behavior via GitHub Issues.

**Maintainer:** Muhammad Idrees Khan
