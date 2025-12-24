# KI-TURB 3D

### Turbulence Analysis & Visualization Suite

**KI-TURB 3D** is a web-based scientific application for **analysis and visualization of turbulence data** obtained from **Lattice Boltzmann Method (LBM)** simulations, including **Direct Numerical Simulation (DNS)** and **Large Eddy Simulation (LES)**.
The tool is designed primarily for **Homogeneous Isotropic Turbulence (HIT)** and is optimized for simulations using **Multiple Relaxation Time (MRT)** collision operators, while remaining fully compatible with **Single Relaxation Time (SRT)**, **BGK**, and **Two Relaxation Time (TRT)** formulations.

KI-TURB 3D provides a unified framework for **post-processing, statistical analysis, and 3D visualization** of turbulence fields, enabling reproducible, research-grade analysis of turbulent flows.

---

## Key Features

* **Energy Spectra Analysis**
  Computation and visualization of 3D kinetic energy spectra *E(k)* with Kolmogorov scaling validation

* **Structure Functions**
  Longitudinal structure functions *S&#8346;(r)* with **Extended Self-Similarity (ESS)** and scaling exponent estimation

* **Isotropy Validation**
  Real-space and spectral isotropy assessment, including Lumley triangle visualization

* **Probability Density Functions (PDFs)**
  PDFs of velocity components, vorticity, dissipation, and enstrophy

* **Flatness Factors**
  Intermittency analysis through flatness factors *F(r)*

* **Time-Series Statistics**
  Turbulent kinetic energy, dissipation rate, Reynolds numbers, integral and Kolmogorov scales, and energy balance tracking

* **3D Field Visualization**
  Interactive volume rendering of velocity and vorticity fields from VTK and HDF5 data

* **Multi-Simulation Comparison**
  Side-by-side comparison of multiple DNS/LES simulations

* **Research-Grade Export**
  Export figures in PNG, PDF, SVG, JPG, WEBP, and HTML formats

* **Customizable Plot Styling**
  Full control over fonts, colors, grids, themes, and layout

* **Report Generation**
  Generation of publication-ready scientific reports containing figures and tables

---

## System Requirements

* **Python**: 3.10 or higher (Python 3.12 recommended)
* **Operating System**: Windows, Linux, or macOS
* **Memory**: &ge; 4 GB RAM (8 GB recommended for large datasets)
* **Browser**: Modern web browser (Chrome, Firefox, Edge, Safari)

---

## Installation

### Cloning the Repository

This repository uses **Git LFS (Large File Storage)** to manage example turbulence datasets.

#### Option 1: With Git LFS (Recommended)

```bash
git lfs install
git clone https://github.com/IdreesKhan3/ki-turb-3d.git
cd ki-turb-3d
```

#### Option 2: Without Git LFS

```bash
git clone https://github.com/IdreesKhan3/ki-turb-3d.git
cd ki-turb-3d
git lfs install
git lfs pull
```

---

## Environment Setup

### Create a Virtual Environment

```bash
python -m venv myenv
```

### Activate the Environment

* **Windows**

```cmd
myenv\Scripts\activate
```

* **Linux / macOS**

```bash
source myenv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard will open at:

```
http://localhost:8501
```

To stop the application, press **Ctrl + C** in the terminal.

---

## Usage Workflow

1. **Load Simulation Data**
   Select a DNS or LES output directory using the sidebar

2. **Navigate Analysis Modules**

   * Overview & metadata
   * Energy spectra
   * Structure functions
   * PDFs
   * Flatness factors
   * Isotropy analysis
   * Time-series statistics
   * 3D volume visualization

3. **Customize Visualization**
   Adjust themes, fonts, colors, and export settings

4. **Export Results**
   Save figures and reports for publications or presentations

---

## Supported Data Formats

* **Energy Spectra**: `spectrum_data*.dat`
* **Structure Functions**: `structure_funcs*_t*.bin`
* **Flatness Data**: `flatness_data*_t*.txt`
* **Isotropy Validation**: `eps_real_validation*.csv`, `reynolds_stress_validation*.csv`
* **Turbulence Statistics**: `turbulence_stats*.csv`
* **Velocity Fields**: `.vti`, `.h5`, `.hdf5`
* **Simulation Parameters**: `simulation.input` (Fortran namelist)

**Note:** The analysis framework assumes **periodic boundary conditions** and is optimized for **forced HIT simulations**.

---

## Project Structure

```
ki-turb-3d/
├── app.py                  # Application entry point
├── pages/                  # Streamlit analysis modules
├── utils/                  # Plotting, reporting, IO utilities
├── data_readers/           # Input format abstraction
├── visualizations/         # Domain-specific visualization tools
├── examples/               # Example DNS and LES datasets
└── requirements.txt
```

---

## Core Dependencies

* `streamlit`
* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `plotly`
* `h5py`
* `pyvista`
* `kaleido`
* `weasyprint`

See `requirements.txt` for full version details.

---

## Citation

If you use **KI-TURB 3D** in your research, please cite the specific released version.

**BibTeX**

```bibtex
@software{ki_turb_3d,
  title   = {KI-TURB 3D: Turbulence Analysis and Visualization Suite},
  author  = {Muhammad Idrees Khan},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/IdreesKhan3/ki-turb-3d},
  license = {MIT}
}
```

---

## License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

## Support

Please use **GitHub Issues** to report bugs, request features, or ask questions related to turbulence analysis workflows.

---

**Maintainer**: Muhammad Idrees Khan

---
## Acknowledgments
This project was developed with the help of AI-based tools as part of the development workflow. All scientific logic, analysis, and results were reviewed and curated by the author, who remains fully responsible for the software’s functionality and outputs. 