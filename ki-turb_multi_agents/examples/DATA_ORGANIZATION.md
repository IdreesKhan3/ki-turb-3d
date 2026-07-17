# Example Data

Example turbulence simulation datasets for KI-TURB 3D.

## Directories

```
examples/
├── DNS/                      # Direct Numerical Simulation (411 MB)
│   ├── 128/                 # 128³ grid
│   ├── 256/                 # 256³ grid  
│   └── 512/                 # 512³ grid
├── LES/                      # Large Eddy Simulation (201 MB)
│   └── 64/                  # 64³ grid with tau analysis
├── agent_reports/            # Agent-generated reports (Autonomous Lab output)
├── ForcingComparison/        # Exported plots (8.6 MB)
├── IsotropyValidation/      # Exported plots (2.3 MB)
└── ESS/                      # Exported plots (1.5 MB)
```

## Usage

**Clone with Git LFS:**
```bash
git lfs install
git clone https://github.com/IdreesKhan3/ki-turb-3d.git
cd ki-turb-3d
streamlit run app.py
```

**In app:** Click "Try Example Data" button to browse datasets.

## File Types

**Data requirements:** Pre-computed: Energy Spectra, Structure Functions, Flatness, Spectral Isotropy, Real Isotropy, Time-Series Stats. From velocity: PDFs, 3D Volume Viewer (no pre-computed stats needed).

Each simulation directory contains:
- `simulation.input` - Parameters (LBM, Fortran namelist format)
- `simulation.json` - Parameters (Navier-Stokes, JSON format; alternative to simulation.input)
- `turbulence_stats*.csv` - Time series
- `spectrum*.dat` - Energy spectra
- `norm*.dat` - Normalized spectra
- `isotropy_coeff_*.dat` - Spectral isotropy
- `flatness_data*_*.txt` - Flatness factors
- `structure_functions_*.txt` - Structure functions (text format)
- `structure_funcs*_t*.bin` - Structure functions (ESS analysis, binary)
- `eps_real_validation*.csv` or `turbulence_validation*.csv` - Real-space validation (u_rms or u_rms_real, frac_x/frac_y/frac_z or E_x/E_y/E_z)
- `*.vti` / `*.h5` / `*.hdf5` - 3D velocity fields (Git LFS)
- `tau_analysis_*.bin` - Effective relaxation time τ_e for Knudsen number (LBM LES only, Git LFS)

**LBM vs Navier-Stokes:** The Overview page supports both. Use `simulation.input` for LBM (DNS/LES) and `simulation.json` for NS. For NS, include `c_sound` and `L` in simulation.json for Mach and Knudsen validation.

## Your Data

Add your simulations to `user/DNS/` or `user/LES/` (ignored by git).

## Git LFS

Large files (`.vti`, `.h5`, `.bin`) use Git LFS. Total download: ~620 MB.
