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
├── ForcingComparison/        # Exported plots (8.6 MB)
├── IsotropyValidation/       # Exported plots (2.3 MB)
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

Each simulation directory contains:
- `simulation.input` - Parameters
- `turbulence_stats*.csv` - Time series
- `spectrum*.dat` - Energy spectra
- `norm*.dat` - Normalized spectra
- `isotropy_coeff*.dat` - Spectral isotropy
- `flatness*.txt` - Flatness factors
- `structure_funcs*.bin` - Structure functions (ESS analysis)
- `eps_real_validation*.csv` - Real-space validation
- `velocity*.vti` / `velocity*.h5` - 3D fields (Git LFS)
- `tau_analysis*.bin` - Effective tau (LES only, Git LFS)

## Your Data

Add your simulations to `user/DNS/` or `user/LES/` (ignored by git).

## Git LFS

Large files (`.vti`, `.h5`, `.bin`) use Git LFS. Total download: ~620 MB.
