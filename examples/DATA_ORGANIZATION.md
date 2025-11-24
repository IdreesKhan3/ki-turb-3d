# Example Data Organization Guide

## Directory Structure

All example data should be placed in:
```
APP/examples/showcase/
```

This is the directory the app looks for when users click "Try Example Data" button.

---

## Complete File Organization

### Single Simulation Directory Structure

Place all files for one simulation directly in `examples/showcase/`:

```
examples/showcase/
├── simulation.input                    # Simulation parameters (REQUIRED for metadata)
├── turbulence_stats1.csv              # Time series statistics (REQUIRED)
│
├── spectrum_DNS_00050.dat              # Energy spectra snapshots
├── spectrum_DNS_00100.dat
├── spectrum_DNS_00150.dat
├── ... (more spectrum files)
│
├── norm_DNS_00050.dat                  # Normalized spectra with Pope model
├── norm_DNS_00100.dat
├── norm_DNS_00150.dat
├── ... (more norm files)
│
├── isotropy_coeff_DNS_00050.dat        # Spectral isotropy coefficients
├── isotropy_coeff_DNS_00100.dat
├── isotropy_coeff_DNS_00150.dat
├── ... (more isotropy files)
│
├── flatness_data1_t050000.txt          # Flatness factors
├── flatness_data1_t100000.txt
├── flatness_data1_t150000.txt
├── ... (more flatness files)
│
├── structure_functions_50000.txt       # Structure functions (text format)
├── structure_functions_100000.txt
├── structure_functions_150000.txt
├── ... (more structure function text files)
│
├── structure_funcs1_t50000.bin         # Structure functions (binary format, for ESS)
├── structure_funcs1_t100000.bin
├── structure_funcs1_t150000.bin
├── ... (more structure function binary files)
│
├── eps_real_validation_data1.csv       # Real-space validation (energy balance, isotropy)
├── eps_real_validation_data2.csv       # Multiple validation files for comparison
│
├── velocity_50000.vti                  # 3D velocity fields (VTK ImageData)
├── velocity_100000.vti
├── velocity_150000.vti
├── ... (more VTI files)
│
└── (other optional files)
```

---

## File Naming Conventions

### 1. Energy Spectra
- **Pattern**: `spectrum*.dat`
- **Examples**: 
  - `spectrum_DNS_00050.dat`
  - `spectrum_LES_00050.dat`
  - `spectrum1_00050.dat`
- **Format**: 2 columns (k, E(k))
- **Purpose**: Energy spectrum E(k) vs wavenumber k

### 2. Normalized Spectra (Pope Model)
- **Pattern**: `norm*.dat`
- **Examples**:
  - `norm_DNS_00050.dat`
  - `norm_LES_00050.dat`
  - `norm1_00050.dat`
- **Format**: Multiple columns (k, E(k), E_pope(k), ...)
- **Purpose**: Normalized spectrum with Pope model validation

### 3. Spectral Isotropy
- **Pattern**: `isotropy_coeff_*.dat`
- **Examples**:
  - `isotropy_coeff_DNS_00050.dat`
  - `isotropy_coeff_1_00050.dat`
- **Format**: Multiple columns (k, E11, E22, E33, dE11/dk, IC_standard, IC_deriv)
- **Purpose**: Spectral isotropy coefficient IC(k) = E₂₂(k)/E₁₁(k)

### 4. Flatness Factors
- **Pattern**: `flatness_data*_*.txt`
- **Examples**:
  - `flatness_data1_t050000.txt`
  - `flatness_data2_t100000.txt`
  - `flatness_DNS_t050000.txt`
- **Format**: 2 columns (r, flatness)
- **Header**: `# r flatness` (optional)
- **Purpose**: Longitudinal flatness F(r) = (1/3)[F_x + F_y + F_z]

### 5. Structure Functions (Text Format)
- **Pattern**: `structure_functions_*.txt`
- **Examples**:
  - `structure_functions_50000.txt`
  - `structure_functions_100000.txt`
- **Format**: Multiple columns (r, S_1, S_2, S_3, S_4, S_5, S_6)
- **Purpose**: Structure functions S_p(r) for orders p=1 to 6

### 6. Structure Functions (Binary Format - for ESS)
- **Pattern**: `structure_funcs*_t*.bin`
- **Examples**:
  - `structure_funcs1_t50000.bin`
  - `structure_funcs2_t100000.bin`
  - `structure_funcs_DNS_t50000.bin`
- **Format**: Fortran binary (see binary_reader.py for format)
- **Purpose**: Extended Self-Similarity (ESS) analysis: S_p vs S_3

### 7. Real-Space Validation (Energy Balance & Isotropy)
- **Pattern**: `eps_real_validation_*.csv`
- **Examples**:
  - `eps_real_validation_data1.csv`
  - `eps_real_validation_data2.csv`
- **Format**: CSV with columns: iter, eps_real, eps_spectral, frac_x, frac_y, frac_z, ...
- **Purpose**: 
  - Energy balance validation (eps_real vs eps_spectral)
  - Real-space isotropy (frac_x, frac_y, frac_z should be ~1/3)

### 8. Time Series Statistics
- **Pattern**: `turbulence_stats1.csv`
- **Format**: CSV with columns: iter, TKE, u_rms, eps, Re, ...
- **Purpose**: Overview page, time evolution plots

### 9. 3D Velocity Fields (VTI Files)
- **Pattern**: `*_*.vti` or `velocity_*.vti`
- **Examples**:
  - `velocity_50000.vti`
  - `ux_50000.vti`
  - `uy_50000.vti`
  - `uz_50000.vti`
- **Format**: VTK ImageData XML with appended binary data
- **Purpose**: 3D visualization (3D Slice Viewer page)

### 10. Simulation Parameters
- **Pattern**: `simulation.input`
- **Format**: Text file with Fortran parameters
- **Purpose**: Display simulation metadata (grid, physical, LBM parameters)

---

## Multi-Simulation Example (Optional)

If you want to demonstrate comparison features, you can organize like this:

### Option A: Multiple Directories
```
examples/
├── showcase/              # Default example (single simulation)
│   └── (all files here)
├── showcase_dns/          # DNS simulation
│   └── (DNS files)
├── showcase_les/          # LES simulation
│   └── (LES files)
└── showcase_mrt/          # MRT simulation
    └── (MRT files)
```

### Option B: Single Directory with Prefixes
```
examples/showcase/
├── simulation.input
├── turbulence_stats1.csv
│
├── spectrum_DNS_00050.dat      # DNS spectra
├── spectrum_LES_00050.dat      # LES spectra
├── spectrum_MRT_00050.dat      # MRT spectra
│
├── flatness_data1_t050000.txt  # DNS flatness
├── flatness_data2_t050000.txt  # LES flatness
├── flatness_data3_t050000.txt  # MRT flatness
│
└── ... (similar for other file types)
```

**Note**: The app groups files by simulation type using regex patterns, so consistent prefixes help.

---

## Minimum Example Dataset

For a working example, include at least:

### Required Files:
1. ✅ `simulation.input` - Parameters
2. ✅ `turbulence_stats1.csv` - Time series
3. ✅ `spectrum*.dat` - At least 5-10 spectrum files
4. ✅ `norm*.dat` - At least 5-10 normalized spectrum files

### Recommended Files:
5. ✅ `isotropy_coeff_*.dat` - At least 5-10 files
6. ✅ `flatness_data*_*.txt` - At least 5-10 files
7. ✅ `structure_functions_*.txt` - At least 1-2 files
8. ✅ `eps_real_validation_*.csv` - At least 1 file

### Optional Files:
9. ⚪ `structure_funcs*_t*.bin` - For ESS analysis
10. ⚪ `*.vti` - For 3D visualization (very large, consider excluding)

**Note**: Each complete grid size dataset is typically **30-40 MB**. For example data:
- **1 grid size**: ~30-40 MB (minimal, single simulation)
- **2 grid sizes**: ~60-80 MB (recommended, shows comparison)
- **3+ grid sizes**: ~90-120+ MB (comprehensive, possible with solutions below)

**Options for 3+ grid sizes (90-120+ MB)**:
1. ✅ **Git LFS** (Large File Storage) - Recommended for large example data
2. ✅ **Separate download** - Host example data on cloud storage (Google Drive, Dropbox, etc.)
3. ✅ **Optional example data** - Make it downloadable separately, not in main repo
4. ✅ **GitHub Releases** - Package example data as downloadable release assets
5. ✅ **Direct in repo** - GitHub allows repos up to several GB, but cloning becomes slow

**Recommendation**: Use **Git LFS** for example data if including 3+ grid sizes, or provide as separate download.

---

## File Size Guidelines

| File Type | Typical Size | Notes |
|-----------|-------------|-------|
| `simulation.input` | ~1 KB | Small text file |
| `turbulence_stats1.csv` | ~100 KB | Time series data |
| `spectrum*.dat` | ~50 KB each | Per snapshot |
| `norm*.dat` | ~50 KB each | Per snapshot |
| `isotropy_coeff_*.dat` | ~10 KB each | Per snapshot |
| `flatness_data*_*.txt` | ~20 KB each | Per snapshot |
| `structure_functions_*.txt` | ~20 KB each | Per snapshot |
| `structure_funcs*_t*.bin` | ~50-200 KB each | Per snapshot |
| `eps_real_validation_*.csv` | ~50-200 KB | Per validation |
| `*.vti` | ~10-100 MB each | Per snapshot (large!) |

**Total per grid size**: ~30-40 MB (with typical file counts)

**Recommendation**: For example data, use smaller datasets:
- 10-20 spectrum files (instead of 100+)
- 1-2 VTI files (they're large)
- Skip very large files unless necessary

---

## Quick Setup Checklist

To set up example data:

- [ ] Create directory: `examples/showcase/`
- [ ] Copy `simulation.input` file
- [ ] Copy `turbulence_stats1.csv` file
- [ ] Copy at least 5-10 `spectrum*.dat` files
- [ ] Copy at least 5-10 `norm*.dat` files
- [ ] Copy at least 5-10 `isotropy_coeff_*.dat` files
- [ ] Copy at least 5-10 `flatness_data*_*.txt` files
- [ ] Copy at least 1-2 `structure_functions_*.txt` files
- [ ] Copy at least 1 `eps_real_validation_*.csv` file
- [ ] (Optional) Copy 1-2 `*.vti` files for 3D visualization
- [ ] (Optional) Copy `structure_funcs*_t*.bin` files for ESS

---

## Testing Your Example Data

After organizing files, test with:

1. Run the app: `streamlit run app.py`
2. Click "Try Example Data" button
3. Check Overview page - should show file availability
4. Test each page:
   - Energy Spectra page
   - Flatness page (when implemented)
   - Structure Functions page (when implemented)
   - Isotropy page (when implemented)
   - 3D Slice Viewer (if VTI files included)

---

## Notes

- **File naming is flexible**: The app uses glob patterns, so variations work
- **Missing files are OK**: The app handles missing files gracefully
- **Keep it small**: Example data should be small enough to include in repository
- **Use consistent prefixes**: Helps with multi-simulation comparison
- **VTI files are large**: Consider excluding from repository, use Git LFS if needed

