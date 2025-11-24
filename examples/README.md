# Example Data

This directory contains example simulation data for demonstrating the dashboard.

## Quick Start

1. **Place your example data in**: `examples/showcase/`
2. **See detailed guide**: [DATA_ORGANIZATION.md](DATA_ORGANIZATION.md)

## Directory Structure

```
examples/
├── README.md                    # This file
├── DATA_ORGANIZATION.md         # Detailed organization guide
└── showcase/                    # Place example data here
    ├── simulation.input
    ├── turbulence_stats1.csv
    ├── spectrum*.dat
    ├── norm*.dat
    ├── isotropy_coeff_*.dat
    ├── flatness_data*_*.txt
    ├── structure_functions_*.txt
    ├── structure_funcs*_t*.bin
    ├── eps_real_validation_*.csv
    └── *.vti
```

## Minimum Required Files

For a working example, include:
- `simulation.input` - Simulation parameters
- `turbulence_stats1.csv` - Time series statistics
- `spectrum*.dat` - Energy spectra (5-10 files minimum)
- `norm*.dat` - Normalized spectra with Pope model (5-10 files)

## File Naming Patterns

| File Type | Pattern | Example |
|-----------|---------|---------|
| Energy Spectra | `spectrum*.dat` | `spectrum_DNS_00050.dat` |
| Normalized Spectra | `norm*.dat` | `norm_DNS_00050.dat` |
| Spectral Isotropy | `isotropy_coeff_*.dat` | `isotropy_coeff_DNS_00050.dat` |
| Flatness | `flatness_data*_*.txt` | `flatness_data1_t050000.txt` |
| Structure Functions (text) | `structure_functions_*.txt` | `structure_functions_50000.txt` |
| Structure Functions (binary) | `structure_funcs*_t*.bin` | `structure_funcs1_t50000.bin` |
| Real-space Validation | `eps_real_validation_*.csv` | `eps_real_validation_data1.csv` |
| 3D Velocity Fields | `*_*.vti` | `velocity_50000.vti` |

## For More Details

See [DATA_ORGANIZATION.md](DATA_ORGANIZATION.md) for:
- Complete file organization guide
- File format specifications
- Multi-simulation setup
- File size guidelines
- Testing checklist

