# kiTurbHIT3D

Real OpenLB lattice-Boltzmann solver for homogeneous isotropic turbulence (HIT).
KI-TURB agents control **all HIT physics** (DHIT/FHIT, DNS/LES, collisions, forcing)
entirely through `case.xml` — the binary is never rewritten per run.

## Modes

| Mode | `HITMode` | `ForcingType` | Physics |
|------|-----------|---------------|---------|
| **DHIT** | `decaying` | `none` | Decaying turbulence after solenoidal random-Fourier IC |
| **FHIT** | `forced` | see forcing table below | Statistically sustained via body forcing |

## Capability matrix (agent-controllable)

### Turbulence regime (`TurbulenceRegime`)

| Regime | Collisions |
|--------|------------|
| **DNS** | `BGK`, `DNS`, `RLB`, `MRT`, `TRT` |
| **LES** | `Smagorinsky`, `WALE`, `ConsistentStrainSmagorinsky`, `ShearSmagorinsky`, `Krause`, `SmagorinskyMRT`*, `DynSmagorinsky` |

\* `SmagorinskyMRT` is accepted in case XML and mapped to Smagorinsky BGK in this OpenLB tree.

### Forcing schemes (`ForcingType`, FHIT only)

| Scheme | Description |
|--------|-------------|
| `none` | DHIT (no forcing) |
| `linear` | Linear deterministic forcing |
| `spectral_low_k` / `low_wavenumber` | Band-limited spectral forcing |
| `abc` | ABC flow forcing |
| `constant` | Uniform body force |
| `ornstein_uhlenbeck` / `ou` | Stochastic OU forcing |

### Forcing patterns (`ForcingPattern`)

| Pattern | Use |
|---------|-----|
| `random_phase` | Random phase each update (default) |
| `fixed_phase` | Fixed spectral phases |
| `sine` / `cosine` | Harmonic temporal modulation |
| `ou_process` | OU temporal evolution |
| `abc_time` | Time-varying ABC coefficients |

### Geometry, mesh & LBM

- `Lx`, `Ly`, `Lz` — periodic cube side lengths
- `Nx`, `Ny`, `Nz`, `Dx`
- `Lattice` — `D3Q19`
- `Collision` — see regime table above
- `Tau`, `Mach`, `Viscosity`, `Reynolds`, `Density`, `CharVelocity`
- `SmagorinskyConstant`, `TRTMagicParameter`

### Initial condition

- `InitialCondition` — e.g. `divergence_free_random`
- `ICKMin`, `ICKMax` — wavenumber band for IC spectrum
- `ICSeed`, `ICSpectrumExponent` — RNG seed and E(k) ~ k^n scaling

### Forcing parameters (FHIT)

- `ForcingKMin`, `ForcingKMax`, `ForcingAmplitude`, `ForcingUpdateInterval`
- `TargetUrms`, `TargetReLambda`, `StatisticallyStationary`

### Runtime & output

- `MaxSteps`, `OutputInterval`, `SampleStartStep`, `CheckpointInterval`
- `WriteVelocity`, `WritePressure`, `WriteVorticity`

## Build

```bash
make    # uses OpenLB generic build (D3Q19<FORCE> + FHIT dynamics)
```

First build may compile the generic OpenLB library (~1–3 min). The KI-TURB backend
runs `make` automatically via `compile_simulation`.

## Usage

```bash
./kiTurbHIT3D case.xml output_dir
```

Output: `velocity_<step>.vti` (and optional `pressure_<step>.vti`) consumed by
the KI-TURB post-processing pipeline.

## Agent API

Agents call `build_simulation_case` with any combination of HIT parameters:

- `turbulence_regime` — `dns` | `les`
- `scheme` / `collision` — any collision from the matrix above
- `hit_mode` — `decaying` | `forced`
- `forcing_scheme` / `forcing_type` — any forcing scheme above
- `forcing_pattern` — any pattern above
- `trt_magic_parameter`, `smagorinsky_constant`, `mach_number`, `relaxation_time`, …

Or pass a full `case` CFDCase dict. The binary never changes; only `case.xml` does.
