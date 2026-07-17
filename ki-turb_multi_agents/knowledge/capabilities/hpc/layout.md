# HPC / GPU runners

Existing hooks:

- `integrations/remote_runner.py`
- `integrations/local_process.py`
- Simulation job model under `schemas/simulation_job.py`
- OpenLB execution settings / memory gates in physics constraints

When adding Slurm/GPU support, extend remote runner abstractions and keep agents talking to `CFDBackend` / job status tools — not raw scheduler scripts in role prompts.
