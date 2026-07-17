# Solver integrations layout

- Contract: `integrations/base.py` (`CFDBackend`)
- Adapters: `integrations/openlb_backend.py`, `palabos_backend.py`, `ansys_backend.py`
- OpenLB-specific: `integrations/openlb/`
- Schemas: `schemas/cfd_case.py`, `schemas/simulation_job.py`, `schemas/openlb_hit.py`
- Agent lifecycle tools: `agents/tools/simulation/`
- Request routing: `agents/langgraph/request_intent.py`, `intent_plans.py`

Production depth today is OpenLB HIT; other solvers are thinner adapters. Prefer extending the shared contract before inventing parallel lifecycles.
