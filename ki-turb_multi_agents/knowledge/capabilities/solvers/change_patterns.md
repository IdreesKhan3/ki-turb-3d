# Extending a solver backend

1. Read `CFDBackend` in `integrations/base.py`
2. Extend or create adapter under `integrations/`
3. Keep case IR in `schemas/cfd_case.py` solver-neutral when possible
4. Update request intent / plan factories only after adapter contract holds
5. Add contract tests under `tests/integrations/`
6. Verify with pytest; do not claim a solver is production-ready without compile/run smoke
