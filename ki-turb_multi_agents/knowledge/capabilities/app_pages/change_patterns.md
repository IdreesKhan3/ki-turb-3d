# Adding or removing a KI-TURB analysis page

Checklist:

1. Add/update entry in `agents/page_schema.py` (`PAGE_SCHEMA`)
2. Add intent constants / routing in `agents/intent_detection.py` if the page has agent tools
3. Implement compute tools under `agents/tools/physics/` when needed
4. Register tools in `agents/runtime/tool_registry.py` (analyst/visualizer sets)
5. Add Streamlit page under `pages/`
6. Add focused tests under `tests/`
7. Verify with `pytest` on the new tests + import check of the page module

Do not touch OpenLB HIT lifecycle graphs unless the page explicitly depends on new simulation products.
