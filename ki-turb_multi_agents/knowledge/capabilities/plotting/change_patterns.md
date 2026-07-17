# Adding plotting functionality

1. Implement or extend a `plot_*` / summary tool in `agents/tools/physics/`
2. Register the tool on the visualizer (and analyst compute tools if a new `compute_*` is needed)
3. Wire `page_schema` plot_tools / compute_tool
4. Update intent detection patterns
5. Add a unit/smoke test
6. Verify with scoped pytest + import check
