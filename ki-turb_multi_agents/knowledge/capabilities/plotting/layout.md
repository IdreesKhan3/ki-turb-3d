# Plotting layout

- Visualizer tools live under `agents/tools/physics/` and are registered for the `visualizer` role
- Page view helpers often live in `pages/<PageName>/views.py`
- Intent → preferred plot tool mapping: `agents/intent_detection.py` + `agents/page_schema.py`
- Role prompts: `agents/langgraph/prompts.py` (`VISUALIZER_PROMPT`)

Prefer extending registered `plot_*` tools over generating ad-hoc plotting scripts in the agent loop.
