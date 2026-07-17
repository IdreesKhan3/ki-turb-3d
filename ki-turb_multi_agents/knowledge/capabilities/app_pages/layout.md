# App pages layout

Primary locations:

- `pages/` — Streamlit page modules (`NN_Title.py`) and page packages
- `agents/page_schema.py` — single source of truth for page workflows (file patterns, compute/plot tools)
- `agents/intent_detection.py` — natural-language intent → page/tool routing
- `pages/00_Autonomous_Lab.py` — chat/agent entry

When discovering pages, prefer `PAGE_SCHEMA` keys and existing page folders over inventing new trees.
