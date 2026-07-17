# External visualization (VTK / ParaView)

Current state: KI-TURB has in-app Plotly volume viewing; VTK/ParaView hooks are planned expansion points.

Likely touch points when integrating:

- Export adapters near `postprocessing/writers.py` / analysis products
- Optional tooling under `agents/tools/` for export/launch
- Documentation of dataset manifest field paths for external consumers

Prefer exporting standard VTK/VTU/XDMF from manifests over embedding a full ParaView server in-process.
