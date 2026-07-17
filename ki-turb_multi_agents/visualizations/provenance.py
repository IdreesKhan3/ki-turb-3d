"""Attach machine-readable HIT provenance to Plotly and Matplotlib figures."""
from __future__ import annotations
from typing import Any, Dict, Optional
import json

REQUIRED_KEYS=("run_id","source_snapshots","physical_time_range","analysis_method_version","normalization","units","validation_status")

def normalize_figure_provenance(value: Optional[Dict[str,Any]]=None, **updates: Any)->Dict[str,Any]:
    data=dict(value or {});data.update({k:v for k,v in updates.items() if v is not None})
    for key in REQUIRED_KEYS:data.setdefault(key,None)
    return data

def stamp_plotly_figure(fig, provenance: Optional[Dict[str,Any]]=None, **updates: Any):
    data=normalize_figure_provenance(provenance,**updates)
    meta=dict(getattr(fig.layout,"meta",None) or {});meta["ki_turb_hit_provenance"]=data
    fig.update_layout(meta=meta)
    return fig

def stamp_matplotlib_figure(fig, provenance: Optional[Dict[str,Any]]=None, **updates: Any):
    data=normalize_figure_provenance(provenance,**updates)
    setattr(fig,"ki_turb_hit_provenance",data)
    return fig

def provenance_json(value: Optional[Dict[str,Any]]=None, **updates: Any)->str:
    return json.dumps(normalize_figure_provenance(value,**updates),indent=2,default=str)
