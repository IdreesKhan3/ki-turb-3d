"""Energy-spectrum averaging and validation helpers for periodic HIT.

The module preserves KI-TURB's original public averaging functions while adding
stationary-window selection, compensated spectra, trustworthy shell filtering,
and uncertainty estimates.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Sequence, Dict, Any
import numpy as np


def _common_grid(data_list):
    if not data_list:
        return None, []
    k0 = np.asarray(data_list[0][0], float)
    valid=[]
    for row in data_list:
        k=np.asarray(row[0],float)
        if k.shape==k0.shape and np.allclose(k,k0,rtol=1e-10,atol=1e-12):
            valid.append(row)
    return k0, valid


def compute_spectrum_time_avg(data_list: List[Tuple[np.ndarray,np.ndarray]]):
    k,valid=_common_grid(data_list)
    if k is None or not valid:return None,None,None
    values=np.stack([np.asarray(row[1],float) for row in valid])
    return k,np.mean(values,axis=0),np.std(values,axis=0,ddof=1 if len(valid)>1 else 0)


def compute_spectrum_time_avg_norm(data_list: List[Tuple[np.ndarray,np.ndarray,np.ndarray]]):
    k,valid=_common_grid(data_list)
    if k is None or not valid:return None,None,None,None
    en=np.stack([np.asarray(row[1],float) for row in valid]); ep=np.stack([np.asarray(row[2],float) for row in valid])
    return k,np.mean(en,axis=0),np.std(en,axis=0,ddof=1 if len(valid)>1 else 0),np.mean(ep,axis=0)


def trustworthy_shell_mask(k: Sequence[float], shape: Sequence[int], spacing: Sequence[float], *, fraction: float=2.0/3.0):
    """Return shells below a conservative fraction of the smallest Nyquist limit."""
    kval=np.asarray(k,float); ny=min(np.pi/float(d) for d in spacing)
    return (kval>0) & (kval<=fraction*ny)


def compensated_spectrum(k: Sequence[float], energy: Sequence[float], dissipation: float, *, exponent: float=5.0/3.0):
    k=np.asarray(k,float); e=np.asarray(energy,float)
    if dissipation<=0:raise ValueError("dissipation must be positive")
    result=np.full_like(e,np.nan,dtype=float); mask=k>0
    result[mask]=e[mask]*k[mask]**exponent/dissipation**(2.0/3.0)
    return result


def stationary_spectrum_summary(data_list, *, start_index: int=0, end_index: Optional[int]=None, confidence_z: float=1.96) -> Dict[str,Any]:
    selected=data_list[start_index:end_index]
    k,mean,std=compute_spectrum_time_avg(selected)
    if k is None:return {"status":"insufficient_data","sample_count":0}
    n=len(selected); stderr=std/np.sqrt(max(n,1))
    return {"status":"ok","sample_count":n,"k":k,"mean":mean,"std":std,"confidence_low":mean-confidence_z*stderr,"confidence_high":mean+confidence_z*stderr}


__all__=["compute_spectrum_time_avg","compute_spectrum_time_avg_norm","trustworthy_shell_mask","compensated_spectrum","stationary_spectrum_summary"]
