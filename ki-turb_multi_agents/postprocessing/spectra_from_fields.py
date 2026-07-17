"""Physically normalized spectra from periodic HIT velocity fields."""
from __future__ import annotations
from typing import List,Optional
import numpy as np
from ._spectral import component_spectra
from .readers import VelocitySnapshot
def compute_energy_spectrum(snapshots:List[VelocitySnapshot],viscosity:Optional[float]=None)->List[dict]:
    results=[]
    for snap in snapshots:
        spacing=snap.spacing or (snap.dx,)*3;spec=component_spectra(snap.velocity,spacing)
        k,E=spec['k'],spec['E'];nu=float(viscosity or (snap.metadata or {}).get('viscosity') or 0)
        eps_spectral=2*nu*float(np.sum(k*k*E)) if nu>0 else None
        eta=(nu**3/eps_spectral)**.25 if eps_spectral and eps_spectral>0 else None
        k_eta=k*eta if eta else np.full_like(k,np.nan);comp=E*np.power(k,5/3,where=k>0,out=np.zeros_like(k))
        mask=(k>0)&(k<=spec['trustworthy_k_max'])&(E>0);slope=None
        idx=np.where(mask)[0]
        if idx.size>=6:
            lo=idx[max(0,idx.size//5)];hi=idx[max(lo+3,4*idx.size//5)]
            if hi>lo:slope=float(np.polyfit(np.log(k[lo:hi]),np.log(E[lo:hi]),1)[0])
        results.append({"step":snap.step,"time":snap.time,"k":k,"E":E,"E11":spec['E11'],"E22":spec['E22'],"E33":spec['E33'],"k_eta":k_eta,"compensated":comp,"epsilon_spectral":eps_spectral,"eta":eta,"inertial_slope":slope,"trustworthy_k_max":spec['trustworthy_k_max'],"normalization":spec['normalization']})
    return results
def average_stationary_spectra(records:List[dict])->dict:
    if not records:raise ValueError('no spectra to average')
    k=np.asarray(records[0]['k']);arr=np.stack([np.asarray(r['E']) for r in records if np.asarray(r['E']).shape==k.shape]);return {"k":k,"E_mean":arr.mean(0),"E_std":arr.std(0,ddof=1) if len(arr)>1 else np.zeros_like(k),"samples":len(arr),"source_steps":[r['step'] for r in records]}
