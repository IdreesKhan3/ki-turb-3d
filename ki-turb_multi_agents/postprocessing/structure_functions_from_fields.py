"""Periodic longitudinal/transverse structure functions in all directions."""
from __future__ import annotations
from typing import List,Sequence
import numpy as np
from .readers import VelocitySnapshot
DEFAULT_ORDERS=(2,3,4,5,6)
def compute_structure_functions(snapshots:List[VelocitySnapshot],orders:Sequence[int]=DEFAULT_ORDERS,max_separation:int=0)->List[dict]:
    out=[]
    for s in snapshots:
        rmax=max_separation or min(s.velocity.shape[:3])//2;spacing=s.spacing or (s.dx,)*3;L={p:np.zeros(rmax) for p in orders};T={p:np.zeros(rmax) for p in orders};signed3=np.zeros(rmax)
        for ri,r in enumerate(range(1,rmax+1)):
            longitudinal=[];transverse=[]
            for axis in range(3):
                du=np.roll(s.velocity,-r,axis=axis)-s.velocity;longitudinal.append(du[...,axis]);transverse.extend(du[...,j] for j in range(3) if j!=axis)
            signed3[ri]=np.mean([np.mean(x**3) for x in longitudinal])
            for p in orders:
                L[p][ri]=np.mean([np.mean(np.abs(x)**p) for x in longitudinal]);T[p][ri]=np.mean([np.mean(np.abs(x)**p) for x in transverse])
        rvals=np.arange(1,rmax+1)*float(np.mean(spacing));out.append({"step":s.step,"time":s.time,"orders":list(orders),"r":rvals,"longitudinal":{str(p):L[p] for p in orders},"transverse":{str(p):T[p] for p in orders},"signed_longitudinal_third":signed3,"S":np.column_stack([L[p] for p in orders])})
    return out
