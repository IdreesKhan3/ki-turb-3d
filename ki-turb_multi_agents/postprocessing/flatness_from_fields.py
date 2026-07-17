"""Direction-averaged periodic velocity-increment flatness."""
from __future__ import annotations
from typing import List
import numpy as np
from .readers import VelocitySnapshot
def compute_flatness(snapshots:List[VelocitySnapshot],max_separation:int=0)->List[dict]:
    out=[]
    for s in snapshots:
        rmax=max_separation or min(s.velocity.shape[:3])//2;vals=np.zeros(rmax)
        for i,r in enumerate(range(1,rmax+1)):
            f=[]
            for axis in range(3):
                du=np.roll(s.velocity[...,axis],-r,axis=axis)-s.velocity[...,axis];m2=np.mean(du**2);f.append(np.mean(du**4)/m2**2 if m2>0 else np.nan)
            vals[i]=np.nanmean(f)
        out.append({"step":s.step,"time":s.time,"r":np.arange(1,rmax+1)*float(np.mean(s.spacing or (s.dx,)*3)),"flatness":vals})
    return out
