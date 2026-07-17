"""Velocity, velocity-gradient and dissipation PDFs using periodic derivatives."""
from __future__ import annotations
from typing import List,Optional
import numpy as np
from .readers import VelocitySnapshot
from .periodic_derivatives import velocity_gradient,strain_rate_tensor
def _pdf(x,bins=101):
    x=np.asarray(x,float).ravel();x=x[np.isfinite(x)];h,e=np.histogram(x,bins=bins,density=True);return .5*(e[:-1]+e[1:]),h
def compute_pdfs(snapshots:List[VelocitySnapshot],bins:int=101,viscosity:Optional[float]=None)->List[dict]:
    out=[]
    for s in snapshots:
        u=s.velocity-np.mean(s.velocity,axis=(0,1,2),keepdims=True);std=np.std(u);uc,h=_pdf(u/std if std else u,bins);g=velocity_gradient(s.velocity,s.spacing or s.dx,method='spectral');gstd=np.std(g);gc,gh=_pdf(g/gstd if gstd else g,bins)
        rec={"step":s.step,"time":s.time,"velocity_bin":uc,"velocity_pdf":h,"gradient_bin":gc,"gradient_pdf":gh}
        if viscosity and viscosity>0:
            S=strain_rate_tensor(s.velocity,s.spacing or s.dx,method='spectral');eps=2*viscosity*np.einsum('...ij,...ij->...',S,S);ec,eh=_pdf(eps/np.mean(eps) if np.mean(eps)>0 else eps,bins);rec.update(dissipation_bin=ec,dissipation_pdf=eh)
        out.append(rec)
    return out
