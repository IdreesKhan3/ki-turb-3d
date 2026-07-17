"""Periodic real-space HIT statistics, stresses, dissipation and anisotropy."""
from __future__ import annotations
from typing import List,Optional
import numpy as np
from .readers import VelocitySnapshot
from .periodic_derivatives import velocity_gradient,strain_rate_tensor,divergence
def compute_real_turbulence_stats(snapshots:List[VelocitySnapshot],viscosity:Optional[float]=None)->List[dict]:
    if viscosity is None or viscosity<=0:raise ValueError('positive viscosity is required for dissipation and Re_lambda')
    nu=float(viscosity);rows=[]
    for s in snapshots:
        u=s.velocity-np.mean(s.velocity,axis=(0,1,2),keepdims=True);flat=u.reshape(-1,3);R=(flat.T@flat)/flat.shape[0];tke=.5*float(np.trace(R));urms=float(np.sqrt(2*tke/3))
        S=strain_rate_tensor(s.velocity,s.spacing or s.dx,method='spectral');eps_real=2*nu*float(np.mean(np.einsum('...ij,...ij->...',S,S)))
        lam=float(np.sqrt(15*nu*urms**2/eps_real)) if eps_real>0 else 0.;rel=urms*lam/nu;eta=(nu**3/eps_real)**.25 if eps_real>0 else np.inf
        kmax=min(np.pi/d for d in (s.spacing or (s.dx,)*3));bij=R/(2*tke)-np.eye(3)/3 if tke>0 else np.zeros((3,3));ii=-.5*float(np.trace(bij@bij));iii=float(np.linalg.det(bij));eig=np.linalg.eigvalsh(bij);realizable=bool(np.all(eig>=-1/3-1e-10) and np.all(eig<=2/3+1e-10))
        div=divergence(s.velocity,s.spacing or s.dx,method='spectral')
        rows.append({"iter":s.step,"time":s.time,"u_rms":urms,"TKE":tke,"eps_real":eps_real,"eps":eps_real,"taylor_lambda":lam,"re_lambda":rel,"eta":eta,"kmax_eta":kmax*eta,"viscosity":nu,"divergence_rms":float(np.sqrt(np.mean(div**2))),"R11":float(R[0,0]),"R22":float(R[1,1]),"R33":float(R[2,2]),"R12":float(R[0,1]),"R13":float(R[0,2]),"R23":float(R[1,2]),"b11":float(bij[0,0]),"b22":float(bij[1,1]),"b33":float(bij[2,2]),"b12":float(bij[0,1]),"b13":float(bij[0,2]),"b23":float(bij[1,2]),"II_b":ii,"III_b":iii,"lumley_realizable":realizable})
    return rows
