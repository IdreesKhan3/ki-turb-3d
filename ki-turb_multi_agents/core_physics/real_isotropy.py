"""Reynolds-stress anisotropy and realizability-correct Lumley invariants."""
import numpy as np,pandas as pd
from pathlib import Path
def load_turbulence_data(csv_path:Path):
    df=pd.read_csv(csv_path);num=lambda *names: next((pd.to_numeric(df[n],errors='coerce').to_numpy() for n in names if n in df),np.zeros(len(df)))
    tke=num('TKE_real','TKE');return {"iter":num('iter'),"iter_norm":num('iter_norm','iter'),"TKE":tke,"u_rms":num('u_rms_real','u_rms'),"eps0":num('eps_real','eps'),"frac_x":num('frac_x','E_x'),"frac_y":num('frac_y','E_y'),"frac_z":num('frac_z','E_z')}
def compute_reynolds_from_fractions(turb):
    k=turb['TKE'];z=np.zeros(len(k));return {"R11":2*k*turb['frac_x'],"R22":2*k*turb['frac_y'],"R33":2*k*turb['frac_z'],"R12":z,"R13":z,"R23":z,"TKE":k}
def load_reynolds_stress(path:Path,turb):
    if not path.exists():return compute_reynolds_from_fractions(turb)
    df=pd.read_csv(path);names=['R11','R22','R33','R12','R13','R23'];a={n:pd.to_numeric(df[n] if n in df else df.iloc[:,i+1],errors='coerce').to_numpy() for i,n in enumerate(names)};a['TKE']=.5*(a['R11']+a['R22']+a['R33']);return a
def anisotropy_tensor(R):
    k=np.maximum(np.asarray(R['TKE'],float),1e-30);return {f'b{i}{j}':np.asarray(R[f'R{i}{j}'])/(2*k)-(1/3 if i==j else 0) for i,j in [(1,1),(2,2),(3,3),(1,2),(1,3),(2,3)]}
def invariants(b):
    n=len(np.asarray(b['b11']));ii=np.empty(n);iii=np.empty(n);real=np.empty(n,dtype=bool)
    for q in range(n):
        B=np.array([[b['b11'][q],b['b12'][q],b['b13'][q]],[b['b12'][q],b['b22'][q],b['b23'][q]],[b['b13'][q],b['b23'][q],b['b33'][q]]]);ii[q]=-.5*np.trace(B@B);iii[q]=np.linalg.det(B);e=np.linalg.eigvalsh(B);real[q]=np.all(e>=-1/3-1e-10)&np.all(e<=2/3+1e-10)
    return {"II_b":ii,"III_b":iii,"anis_index":np.sqrt(np.maximum(-2*ii,0)),"eta":np.sqrt(np.maximum(-ii/3,0)),"xi":np.cbrt(iii/2),"realizable":real}
