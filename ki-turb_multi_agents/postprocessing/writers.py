"""Write validated HIT products and register complete checksum-backed manifest entries."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
from typing import List
import numpy as np
from schemas import DatasetFile,DatasetManifest
SIM_TAG='data1'
def _sha(path):
    h=hashlib.sha256();h.update(path.read_bytes());return 'sha256:'+h.hexdigest()
def _register(m,path,kind,fmt,step=None,*,variable=None,source_steps=None,metadata=None):
    m.add_file(DatasetFile(path=(str(path.relative_to(Path(m.base_dir))) if path.is_relative_to(Path(m.base_dir)) else str(path)),kind=kind,variable=variable,format=fmt,time_step=step,size_bytes=path.stat().st_size,checksum=_sha(path),complete=True,source_steps=source_steps or ([step] if step is not None else []),metadata=metadata or {}))
def write_spectra(m,items,out):
    d=out/'spectra';d.mkdir(parents=True,exist_ok=True)
    for q in items:
        step=q['step'];p=d/f'spectrum_{SIM_TAG}_{step}.dat';np.savetxt(p,np.column_stack([q['k'],q['E']]),header='k E(k); Parseval shell sum');_register(m,p,'energy_spectrum','dat',step,metadata={'normalization':q.get('normalization'),'trustworthy_k_max':q.get('trustworthy_k_max')})
        n=d/f'norm_{SIM_TAG}_{step}.dat';np.savetxt(n,np.column_stack([q['k_eta'],q['E'],q['compensated']]),header='k_eta E compensated_k53E');_register(m,n,'normalized_spectrum','dat',step)
def write_isotropy(m,items,out):
    d=out/'isotropy';d.mkdir(parents=True,exist_ok=True)
    for q in items:
        p=d/f'isotropy_coeff_{SIM_TAG}_{q["step"]}.dat'
        np.savetxt(
            p,
            q['columns'],
            header=(
                'Isotropy Coefficients: Standard IC = E22/E11 (≈1 for isotropic components)\n'
                'Derivative-based IC = [2E22 - k_phys dE11/dk] / [2E11]\n'
                'Columns: k, E11(k), E22(k), E33(k), dE11/dk, IC_standard, IC_derivative'
            ),
        )
        _register(m,p,'spectral_isotropy','dat',q['step'])
def write_flatness(m,items,out):
    d=out/'flatness';d.mkdir(parents=True,exist_ok=True)
    for q in items:
        p=d/f'flatness_{SIM_TAG}_t{q["step"]}.txt';np.savetxt(p,np.column_stack([q['r'],q['flatness']]),header='r flatness');_register(m,p,'flatness','txt',q['step'])
def write_structure_functions(m,items,out):
    d=out/'structure_functions';d.mkdir(parents=True,exist_ok=True)
    for q in items:
        orders=q['orders'];cols=[q['r']]+[q['longitudinal'][str(p)] for p in orders]+[q['transverse'][str(p)] for p in orders]+[q['signed_longitudinal_third']];p=d/f'structure_functions1_t{q["step"]}.txt';np.savetxt(p,np.column_stack(cols),header='r '+' '.join(f'SL{n}' for n in orders)+' '+' '.join(f'ST{n}' for n in orders)+' signed_SL3');_register(m,p,'structure_functions','txt',q['step'])
def write_stats(m,stats,spectra,out,max_step=0,energy_balance=None):
    d=out/'stats';d.mkdir(parents=True,exist_ok=True);by_step={x['step']:x for x in spectra}
    ts=d/f'turbulence_stats_{SIM_TAG}.csv';cols=['iter','time','TKE','u_rms','eps_real','eps_spectral','taylor_lambda','re_lambda','eta','kmax_eta','divergence_rms','II_b','III_b','lumley_realizable']
    with ts.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();[w.writerow({k:r.get(k if k!='iter' else 'iter') if k!='eps_spectral' else by_step.get(r['iter'],{}).get('epsilon_spectral') for k in cols}) for r in stats]
    _register(m,ts,'turbulence_stats','csv',source_steps=[r['iter'] for r in stats])
    epsp=d/f'eps_real_validation_{SIM_TAG}.csv';ecols=['iter','iter_norm','eps_real','eps_spectral','TKE_real','u_rms_real','energy_balance_residual','energy_balance_relative_error','eta','kmax_eta','frac_x','frac_y','frac_z']
    residual=(energy_balance or {}).get('residual',[]);relative=(energy_balance or {}).get('relative_error',[])
    with epsp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ecols);w.writeheader()
        for i,r in enumerate(stats):
            total=r['R11']+r['R22']+r['R33'] or 1.;w.writerow({'iter':r['iter'],'iter_norm':r['iter']/max_step if max_step else 0,'eps_real':r['eps_real'],'eps_spectral':by_step.get(r['iter'],{}).get('epsilon_spectral'),'TKE_real':r['TKE'],'u_rms_real':r['u_rms'],'energy_balance_residual':residual[i] if i<len(residual) else None,'energy_balance_relative_error':relative[i] if i<len(relative) else None,'eta':r['eta'],'kmax_eta':r['kmax_eta'],'frac_x':r['R11']/total,'frac_y':r['R22']/total,'frac_z':r['R33']/total})
    _register(m,epsp,'dissipation_validation','csv',source_steps=[r['iter'] for r in stats])
    rp=d/f'reynolds_stress_validation_{SIM_TAG}.csv';rcols=['iter','R11','R22','R33','R12','R13','R23','b11','b22','b33','b12','b13','b23','II_b','III_b','lumley_realizable']
    with rp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rcols);w.writeheader();[w.writerow({k:r.get(k) for k in rcols}) for r in stats]
    _register(m,rp,'reynolds_stress','csv',source_steps=[r['iter'] for r in stats])
def write_pdfs(m,items,out):
    d=out/'pdfs';d.mkdir(parents=True,exist_ok=True)
    for q in items:
        for prefix,x,y,kind in [('velocity',q['velocity_bin'],q['velocity_pdf'],'velocity_pdf'),('gradient',q['gradient_bin'],q['gradient_pdf'],'gradient_pdf')]:
            p=d/f'{prefix}_pdf_{SIM_TAG}_{q["step"]}.dat';np.savetxt(p,np.column_stack([x,y]),header='bin_center pdf');_register(m,p,kind,'dat',q['step'])
        if 'dissipation_bin' in q:
            p=d/f'dissipation_pdf_{SIM_TAG}_{q["step"]}.dat';np.savetxt(p,np.column_stack([q['dissipation_bin'],q['dissipation_pdf']]),header='bin_center pdf');_register(m,p,'dissipation_pdf','dat',q['step'])
def write_kiturb_outputs(m,products,processed_dir,max_step=0):
    out=Path(processed_dir);out.mkdir(parents=True,exist_ok=True)
    if products.get('spectra'):write_spectra(m,products['spectra'],out)
    if products.get('spectral_isotropy'):write_isotropy(m,products['spectral_isotropy'],out)
    if products.get('flatness'):write_flatness(m,products['flatness'],out)
    if products.get('structure_functions'):write_structure_functions(m,products['structure_functions'],out)
    if products.get('real_stats'):write_stats(m,products['real_stats'],products.get('spectra',[]),out,max_step,products.get('energy_balance'))
    if products.get('pdfs'):write_pdfs(m,products['pdfs'],out)
    m.postprocessing['status']=products.get('validation_status','completed');manifest_path=Path(m.base_dir)/'dataset_manifest.json';manifest_path.write_text(m.to_json());return m
