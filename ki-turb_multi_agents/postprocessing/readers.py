"""Manifest-driven, variable-aware CFD field readers."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable,List,Optional,Sequence
import numpy as np
from schemas import DatasetManifest,DatasetFile
_STEP_RE=re.compile(r"(\d+)")
@dataclass
class VelocitySnapshot:
    step:int;time:Optional[float];velocity:np.ndarray;dx:float=1.0;spacing:Optional[tuple]=None;metadata:Optional[dict]=None

def _vtk(path,array_name):
    try:
        import pyvista as pv
        grid=pv.read(str(path));names=list(grid.point_data.keys())
        wanted=next((n for n in names if n.lower()==array_name.lower()),None)
        if wanted is None:raise ValueError(f"array '{array_name}' not found in {path}; arrays={names}")
        arr=np.asarray(grid.point_data[wanted]);dims=tuple(int(x) for x in getattr(grid,"dimensions",(0,0,0)))
        if len(dims)!=3 or 0 in dims:raise ValueError(f"cannot infer structured dimensions for {path}")
        comp=1 if arr.ndim==1 else arr.shape[1];arr=arr.reshape((*dims,comp),order="F")
        return arr,getattr(grid,"spacing",(1,1,1))
    except ImportError:
        from data_readers.vti_reader import read_vti_file
        if path.suffix.lower()!='.vti':raise RuntimeError("pyvista is required for parallel/multiblock VTK")
        data=read_vti_file(str(path));key=next((k for k in data if k.lower()==array_name.lower()),None)
        if key is None:raise ValueError(f"array '{array_name}' not found in {path}")
        return np.asarray(data[key]),(1,1,1)
def _hdf5(path,array_name,fortran_order=False):
    import h5py
    with h5py.File(path,'r') as h:
        candidates=[];h.visititems(lambda n,o:candidates.append(n) if isinstance(o,h5py.Dataset) else None)
        key=next((n for n in candidates if n.split('/')[-1].lower()==array_name.lower()),None)
        if key is None:raise ValueError(f"dataset '{array_name}' not found in {path}; datasets={candidates}")
        a=np.asarray(h[key]);spacing=tuple(h[key].attrs.get('spacing',(1,1,1)))
    if fortran_order and a.ndim>=3:a=np.transpose(a,tuple(reversed(range(a.ndim-1)))+(a.ndim-1,))
    return a,spacing
def read_field(path,array_name,fortran_order=False):
    path=Path(path);s=path.suffix.lower()
    if s in {'.h5','.hdf5'}:return _hdf5(path,array_name,fortran_order)
    if s in {'.vti','.pvti','.vtu','.pvtu','.vtm'}:return _vtk(path,array_name)
    if s=='.npy':return np.load(path),(1,1,1)
    raise ValueError(f"unsupported field format: {s}")
def load_velocity_snapshots(files:Iterable[Path]|DatasetManifest,dx:float=1.0,fortran_order:bool=False)->List[VelocitySnapshot]:
    entries=[]
    if isinstance(files,DatasetManifest):
        base=Path(files.base_dir);entries=[(base/f.path,f) for f in files.files if f.kind=='velocity_field' and f.complete]
    else:entries=[(Path(p),None) for p in files]
    out=[]
    for path,item in entries:
        if not path.is_file():continue
        velocity,spacing=read_field(path,'velocity',fortran_order)
        if velocity.ndim!=4 or velocity.shape[-1]!=3:raise ValueError(f"velocity field must be (nx,ny,nz,3): {path} -> {velocity.shape}")
        matches=_STEP_RE.findall(path.stem);step=item.time_step if item and item.time_step is not None else int(matches[-1]) if matches else 0
        time=item.time_value if item else None;sp=tuple(float(x) for x in (item.spacing if item and item.spacing else spacing));out.append(VelocitySnapshot(step,time,np.ascontiguousarray(velocity,float),sp[0] if sp else dx,sp,{"source":str(path),"manifest":item.model_dump() if item else None}))
    return sorted(out,key=lambda x:x.step)
