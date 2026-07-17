"""Manifest/product-driven HIT visualization with per-figure provenance."""
from __future__ import annotations
import html,json
from pathlib import Path
from typing import Dict,List,Optional,Any
import numpy as np
from pydantic import BaseModel,ConfigDict,Field
from schemas.hit_analysis_products import HITAnalysisProducts
from visualizations.provenance import normalize_figure_provenance

class VisualArtifact(BaseModel):
    model_config=ConfigDict(extra="allow")
    name:str;path:str;kind:str="figure";metadata_path:Optional[str]=None
class VisualizationResult(BaseModel):
    model_config=ConfigDict(extra="allow")
    output_dir:str;artifacts:List[VisualArtifact]=Field(default_factory=list);dashboard:Optional[str]=None;warnings:List[str]=Field(default_factory=list)

class HITVisualizationAgent:
    def generate(self,products:HITAnalysisProducts,output_dir:str|Path,*,image_format:str="png",dpi:int=160,manifest_path:Optional[str|Path]=None)->VisualizationResult:
        try:import matplotlib.pyplot as plt
        except ImportError as exc:raise RuntimeError("matplotlib is required") from exc
        target=Path(output_dir).expanduser().resolve();target.mkdir(parents=True,exist_ok=True);result=VisualizationResult(output_dir=str(target))
        def provenance(base:Optional[Dict[str,Any]]=None,**extra):
            d=dict(base or {});d.update(extra);d.setdefault("run_id",products.run_id);d.setdefault("validation_status",products.validation_status);return normalize_figure_provenance(d)
        def save(name:str,meta:Dict[str,Any]):
            fig=plt.gcf();fig.tight_layout();path=target/f"{name}.{image_format}";fig.savefig(path,dpi=dpi);plt.close(fig)
            mp=target/f"{name}.json";mp.write_text(json.dumps(provenance(meta),indent=2,default=str),encoding="utf-8");result.artifacts.append(VisualArtifact(name=name,path=str(path),metadata_path=str(mp)))
        h=products.time_history
        if h and h.time:
            base=h.provenance.model_dump(mode="json");base["physical_time_range"]=[h.time[0],h.time[-1]];base["source_snapshots"]=h.step
            def line(name,ys,labels,ylabel,title):
                if not any(len(y)==len(h.time) and len(y)>0 for y in ys):return
                plt.figure()
                for y,label in zip(ys,labels):
                    if len(y)==len(h.time):plt.plot(h.time,y,label=label)
                if len(labels)>1:plt.legend()
                plt.xlabel("Physical time");plt.ylabel(ylabel);plt.title(title);save(name,base)
            line("tke_history",[h.tke],["TKE"],"TKE","Turbulent kinetic energy")
            line("energy_input_dissipation",[h.forcing_power,h.dissipation],["Forcing power","Dissipation"],"Power per unit mass","Energy input and dissipation")
            line("re_lambda_history",[h.re_lambda],[r"$Re_\lambda$"],r"$Re_\lambda$","Taylor Reynolds number")
            line("mach_density_health",[h.mach_max,h.density_min,h.density_max],["Maximum Mach","Minimum density","Maximum density"],"Value","Simulation health")
            line("kmax_eta_history",[h.kmax_eta],[r"$k_{max}\eta$"],r"$k_{max}\eta$","Resolution adequacy")
        if products.spectra:
            steps=[x.step for x in products.spectra];times=[x.time for x in products.spectra if x.time is not None];meta={"source_snapshots":steps,"physical_time_range":[min(times),max(times)] if times else None,"analysis_method_version":"spectral-shell-v2","normalization":"Parseval shell sum","units":{"x":"1/length","y":"velocity^2 length"}}
            plt.figure()
            for x in products.spectra:plt.loglog(x.wavenumber,x.energy,alpha=.35)
            plt.xlabel("Wavenumber");plt.ylabel("E(k)");plt.title("Energy spectra");save("energy_spectra",meta)
            comp=[x for x in products.spectra if x.compensated_energy]
            if comp:
                plt.figure()
                for x in comp:plt.semilogx(x.wavenumber,x.compensated_energy,alpha=.35)
                plt.xlabel("Wavenumber");plt.ylabel(r"$E(k)k^{5/3}\epsilon^{-2/3}$");plt.title("Compensated spectra");save("compensated_spectra",meta)
        if products.spectral_isotropy:
            x=products.spectral_isotropy[-1];meta=x.provenance.model_dump(mode="json");meta["source_snapshots"]=[x.step]
            plt.figure();plt.loglog(x.wavenumber,x.e11,label="E11");plt.loglog(x.wavenumber,x.e22,label="E22");plt.loglog(x.wavenumber,x.e33,label="E33");plt.xlabel("Wavenumber");plt.ylabel("Component shell energy");plt.title("Component isotropy (target equality)");plt.legend();save("spectral_isotropy",meta)
        if products.reynolds_stress:
            rows=products.reynolds_stress;steps=[x.step for x in rows];meta=rows[-1].provenance.model_dump(mode="json");meta["source_snapshots"]=steps
            x=rows[-1];plt.figure();plt.bar(["R11","R22","R33","R12","R13","R23"],[x.r11,x.r22,x.r33,x.r12,x.r13,x.r23]);plt.ylabel("Reynolds stress");plt.title("Reynolds-stress components");save("reynolds_stress",meta)
            plt.figure();b=np.array([[x.b11 or 0,x.b12 or 0,x.b13 or 0],[x.b12 or 0,x.b22 or 0,x.b23 or 0],[x.b13 or 0,x.b23 or 0,x.b33 or 0]]);im=plt.imshow(b);plt.colorbar(im,label=r"$b_{ij}$");plt.xticks(range(3),["x","y","z"]);plt.yticks(range(3),["x","y","z"]);plt.title("Reynolds-stress anisotropy tensor");save("anisotropy_tensor",meta)
            good=[r for r in rows if r.invariant_ii is not None and r.invariant_iii is not None]
            if good:
                plt.figure();plt.plot([r.invariant_iii for r in good],[r.invariant_ii for r in good],marker="o");plt.xlabel(r"$III_b$");plt.ylabel(r"$II_b$");plt.title("Lumley-invariant trajectory");save("lumley_invariants",meta)
        for i,x in enumerate(products.pdfs):
            plt.figure();plt.semilogy(x.bin_center,x.density);plt.xlabel(x.variable);plt.ylabel("Probability density");plt.title(f"PDF of {x.variable}");save(f"pdf_{i:03d}_{self._safe(x.variable)}",x.provenance.model_dump(mode="json"))
        if products.structure_functions:
            x=products.structure_functions[-1];plt.figure()
            for order in x.orders:
                y=x.longitudinal.get(str(order));
                if y:plt.loglog(x.separation,np.abs(y),label=f"S{order}")
            plt.xlabel("Separation");plt.ylabel("Longitudinal structure function");plt.title("Structure functions");plt.legend();save("structure_functions",x.provenance.model_dump(mode="json"))
            if x.signed_longitudinal_third:
                plt.figure();plt.plot(x.separation,x.signed_longitudinal_third);plt.xlabel("Separation");plt.ylabel(r"$\langle(\delta u_L)^3\rangle$");plt.title("Signed third-order longitudinal structure function");save("signed_third_order",x.provenance.model_dump(mode="json"))
        if products.flatness:
            x=products.flatness[-1];plt.figure();plt.semilogx(x.separation,x.flatness);plt.xlabel("Separation");plt.ylabel("Flatness");plt.title("Increment flatness");save("flatness",x.provenance.model_dump(mode="json"))
        if manifest_path:
            try:self._field_figures(Path(manifest_path),target,result,products,image_format,dpi,save)
            except Exception as exc:result.warnings.append(f"volume-field visualization unavailable: {exc}")
        else:result.warnings.append("volume-field figures skipped: no authoritative manifest_path supplied")
        dashboard=target/"index.html";dashboard.write_text(self._dashboard(result.artifacts,result.warnings,products),encoding="utf-8");result.dashboard=str(dashboard);return result
    def _field_figures(self,manifest,target,result,products,fmt,dpi,save):
        import matplotlib.pyplot as plt
        from postprocessing.readers import load_velocity_snapshots
        from schemas import DatasetManifest
        from postprocessing.periodic_derivatives import vorticity,q_criterion,velocity_gradient_tensor
        manifest_obj=DatasetManifest.from_json(Path(manifest).read_text(encoding="utf-8"))
        snaps=load_velocity_snapshots(manifest_obj)
        if not snaps:raise ValueError("no complete velocity snapshots")
        s=snaps[-1];u=s.velocity;spacing=s.spacing or (s.dx,)*3;mid=u.shape[2]//2;speed=np.linalg.norm(u,axis=-1);omega=np.linalg.norm(vorticity(u,spacing),axis=-1);q=q_criterion(u,spacing)
        grad=velocity_gradient_tensor(u,spacing);sym=.5*(grad+np.swapaxes(grad,-1,-2));eig=np.linalg.eigvalsh(sym);lambda2=eig[...,1]
        meta={"source_snapshots":[s.step],"physical_time_range":[s.time,s.time],"analysis_method_version":"spectral-derivatives-v2","normalization":"physical field","units":{"velocity":"manifest units"}}
        for name,field,title in [("velocity_magnitude",speed,"Velocity magnitude"),("vorticity_magnitude",omega,"Vorticity magnitude"),("q_criterion",q,"Q criterion"),("lambda2",lambda2,"Lambda2")]:
            plt.figure();im=plt.imshow(field[:,:,mid].T,origin="lower",aspect="equal");plt.colorbar(im);plt.xlabel("x index");plt.ylabel("y index");plt.title(f"{title}, central z-plane");save(name,meta)
    @staticmethod
    def _safe(v):return "".join(c if c.isalnum() else "_" for c in v.lower()).strip("_")
    @staticmethod
    def _dashboard(arts,warnings,products):
        cards="".join(f"<section><h2>{html.escape(a.name.replace('_',' ').title())}</h2><img src='{html.escape(Path(a.path).name)}'><p><a href='{html.escape(Path(a.metadata_path).name)}'>provenance</a></p></section>" for a in arts)
        warn="".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        return "<!doctype html><html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1200px;margin:auto}img{max-width:100%}section{margin:2rem 0}</style></head><body>"+f"<h1>KI-TURB HIT report figures</h1><p>Run {html.escape(products.run_id or 'unknown')}</p><ul>{warn}</ul>"+cards+"</body></html>"
__all__=["VisualArtifact","VisualizationResult","HITVisualizationAgent"]
