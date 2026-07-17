"""Reproducible, cached, isolated OpenLB compilation controller."""
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict,List,Optional
from pydantic import BaseModel,ConfigDict,Field
from integrations.openlb.provenance import OpenLBProvenanceCollector,ProvenanceRecord
from schemas.openlb_hit import BuildProfile
class ToolchainInfo(BaseModel):
    model_config=ConfigDict(extra="allow")
    cxx:Optional[str]=None;mpicxx:Optional[str]=None;mpirun:Optional[str]=None;make:Optional[str]=None;nvcc:Optional[str]=None;hipcc:Optional[str]=None;cpu_count:Optional[int]=None
class CompileResult(BaseModel):
    model_config=ConfigDict(extra="allow")
    success:bool;profile:BuildProfile;command:List[str]=Field(default_factory=list);return_code:Optional[int]=None;log_path:str;executable:Optional[str]=None;executable_sha256:Optional[str]=None;source_fingerprint:Optional[str]=None;diagnostics:List[str]=Field(default_factory=list);provenance:Optional[ProvenanceRecord]=None;cache_hit:bool=False
class OpenLBCompileAgent:
    def __init__(self,make_program=None):self.make_program=make_program or shutil.which("make") or "make";self.provenance_collector=OpenLBProvenanceCollector()
    def detect_toolchain(self):return ToolchainInfo(cxx=shutil.which(os.getenv("CXX","g++")),mpicxx=shutil.which("mpicxx") or shutil.which("mpiCC"),mpirun=shutil.which("mpirun") or shutil.which("mpiexec"),make=shutil.which(self.make_program),nvcc=shutil.which("nvcc"),hipcc=shutil.which("hipcc"),cpu_count=os.cpu_count())
    def compile(self,app_dir,artifact_dir,*,app_name="kiTurbHIT3D",profile=BuildProfile.SERIAL,jobs=None,clean=False,extra_make_args=None,environment=None,smoke_test_args=None):
        profile=BuildProfile(profile);source=Path(app_dir).resolve();art=Path(artifact_dir).resolve();art.mkdir(parents=True,exist_ok=True);log=art/"compile.log";tools=self.detect_toolchain();missing=self._missing(profile,tools)
        if missing:return CompileResult(success=False,profile=profile,log_path=str(log),diagnostics=missing)
        fingerprint=self.source_fingerprint(source,self._find_root(source));cache_key=hashlib.sha256((fingerprint+profile.value+json.dumps(environment or {},sort_keys=True)).encode()).hexdigest()[:20];cache=art/"cache"/cache_key;cached=cache/app_name
        output=art/app_name
        if cached.is_file():shutil.copy2(cached,output);output.chmod(output.stat().st_mode|0o111);return self._result(True,profile,log,output,fingerprint,[],[],True,source)
        build_dir=art/"build"/cache_key;build_dir.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ);env.update(self._profile_environment(profile));env.update(environment or {})
        command=[self.make_program,f"-j{jobs or max(1,min(tools.cpu_count or 1,16))}",f"BUILD_DIR={build_dir}",f"OUTPUT_PATH={output}",f"KITURB_PROFILE={profile.value}",*(extra_make_args or [])]
        lock=source/".kiturb-build.lock"
        with self._lock(lock):
            with log.open("w") as h:
                h.write(json.dumps({"profile":profile.value,"environment":{k:env[k] for k in env if k in {"CXX","CXXFLAGS","PARALLEL_MODE"}},"fingerprint":fingerprint},indent=2)+"\n")
                if clean:subprocess.run([self.make_program,"clean",f"BUILD_DIR={build_dir}",f"OUTPUT_PATH={output}"],cwd=source,env=env,stdout=h,stderr=subprocess.STDOUT,text=True)
                proc=subprocess.run(command,cwd=source,env=env,stdout=h,stderr=subprocess.STDOUT,text=True)
        diagnostics=[]
        if proc.returncode!=0 or not output.is_file():diagnostics=self.diagnose_log(log);return CompileResult(success=False,profile=profile,command=command,return_code=proc.returncode,log_path=str(log),source_fingerprint=fingerprint,diagnostics=diagnostics)
        output.chmod(output.stat().st_mode|0o111)
        if smoke_test_args is not None:
            sm=subprocess.run([str(output),*smoke_test_args],cwd=art,capture_output=True,text=True,timeout=60);(art/"smoke_test.log").write_text(sm.stdout+"\n"+sm.stderr)
            if sm.returncode:diagnostics.append(f"smoke test failed with code {sm.returncode}")
        if diagnostics:return CompileResult(success=False,profile=profile,command=command,return_code=0,log_path=str(log),executable=str(output),source_fingerprint=fingerprint,diagnostics=diagnostics)
        cache.mkdir(parents=True,exist_ok=True);shutil.copy2(output,cached)
        return self._result(True,profile,log,output,fingerprint,command,diagnostics,False,source)
    def _result(self,success,profile,log,output,fingerprint,command,diagnostics,cache_hit,source):
        prov=self.provenance_collector.collect(openlb_root=self._find_root(source),app_dir=source,executable=output,compiler=os.getenv("CXX"),build_profile=profile.value,build_command=command,build_flags=os.getenv("CXXFLAGS","").split(),environment_keys=["CXX","CXXFLAGS","OMP_NUM_THREADS"]);(output.parent/"build_provenance.json").write_text(prov.model_dump_json(indent=2));return CompileResult(success=success,profile=profile,command=command,return_code=0,log_path=str(log),executable=str(output),executable_sha256=prov.executable_sha256,source_fingerprint=fingerprint,diagnostics=diagnostics,provenance=prov,cache_hit=cache_hit)
    @staticmethod
    def source_fingerprint(source,openlb_root=None):
        d=hashlib.sha256();source=Path(source);root=Path(openlb_root) if openlb_root else source.parent
        for p in sorted(source.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".cpp",".h",".hh",".hpp",".mk",""} and not p.name.startswith(".kiturb"):
                d.update(str(p.relative_to(source)).encode());d.update(p.read_bytes())
        for rel in ("global.mk","config.mk","rules.mk"):
            q=root/rel
            if q.is_file():d.update(rel.encode());d.update(q.read_bytes())
        try:
            commit=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True,timeout=10).stdout.strip()
            d.update(commit.encode())
        except Exception:pass
        return "sha256:"+d.hexdigest()
    @staticmethod
    def diagnose_log(path):
        t=Path(path).read_text(errors="replace").lower() if Path(path).exists() else ""
        m={"fatal error:":"missing or invalid header/include","undefined reference":"linker error","cannot find -l":"link library missing","no such file":"file/tool missing","killed signal":"compiler killed, likely memory pressure","not supported":"requested dynamics/profile is incompatible"};r=[v for k,v in m.items() if k in t];return r or ["OpenLB compilation failed; inspect compile.log"]
    @staticmethod
    def _missing(profile,t):
        if not t.make:return ["make was not found"]
        if profile in {BuildProfile.CUDA,BuildProfile.HIP}:return [f"{profile.value} is not supported by the bundled legacy OpenLB; use a versioned GPU-capable OpenLB adapter"]
        if profile in {BuildProfile.MPI,BuildProfile.MPI_OPENMP} and not t.mpicxx:return ["MPI compiler wrapper was not found"]
        if not t.cxx and profile not in {BuildProfile.MPI,BuildProfile.MPI_OPENMP}:return ["C++ compiler was not found"]
        return []
    @staticmethod
    def _profile_environment(profile):
        env={}
        if profile == BuildProfile.MPI:
            env["CXX"]=shutil.which("mpicxx") or "mpicxx";env["PARALLEL_MODE"]="MPI"
        elif profile == BuildProfile.MPI_OPENMP:
            env["CXX"]=shutil.which("mpicxx") or "mpicxx";env["PARALLEL_MODE"]="HYBRID"
        elif profile == BuildProfile.OPENMP:
            env["PARALLEL_MODE"]="OMP"
        if profile==BuildProfile.DEBUG:env["KITURB_DEBUG"]="1"
        if profile==BuildProfile.SANITIZER:env["KITURB_SANITIZER"]="1"
        return env
    @staticmethod
    def _find_root(app):
        app=Path(app).resolve()
        defs=app/"definitions.mk"
        if defs.is_file():
            for line in defs.read_text(errors="replace").splitlines():
                s=line.strip()
                if s.startswith("ROOT") and ":=" in s:
                    root=(app/s.split(":=",1)[1].strip()).resolve()
                    if (root/"src").is_dir() and (root/"global.mk").is_file():
                        return root
                    break
        for p in (app,*app.parents):
            if (p/"src").is_dir() and ((p/"examples").is_dir() or (p/"global.mk").is_file()):
                return p
        # SolverApps/<app> sits beside cfd_solvers/openLB
        sibling=app.parent.parent/"openLB"
        if (sibling/"src").is_dir() and (sibling/"global.mk").is_file():
            return sibling.resolve()
        return app.parent
    @staticmethod
    def _stale_lock(path: Path) -> bool:
        """True when the lock file is orphaned (holder PID no longer running)."""
        try:
            pid = int(path.read_text().strip())
        except (OSError, ValueError):
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        return False

    @contextmanager
    def _lock(self,path,timeout=300):
        start=time.monotonic();fd=None
        while fd is None:
            try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
            except FileExistsError:
                if self._stale_lock(Path(path)):
                    try:
                        Path(path).unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic()-start>timeout:raise TimeoutError(f"build lock timeout: {path}")
                time.sleep(.25)
        try:yield
        finally:
            os.close(fd)
            try:Path(path).unlink()
            except FileNotFoundError:pass
