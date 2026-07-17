"""Common execution runners for local, MPI, Docker, Slurm and SSH jobs."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunnerError(RuntimeError):
    pass


class ExecutionHandle(BaseModel):
    model_config = ConfigDict(extra="allow")

    runner: str
    external_id: str
    command: List[str]
    cwd: str
    log_path: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class ExecutionStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    state: str
    return_code: Optional[int] = None
    message: str = ""


class ExecutionRunner(ABC):
    name = "base"

    @abstractmethod
    def submit(
        self,
        command: List[str],
        *,
        cwd: str | Path,
        log_path: str | Path,
        environment: Optional[Dict[str, str]] = None,
    ) -> ExecutionHandle:
        raise NotImplementedError

    @abstractmethod
    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        raise NotImplementedError

    def checkpoint(self, handle: ExecutionHandle) -> ExecutionStatus:
        """Request a cooperative checkpoint through the run output directory."""
        if not handle.command:
            return ExecutionStatus(state="running", message="checkpoint request unavailable")
        output = Path(handle.command[-1])
        if not output.is_absolute():
            output = Path(handle.cwd) / output
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoint.request").touch()
        return ExecutionStatus(state="checkpointing", message=str(output / "checkpoint.request"))

    def fetch(self, handle: ExecutionHandle, source: str | Path, destination: str | Path) -> Path:
        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            raise RunnerError(f"source does not exist: {src}")
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return dst


class LocalRunner(ExecutionRunner):
    name = "local"

    def __init__(self) -> None:
        self._processes: Dict[str, subprocess.Popen] = {}
        self._log_handles: Dict[str, object] = {}

    def submit(self, command, *, cwd, log_path, environment=None) -> ExecutionHandle:
        if not command:
            raise RunnerError("command must not be empty")
        workdir = Path(cwd).resolve()
        log = Path(log_path).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(environment or {})
        log_handle = log.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(workdir),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            raise RunnerError(f"failed to launch {command[0]}: {exc}") from exc
        key = str(process.pid)
        self._processes[key] = process
        self._log_handles[key] = log_handle
        return ExecutionHandle(
            runner=self.name,
            external_id=key,
            command=list(command),
            cwd=str(workdir),
            log_path=str(log),
        )

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        process = self._processes.get(handle.external_id)
        if process is not None:
            code = process.poll()
            if code is None:
                return ExecutionStatus(state="running")
            self._close_log(handle.external_id)
            return ExecutionStatus(state="completed" if code == 0 else "failed", return_code=code)
        try:
            os.kill(int(handle.external_id), 0)
            return ExecutionStatus(state="running", message="process is alive but not owned by this session")
        except ProcessLookupError:
            return ExecutionStatus(state="unknown", message="process is no longer alive; return code unavailable")
        except (PermissionError, ValueError):
            return ExecutionStatus(state="unknown", message="cannot inspect process")

    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        process = self._processes.get(handle.external_id)
        try:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                code = process.returncode
            else:
                os.kill(int(handle.external_id), signal.SIGTERM)
                code = None
        except ProcessLookupError:
            code = None
        except subprocess.TimeoutExpired:
            if process is not None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                code = process.returncode
            else:
                code = None
        self._close_log(handle.external_id)
        return ExecutionStatus(state="cancelled", return_code=code)

    def _close_log(self, key: str) -> None:
        handle = self._log_handles.pop(key, None)
        if handle is not None:
            handle.close()


class MPIRunner(LocalRunner):
    name = "mpi"

    def __init__(self, num_procs: int, launcher: Optional[str] = None) -> None:
        super().__init__()
        if num_procs <= 0:
            raise ValueError("num_procs must be positive")
        self.num_procs = num_procs
        self.launcher = launcher or shutil.which("mpirun") or shutil.which("mpiexec")
        if not self.launcher:
            raise RunnerError("MPI launcher was not found")

    def submit(self, command, *, cwd, log_path, environment=None) -> ExecutionHandle:
        handle = super().submit(
            [self.launcher, "-np", str(self.num_procs), *command],
            cwd=cwd,
            log_path=log_path,
            environment=environment,
        )
        handle.runner = self.name
        return handle


class DockerRunner(LocalRunner):
    name = "docker"

    def __init__(self, image: str) -> None:
        super().__init__()
        if not image:
            raise ValueError("container image is required")
        self.image = image
        self.docker = shutil.which("docker")
        if not self.docker:
            raise RunnerError("docker was not found")

    def submit(self, command, *, cwd, log_path, environment=None) -> ExecutionHandle:
        workdir = Path(cwd).resolve()
        env_args: List[str] = []
        for key, value in (environment or {}).items():
            env_args.extend(["-e", f"{key}={value}"])
        docker_command = [
            self.docker,
            "run",
            "--rm",
            *env_args,
            "-v",
            f"{workdir}:/work",
            "-w",
            "/work",
            self.image,
            *command,
        ]
        handle = super().submit(docker_command, cwd=workdir, log_path=log_path)
        handle.runner = self.name
        return handle


class SlurmRunner(ExecutionRunner):
    name = "slurm"

    def __init__(self, *, partition: Optional[str] = None, walltime: Optional[str] = None) -> None:
        self.partition = partition
        self.walltime = walltime
        self.sbatch = shutil.which("sbatch")
        self.squeue = shutil.which("squeue")
        self.sacct = shutil.which("sacct")
        self.scancel = shutil.which("scancel")
        if not self.sbatch:
            raise RunnerError("sbatch was not found")

    def submit(self, command, *, cwd, log_path, environment=None) -> ExecutionHandle:
        workdir = Path(cwd).resolve()
        log = Path(log_path).resolve()
        script = workdir / f"kiturb_{uuid.uuid4().hex[:8]}.sbatch"
        lines = ["#!/usr/bin/env bash"]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.walltime:
            lines.append(f"#SBATCH --time={self.walltime}")
        lines.append(f"#SBATCH --output={log}")
        lines.append("set -euo pipefail")
        for key, value in (environment or {}).items():
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.append(" ".join(shlex.quote(token) for token in command))
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = subprocess.run(
            [self.sbatch, "--parsable", str(script)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or "sbatch failed")
        job_id = result.stdout.strip().split(";")[0]
        return ExecutionHandle(
            runner=self.name,
            external_id=job_id,
            command=list(command),
            cwd=str(workdir),
            log_path=str(log),
            metadata={"script": str(script)},
        )

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        if self.squeue:
            result = subprocess.run(
                [self.squeue, "-h", "-j", handle.external_id, "-o", "%T"],
                capture_output=True,
                text=True,
            )
            state = result.stdout.strip().lower()
            if state:
                mapped = "running" if state in {"running", "completing"} else "queued"
                return ExecutionStatus(state=mapped, message=state)
        if self.sacct:
            result = subprocess.run(
                [self.sacct, "-n", "-j", handle.external_id, "--format=State,ExitCode"],
                capture_output=True,
                text=True,
            )
            line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
            upper = line.upper()
            if "COMPLETED" in upper:
                return ExecutionStatus(state="completed", return_code=0, message=line)
            if any(token in upper for token in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
                return ExecutionStatus(state="failed", message=line)
        return ExecutionStatus(state="unknown")

    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        if not self.scancel:
            raise RunnerError("scancel was not found")
        result = subprocess.run([self.scancel, handle.external_id], capture_output=True, text=True)
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or "scancel failed")
        return ExecutionStatus(state="cancelled")


class SSHRunner(ExecutionRunner):
    name = "ssh"

    def __init__(self, host: str, *, remote_root: str = "~/kiturb-runs") -> None:
        if not host:
            raise ValueError("SSH host is required")
        self.host = host
        self.remote_root = remote_root
        self.ssh = shutil.which("ssh")
        self.scp = shutil.which("scp")
        if not self.ssh or not self.scp:
            raise RunnerError("ssh and scp are required")

    def submit(self, command, *, cwd, log_path, environment=None) -> ExecutionHandle:
        local_cwd = Path(cwd).resolve()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        remote_dir = f"{self.remote_root}/{run_id}"
        subprocess.run([self.ssh, self.host, "mkdir", "-p", remote_dir], check=True)
        subprocess.run([self.scp, "-r", f"{local_cwd}/.", f"{self.host}:{remote_dir}/"], check=True)
        exports = " ".join(
            f"{shlex.quote(key)}={shlex.quote(value)}" for key, value in (environment or {}).items()
        )
        remote_log = f"{remote_dir}/run.log"
        remote_command = (
            f"cd {shlex.quote(remote_dir)} && "
            f"nohup env {exports} {' '.join(shlex.quote(x) for x in command)} "
            f"> {shlex.quote(remote_log)} 2>&1 < /dev/null & echo $!"
        )
        result = subprocess.run(
            [self.ssh, self.host, remote_command], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or "remote launch failed")
        pid = result.stdout.strip().splitlines()[-1]
        return ExecutionHandle(
            runner=self.name,
            external_id=pid,
            command=list(command),
            cwd=str(local_cwd),
            log_path=str(log_path),
            metadata={"host": self.host, "remote_dir": remote_dir, "remote_log": remote_log},
        )

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        result = subprocess.run(
            [self.ssh, self.host, "kill", "-0", handle.external_id],
            capture_output=True,
            text=True,
        )
        return ExecutionStatus(state="running" if result.returncode == 0 else "unknown")

    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        subprocess.run([self.ssh, self.host, "kill", "-TERM", handle.external_id], check=False)
        return ExecutionStatus(state="cancelled")

    def fetch(self, handle: ExecutionHandle, source: str | Path, destination: str | Path) -> Path:
        remote_dir = handle.metadata.get("remote_dir")
        if not remote_dir:
            raise RunnerError("remote directory is missing from handle")
        destination_path = Path(destination).resolve()
        destination_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self.scp, "-r", f"{self.host}:{remote_dir}/{source}", str(destination_path)],
            check=True,
        )
        return destination_path


class CloudRunner(ExecutionRunner):
    """Provider-neutral cloud runner driven by administrator command hooks.

    Configure KITURB_CLOUD_SUBMIT_CMD, KITURB_CLOUD_STATUS_CMD and
    KITURB_CLOUD_CANCEL_CMD as executable command prefixes.  No command is
    accepted from an LLM tool call; these hooks are deployment configuration.
    """
    name = "cloud"

    def __init__(self) -> None:
        self.submit_cmd = os.getenv("KITURB_CLOUD_SUBMIT_CMD")
        self.status_cmd = os.getenv("KITURB_CLOUD_STATUS_CMD")
        self.cancel_cmd = os.getenv("KITURB_CLOUD_CANCEL_CMD")
        self.fetch_cmd = os.getenv("KITURB_CLOUD_FETCH_CMD")
        if not self.submit_cmd or not self.status_cmd or not self.cancel_cmd:
            raise RunnerError(
                "cloud execution requires KITURB_CLOUD_SUBMIT_CMD, "
                "KITURB_CLOUD_STATUS_CMD and KITURB_CLOUD_CANCEL_CMD"
            )

    @staticmethod
    def _prefix(value: str) -> List[str]:
        tokens = shlex.split(value)
        if not tokens or not shutil.which(tokens[0]):
            raise RunnerError(f"cloud hook executable was not found: {value}")
        return tokens

    def submit(self, command, *, cwd, log_path, environment=None) -> ExecutionHandle:
        workdir = Path(cwd).resolve()
        descriptor = workdir / f"cloud_job_{uuid.uuid4().hex[:10]}.json"
        descriptor.write_text(json.dumps({
            "command": list(command), "cwd": str(workdir),
            "log_path": str(Path(log_path).resolve()),
            "environment": dict(environment or {}),
        }, indent=2), encoding="utf-8")
        result = subprocess.run(
            [*self._prefix(self.submit_cmd), str(descriptor)],
            cwd=str(workdir), capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or "cloud submission failed")
        job_id = result.stdout.strip().splitlines()[-1].strip()
        if not job_id:
            raise RunnerError("cloud submit hook returned no job identifier")
        return ExecutionHandle(
            runner=self.name, external_id=job_id, command=list(command),
            cwd=str(workdir), log_path=str(Path(log_path).resolve()),
            metadata={"descriptor": str(descriptor)},
        )

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        result = subprocess.run(
            [*self._prefix(self.status_cmd), handle.external_id],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return ExecutionStatus(state="unknown", message=result.stderr.strip())
        raw = result.stdout.strip().lower()
        for state in ("completed", "failed", "cancelled", "running", "queued", "checkpointing"):
            if state in raw:
                return ExecutionStatus(state=state, message=raw)
        return ExecutionStatus(state="unknown", message=raw)

    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        result = subprocess.run(
            [*self._prefix(self.cancel_cmd), handle.external_id],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or "cloud cancellation failed")
        return ExecutionStatus(state="cancelled", message=result.stdout.strip())

    def fetch(self, handle: ExecutionHandle, source: str | Path, destination: str | Path) -> Path:
        if not self.fetch_cmd:
            raise RunnerError("cloud fetching requires KITURB_CLOUD_FETCH_CMD")
        destination_path = Path(destination).resolve()
        destination_path.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [*self._prefix(self.fetch_cmd), handle.external_id, str(source), str(destination_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or "cloud fetch failed")
        return destination_path


def runner_from_config(config) -> ExecutionRunner:
    mode = str(getattr(config.mode, "value", config.mode)).lower()
    if mode == "local":
        return LocalRunner()
    if mode == "mpi":
        return MPIRunner(config.num_procs)
    if mode == "docker":
        return DockerRunner(config.container_image)
    if mode == "slurm":
        return SlurmRunner(partition=config.queue, walltime=config.walltime)
    if mode == "ssh":
        return SSHRunner(config.host)
    if mode == "cloud":
        return CloudRunner()
    raise RunnerError(f"unknown execution mode: {mode}")


__all__ = [
    "RunnerError",
    "ExecutionHandle",
    "ExecutionStatus",
    "ExecutionRunner",
    "LocalRunner",
    "MPIRunner",
    "DockerRunner",
    "SlurmRunner",
    "SSHRunner",
    "CloudRunner",
    "runner_from_config",
]
