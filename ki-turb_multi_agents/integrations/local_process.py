"""Local process execution shared by CFD backends.

Launches solver executables as child processes with ``shell=False``, streams
their combined output to a log file, and exposes non-blocking status polling and
termination by PID. Backends use this instead of calling ``subprocess`` directly
so process handling is consistent and testable.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional


@dataclass
class ProcessHandle:
    """Reference to a launched local process."""

    pid: int
    log_path: str
    argv: List[str]
    cwd: str
    returncode: Optional[int] = None


class LocalProcessError(RuntimeError):
    """Raised when a local process cannot be launched."""


class LocalProcessRunner:
    """Launches and tracks local child processes.

    Popen objects are retained in-process so completed processes report an exact
    return code. When the original handle is unavailable (for example a job
    polled in a later session), liveness falls back to a signal-0 probe.
    """

    def __init__(self) -> None:
        self._processes: Dict[int, subprocess.Popen] = {}

    def spawn(
        self,
        argv: List[str],
        *,
        cwd: os.PathLike,
        log_path: os.PathLike,
        env: Optional[Mapping[str, str]] = None,
    ) -> ProcessHandle:
        """Start ``argv`` in ``cwd``, redirecting stdout and stderr to ``log_path``."""
        if not argv:
            raise LocalProcessError("argv must not be empty")

        cwd_path = Path(cwd)
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)

        process_env = dict(os.environ)
        if env:
            process_env.update(env)

        try:
            log_handle = open(log, "w")
        except OSError as exc:
            raise LocalProcessError(f"cannot open log file '{log}': {exc}") from exc

        try:
            process = subprocess.Popen(
                argv,
                cwd=str(cwd_path),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=process_env,
                shell=False,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            log_handle.close()
            raise LocalProcessError(f"failed to launch {argv[0]!r}: {exc}") from exc

        self._processes[process.pid] = process
        return ProcessHandle(
            pid=process.pid,
            log_path=str(log),
            argv=list(argv),
            cwd=str(cwd_path),
        )

    def poll(self, pid: int) -> Optional[int]:
        """Return the exit code, or ``None`` while the process is still running."""
        process = self._processes.get(pid)
        if process is not None:
            return process.poll()
        state = self._proc_state(pid)
        if state == "Z":
            # Defunct children still answer signal 0 but have already exited.
            return 0
        return None if self._pid_alive(pid) else 0

    def is_running(self, pid: int) -> bool:
        return self.poll(pid) is None

    def terminate(self, pid: int, *, timeout: float = 10.0) -> bool:
        """Terminate a process, escalating to SIGKILL if it does not exit."""
        process = self._processes.get(pid)
        if process is not None:
            if process.poll() is not None:
                return True
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
            return True
        return self._kill_pid(pid)

    @staticmethod
    def _proc_state(pid: int) -> Optional[str]:
        """Linux process state letter from /proc, e.g. R/S/Z."""
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("State:"):
                    parts = line.split()
                    return parts[1] if len(parts) > 1 else None
        except OSError:
            return None
        return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _kill_pid(pid: int) -> bool:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False
        return True
