"""Collect reproducible OpenLB build and execution provenance."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProvenanceRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    collected_at: str
    openlb_root: Optional[str] = None
    openlb_version: Optional[str] = None
    openlb_commit: Optional[str] = None
    app_dir: Optional[str] = None
    app_commit: Optional[str] = None
    compiler: Optional[str] = None
    compiler_version: Optional[str] = None
    build_profile: Optional[str] = None
    build_command: List[str] = Field(default_factory=list)
    build_flags: List[str] = Field(default_factory=list)
    executable: Optional[str] = None
    executable_sha256: Optional[str] = None
    hostname: str
    platform: str
    architecture: str
    cpu_count: Optional[int] = None
    environment: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OpenLBProvenanceCollector:
    def collect(
        self,
        *,
        openlb_root: str | Path | None = None,
        app_dir: str | Path | None = None,
        executable: str | Path | None = None,
        compiler: str | None = None,
        build_profile: str | None = None,
        build_command: Optional[List[str]] = None,
        build_flags: Optional[List[str]] = None,
        environment_keys: Optional[List[str]] = None,
    ) -> ProvenanceRecord:
        root = Path(openlb_root).resolve() if openlb_root else None
        app = Path(app_dir).resolve() if app_dir else None
        exe = Path(executable).resolve() if executable and Path(executable).exists() else None
        compiler_path = shutil.which(compiler or os.environ.get("CXX", "g++"))
        return ProvenanceRecord(
            collected_at=datetime.now(timezone.utc).isoformat(),
            openlb_root=str(root) if root else None,
            openlb_version=self._detect_version(root),
            openlb_commit=self._git_commit(root),
            app_dir=str(app) if app else None,
            app_commit=self._git_commit(app),
            compiler=compiler_path,
            compiler_version=self._compiler_version(compiler_path),
            build_profile=build_profile,
            build_command=list(build_command or []),
            build_flags=list(build_flags or []),
            executable=str(exe) if exe else None,
            executable_sha256=self.sha256(exe) if exe else None,
            hostname=platform.node(),
            platform=platform.platform(),
            architecture=platform.machine(),
            cpu_count=os.cpu_count(),
            environment={key: os.environ[key] for key in (environment_keys or []) if key in os.environ},
        )

    @staticmethod
    def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _run(argv: List[str], cwd: Optional[Path] = None) -> Optional[str]:
        try:
            result = subprocess.run(
                argv,
                cwd=str(cwd) if cwd else None,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return (result.stdout or result.stderr).strip() or None

    def _git_commit(self, path: Optional[Path]) -> Optional[str]:
        if not path or not path.exists():
            return None
        return self._run(["git", "rev-parse", "HEAD"], cwd=path)

    def _detect_version(self, root: Optional[Path]) -> Optional[str]:
        if not root:
            return None
        for name in ("VERSION", "version.txt", "release.txt"):
            candidate = root / name
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace").strip()
        describe = self._run(["git", "describe", "--tags", "--always"], cwd=root)
        return describe

    def _compiler_version(self, compiler_path: Optional[str]) -> Optional[str]:
        if not compiler_path:
            return None
        output = self._run([compiler_path, "--version"])
        return output.splitlines()[0] if output else None


__all__ = ["ProvenanceRecord", "OpenLBProvenanceCollector"]
