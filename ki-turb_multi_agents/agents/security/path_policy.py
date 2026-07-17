"""Restrict file writes to allowlisted directories.

A :class:`PathPolicy` holds a set of approved root directories. A write is
permitted only if the fully-resolved target (after following symlinks and
collapsing ``..``) lies inside one of those roots. This blocks absolute paths
outside the approved roots, directory traversal, and symlink escapes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

PathLike = Union[str, os.PathLike]


class PathPolicyError(PermissionError):
    """Raised when a path is not permitted for writing."""


class PathPolicy:
    """Enforce that writes stay within an allowlist of resolved root directories."""

    def __init__(self, allowed_roots: Iterable[PathLike]):
        roots: List[Path] = []
        for r in allowed_roots:
            if not r:
                continue
            # resolve() collapses ``..`` and follows symlinks on existing parents.
            roots.append(Path(r).expanduser().resolve())
        if not roots:
            raise ValueError("PathPolicy requires at least one allowed root")
        # De-duplicate while preserving order.
        seen = set()
        self.allowed_roots: List[Path] = []
        for root in roots:
            if root not in seen:
                seen.add(root)
                self.allowed_roots.append(root)

    # -- core checks --------------------------------------------------------
    def _resolve(self, path: PathLike) -> Path:
        return Path(path).expanduser().resolve()

    def is_within(self, path: PathLike, root: PathLike) -> bool:
        resolved = self._resolve(path)
        root_resolved = Path(root).expanduser().resolve()
        return resolved == root_resolved or root_resolved in resolved.parents

    def is_write_allowed(self, path: PathLike) -> bool:
        """True if ``path`` resolves to a location inside an approved root."""
        try:
            resolved = self._resolve(path)
        except (OSError, RuntimeError):
            return False
        return any(
            resolved == root or root in resolved.parents
            for root in self.allowed_roots
        )

    def resolve_write(self, path: PathLike) -> Path:
        """Return the resolved path if writing is allowed, else raise PathPolicyError."""
        resolved = self._resolve(path)
        if not self.is_write_allowed(resolved):
            raise PathPolicyError(
                f"Write denied: '{path}' resolves to '{resolved}', which is outside "
                f"approved folders: {', '.join(str(r) for r in self.allowed_roots)}"
            )
        return resolved

    def assert_write_allowed(self, path: PathLike) -> None:
        self.resolve_write(path)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        roots = ", ".join(str(r) for r in self.allowed_roots)
        return f"PathPolicy(allowed_roots=[{roots}])"


def default_policy(
    project_root: PathLike,
    *,
    export_dir: Optional[PathLike] = None,
    case_dir: Optional[PathLike] = None,
    extra_roots: Optional[Iterable[PathLike]] = None,
) -> PathPolicy:
    """Build a policy covering the project tree plus export and case directories.

    ``export_dir`` and ``case_dir`` default to ``<project>/exports`` and
    ``<project>/cases`` and may point outside the project root.
    """
    project_root = Path(project_root).expanduser().resolve()
    roots: List[PathLike] = [project_root]

    roots.append(export_dir if export_dir is not None else project_root / "exports")
    roots.append(case_dir if case_dir is not None else project_root / "cases")
    if extra_roots:
        roots.extend(extra_roots)
    return PathPolicy(roots)
