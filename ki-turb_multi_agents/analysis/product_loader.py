"""Solver-neutral loader bridging manifests, on-disk products, and agent session state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import DatasetManifest
from schemas.hit_analysis_products import HITAnalysisProducts

from .manifest_index import MANIFEST_KIND_TO_SESSION_KEY, SESSION_KEY_ALIASES


def _resolve_path(path: str | Path, *, base_dir: Path, project_root: Path) -> Optional[Path]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if candidate.is_file():
        return candidate
    alt = (project_root / path).resolve()
    return alt if alt.is_file() else None


class AnalysisProductLoader:
    """Unified access to CFD analysis data for any supported backend."""

    def __init__(
        self,
        project_root: Path,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.session_context = session_context or {}
        self._manifest: Optional[DatasetManifest] = None
        self._products: Optional[HITAnalysisProducts] = None

    @classmethod
    def from_manifest_path(
        cls,
        project_root: Path,
        manifest_path: str | Path,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> "AnalysisProductLoader":
        loader = cls(project_root, session_context)
        path = Path(manifest_path)
        if not path.is_absolute():
            path = (Path(project_root) / path).resolve()
        if path.is_file():
            loader._manifest = DatasetManifest.from_json(path.read_text(encoding="utf-8"))
            if session_context is not None:
                session_context["dataset_manifest"] = loader._manifest.model_dump(mode="json")
                session_context["manifest_path"] = str(path)
        return loader

    def manifest(self) -> Optional[DatasetManifest]:
        if self._manifest is not None:
            return self._manifest
        payload = self.session_context.get("dataset_manifest")
        if isinstance(payload, dict) and payload.get("manifest_id"):
            self._manifest = DatasetManifest.model_validate(payload)
            return self._manifest
        manifest_path = self.session_context.get("manifest_path")
        if manifest_path:
            path = Path(str(manifest_path))
            if not path.is_absolute():
                path = (self.project_root / path).resolve()
            if path.is_file():
                self._manifest = DatasetManifest.from_json(path.read_text(encoding="utf-8"))
        if self._manifest is None:
            base = self.base_dir()
            for name in ("dataset_manifest.json", "manifest.json"):
                candidate = base / name
                if candidate.is_file():
                    self._manifest = DatasetManifest.from_json(candidate.read_text(encoding="utf-8"))
                    break
        return self._manifest

    def products(self, *, reload: bool = False) -> Optional[HITAnalysisProducts]:
        if self._products is not None and not reload:
            return self._products
        manifest = self.manifest()
        if manifest is not None:
            try:
                from postprocessing.hit_products_adapter import load_products_from_manifest

                self._products = load_products_from_manifest(manifest)
                return self._products
            except Exception:
                pass
        for path in self.files_of_kind("analysis_products"):
            try:
                self._products = HITAnalysisProducts.load(path)
                return self._products
            except Exception:
                continue
        candidate = self.base_dir() / "processed" / "hit_analysis_products.json"
        if candidate.is_file():
            try:
                self._products = HITAnalysisProducts.load(candidate)
            except Exception:
                self._products = None
        return self._products

    def base_dir(self) -> Path:
        manifest = self.manifest()
        if manifest is not None and manifest.base_dir:
            return Path(manifest.base_dir)
        data_dir = self.session_context.get("data_directory")
        if data_dir:
            path = Path(str(data_dir))
            if not path.is_absolute():
                path = (self.project_root / path).resolve()
            if path.is_dir():
                return path
        return self.project_root

    def backend(self) -> Optional[str]:
        manifest = self.manifest()
        if manifest and manifest.backend:
            return manifest.backend
        return self.session_context.get("backend")

    def validation_status(self) -> str:
        products = self.products()
        if products is not None and products.validation_status:
            return str(products.validation_status)
        manifest = self.manifest()
        if manifest is not None:
            status = (manifest.postprocessing or {}).get("validation_status")
            if status:
                return str(status)
        return "unvalidated"

    def files_of_kind(self, kind: str) -> List[Path]:
        manifest = self.manifest()
        resolved: List[Path] = []
        if manifest is not None:
            for entry in manifest.files_of_kind(kind):
                path = _resolve_path(entry.path, base_dir=Path(manifest.base_dir), project_root=self.project_root)
                if path is not None:
                    resolved.append(path)
        if resolved:
            return resolved

        session_key = MANIFEST_KIND_TO_SESSION_KEY.get(kind, kind)
        indexed = self.session_context.get("all_loaded_files") or {}
        for key in (session_key, SESSION_KEY_ALIASES.get(session_key, "")):
            if not key:
                continue
            for item in indexed.get(key, []):
                if isinstance(item, dict) and item.get("full_path"):
                    path = Path(item["full_path"])
                    if path.is_file():
                        resolved.append(path)
                elif isinstance(item, str) and Path(item).is_file():
                    resolved.append(Path(item))

        if not resolved:
            from utils.file_detector import detect_simulation_files

            detector_key = MANIFEST_KIND_TO_SESSION_KEY.get(kind, kind)
            files = detect_simulation_files(str(self.base_dir()))
            for item in files.get(detector_key, []):
                path = Path(item)
                if path.is_file():
                    resolved.append(path)
        return resolved

    def velocity_snapshots(self, *, step: Optional[int] = None) -> List[Path]:
        paths = self.files_of_kind("velocity_field")
        if not paths:
            indexed = self.session_context.get("all_loaded_files") or {}
            for key in ("velocity_files", "velocity_vti", "velocity_h5"):
                for item in indexed.get(key, []):
                    if isinstance(item, dict) and item.get("full_path"):
                        paths.append(Path(item["full_path"]))
        paths = sorted({p.resolve() for p in paths if p.is_file()})
        if step is None:
            return paths
        for path in paths:
            if str(step) in path.stem:
                return [path]
        return paths[:1] if paths else []

    def grid_spacing(self) -> Tuple[float, float, float]:
        manifest = self.manifest()
        if manifest is not None:
            for entry in manifest.files_of_kind("velocity_field"):
                if entry.spacing and len(entry.spacing) == 3:
                    return tuple(float(v) for v in entry.spacing)  # type: ignore[return-value]
        try:
            from pages.PDFs.pdf_params import get_grid_spacing_options

            options = get_grid_spacing_options(self.base_dir())
            if options:
                return next(iter(options.values()))
        except Exception:
            pass
        return (1.0, 1.0, 1.0)

    def viscosity(self) -> Optional[float]:
        manifest = self.manifest()
        if manifest is not None:
            case = manifest.case or manifest.requested_config or {}
            for key in ("hit", "solver", "scaling"):
                block = case.get(key) or {}
                nu = block.get("physical_viscosity") or block.get("viscosity")
                if nu is not None:
                    return float(nu)
        try:
            from data_readers.parameter_reader import read_parameters

            for candidate in (self.base_dir() / "simulation.input", self.base_dir() / "simulation.json"):
                if candidate.exists():
                    params = read_parameters(str(candidate))
                    if params and "nu" in params:
                        return float(params["nu"])
        except Exception:
            pass
        return None

    def hdf5_fortran_order(self) -> bool:
        return bool(self.session_context.get("hdf5_fortran_order", True))

    def enrich_session_files(self, all_loaded_files: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Register manifest files into the session index. Returns hint keys set."""
        hints: Dict[str, str] = {}
        manifest = self.manifest()
        if manifest is None:
            return hints
        base_dir = Path(manifest.base_dir)
        for entry in manifest.files:
            session_key = MANIFEST_KIND_TO_SESSION_KEY.get(entry.kind)
            if not session_key:
                continue
            resolved = _resolve_path(entry.path, base_dir=base_dir, project_root=self.project_root)
            if resolved is None:
                continue
            all_loaded_files.setdefault(session_key, [])
            payload = {
                "full_path": str(resolved),
                "directory": str(resolved.parent),
                "filename": resolved.name,
                "kind": entry.kind,
                "backend": manifest.backend,
            }
            existing = {item.get("full_path") for item in all_loaded_files[session_key]}
            if payload["full_path"] not in existing:
                all_loaded_files[session_key].append(payload)
            if entry.kind == "energy_spectrum":
                hints["spectra_data_directory"] = str(resolved.parent)
            if entry.kind in {"dissipation_validation", "turbulence_stats", "reynolds_stress"}:
                hints["stats_data_directory"] = str(resolved.parent)
            if entry.kind == "spectral_isotropy":
                hints["isotropy_data_directory"] = str(resolved.parent)
            if entry.kind == "structure_functions":
                hints["structure_functions_data_directory"] = str(resolved.parent)
            if entry.kind == "analysis_products":
                hints["analysis_products_path"] = str(resolved)
        if self._products is not None:
            self.session_context["analysis_products"] = self._products.model_dump(mode="json")
        return hints

    def overview_payload(self) -> Dict[str, Any]:
        """Solver-neutral summary for the Overview page and agent tools."""
        from utils.file_detector import detect_simulation_files

        base = self.base_dir()
        files = detect_simulation_files(str(base))
        payload: Dict[str, Any] = {
            "directory": base.name,
            "path": str(base),
            "backend": self.backend(),
            "files": files,
            "validation_status": self.validation_status(),
            "analysis_products_loaded": self.products() is not None,
        }
        products = self.products()
        if products and products.time_history is not None:
            history = products.time_history
            if history.divergence_rms:
                payload["divergence_rms_max"] = float(max(history.divergence_rms))
            if history.mach_max:
                payload["mach_max"] = float(max(history.mach_max))
            if history.re_lambda:
                payload["re_lambda_last"] = float(history.re_lambda[-1])
            if history.kmax_eta:
                payload["kmax_eta_min"] = float(min(history.kmax_eta))
        if products and products.resolution is not None:
            payload["kmax_eta_min"] = products.resolution.kmax_eta_min
        manifest = self.manifest()
        if manifest is not None:
            payload["manifest_id"] = manifest.manifest_id
            payload["postprocessing_status"] = (manifest.postprocessing or {}).get("status")
        return payload


__all__ = ["AnalysisProductLoader"]
