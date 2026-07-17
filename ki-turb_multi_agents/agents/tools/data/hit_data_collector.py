"""Variable-aware, checksum-backed collection of OpenLB HIT outputs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from schemas import DatasetFile, DatasetManifest
from schemas.unit_system import UnitSystem


class CollectionResult(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    manifest: DatasetManifest
    manifest_path: str
    copied_files: int
    skipped_files: int
    warnings: List[str] = Field(default_factory=list)


class HITDataCollector:
    _STEP_RE = re.compile(r"(?:step|iter|time|t)[_-]?(\d+)", re.IGNORECASE)
    _TRAILING_INTEGER_RE = re.compile(r"(\d+)(?!.*\d)")
    _DATA_ARRAY = re.compile(rb"<DataArray\b([^>]*)>", re.IGNORECASE)
    _ATTR = re.compile(rb"([A-Za-z_][A-Za-z0-9_]*)=[\"']([^\"']*)[\"']")

    VARIABLE_ALIASES: Dict[str, str] = {
        "velocity": "velocity_field", "vel": "velocity_field", "u": "velocity_field",
        "pressure": "pressure_field", "p": "pressure_field",
        "density": "density_field", "rho": "density_field",
        "vorticity": "vorticity_field", "omega": "vorticity_field",
        "forcing": "forcing_field", "force": "forcing_field",
        "population": "population_field", "populations": "population_field",
        "spectrum": "energy_spectrum", "isotropy": "spectral_isotropy",
    }
    DEFAULT_UNITS = {
        "velocity_field": "physical_velocity",
        "pressure_field": "lattice_pressure",
        "density_field": "lattice_density",
        "vorticity_field": "1/physical_time",
        "forcing_field": "lattice_acceleration",
    }

    def collect(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        *,
        source_job_id: Optional[str] = None,
        source_simulation: Optional[str] = None,
        backend: str = "openlb",
        case: Optional[dict] = None,
        provenance: Optional[dict] = None,
        units: Optional[Dict[str, str]] = None,
        unit_system: Optional[UnitSystem] = None,
        copy: bool = True,
        require_nonempty: bool = True,
        expected_kinds: Optional[Sequence[str]] = None,
    ) -> CollectionResult:
        source = Path(source_dir).expanduser().resolve()
        target = Path(target_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"OpenLB output directory not found: {source}")
        target.mkdir(parents=True, exist_ok=True)
        resolved_units = unit_system
        if resolved_units is None and units:
            resolved_units = UnitSystem.from_legacy_labels(units, source_backend=backend)
        unit_map = {
            **self.DEFAULT_UNITS,
            **(resolved_units.field_labels() if resolved_units else {}),
            **(units or {}),
        }
        step_times = self._read_step_times(source)

        manifest = DatasetManifest(
            manifest_id=f"ds_{uuid.uuid4().hex[:12]}",
            base_dir=str(target),
            source_job_id=source_job_id,
            source_simulation=source_simulation,
            run_id=source_job_id,
            backend=backend,
            case=case or {},
            requested_config=case or {},
            effective_config=provenance or {},
            provenance=provenance or {},
            units=unit_map,
            unit_system=resolved_units,
            status="fetching",
        )
        warnings: List[str] = []
        copied = skipped = 0
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp") or path.name.startswith("."):
                continue
            if path.stat().st_size == 0:
                warnings.append(f"skipped empty file: {path.relative_to(source)}")
                skipped += 1
                continue
            relative = path.relative_to(source)
            destination = target / relative
            if copy and path.resolve() != destination.resolve():
                self._atomic_copy(path, destination)
                collected_path = destination
                copied += 1
            else:
                collected_path = path
            kind, fmt, metadata = self.classify(collected_path)
            step = self.parse_step(collected_path)
            variable = kind.removesuffix("_field") if kind.endswith("_field") else metadata.get("variable")
            item_units = unit_map.get(kind) or unit_map.get(str(variable))
            complete = self._validate_complete(collected_path, kind, metadata)
            if not complete:
                warnings.append(f"file failed completeness validation: {relative}")
            manifest.add_file(DatasetFile(
                path=str(collected_path.relative_to(target)) if self._is_relative_to(collected_path, target) else str(collected_path),
                kind=kind,
                variable=str(variable) if variable else None,
                format=fmt,
                time_step=step,
                time_value=step_times.get(step) if step is not None else None,
                size_bytes=collected_path.stat().st_size,
                checksum=self.sha256(collected_path),
                complete=complete,
                components=metadata.get("components"),
                shape=metadata.get("shape"),
                spacing=metadata.get("spacing"),
                origin=metadata.get("origin"),
                units=item_units,
                metadata=metadata,
            ))

        if require_nonempty and not manifest.files:
            raise RuntimeError("no complete OpenLB output files were collected")
        present = {item.kind for item in manifest.files if item.complete}
        for expected in expected_kinds or []:
            if expected not in present:
                warnings.append(f"expected output kind is missing: {expected}")
        manifest.validation["collection_warnings"] = warnings
        manifest.validation["complete"] = not any("missing" in warning or "failed" in warning for warning in warnings)
        manifest.status = "fetched" if manifest.validation["complete"] else "fetched_with_warnings"
        manifest_path = target / "dataset_manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(manifest.to_json(), encoding="utf-8")
        temporary.replace(manifest_path)
        return CollectionResult(
            manifest=manifest,
            manifest_path=str(manifest_path),
            copied_files=copied,
            skipped_files=skipped,
            warnings=warnings,
        )

    def classify(self, path: str | Path) -> Tuple[str, Optional[str], Dict[str, object]]:
        target = Path(path)
        lower_name = target.name.lower()
        suffix = target.suffix.lower()
        fmt = {
            ".h5": "hdf5", ".hdf5": "hdf5", ".vti": "vti", ".pvti": "pvti",
            ".vtu": "vtu", ".pvtu": "pvtu", ".pvd": "pvd", ".vtm": "vtm",
            ".csv": "csv", ".dat": "dat", ".json": "json", ".jsonl": "jsonl",
            ".log": "log", ".txt": "txt", ".npy": "npy", ".npz": "npz", ".khf": "kiturb_hit_field",
        }.get(suffix, suffix.lstrip(".") or None)
        metadata: Dict[str, object] = {}

        if lower_name.startswith("forcing_state_"):
            return "forcing_state", fmt, metadata
        if lower_name.startswith("checkpoint_") or suffix == ".khf":
            metadata.update(self._khf_metadata(target))
            return "checkpoint", fmt, metadata
        if lower_name == "diagnostics.jsonl":
            return "diagnostics", fmt, metadata
        if lower_name in {"effective_openlb.json", "initial_condition_diagnostics.json"}:
            return "metadata", fmt, metadata
        if suffix in {".vtm", ".pvd"}:
            return "field_collection", fmt, metadata

        array_names: List[str] = []
        if suffix in {".vti", ".pvti", ".vtu", ".pvtu"}:
            metadata.update(self._vtk_metadata(target))
            array_names = list(metadata.get("array_names", []))
        elif suffix in {".h5", ".hdf5"}:
            metadata.update(self._hdf5_metadata(target))
            array_names = list(metadata.get("array_names", []))

        tokens = [target.stem.lower(), *[name.lower() for name in array_names]]
        for token in tokens:
            for alias, kind in self.VARIABLE_ALIASES.items():
                if re.search(rf"(^|[^a-z0-9]){re.escape(alias)}([^a-z0-9]|$)", token):
                    metadata.setdefault("components", 3 if kind in {"velocity_field", "vorticity_field", "forcing_field"} else 1)
                    metadata["variable"] = kind.removesuffix("_field")
                    shape = metadata.get("shape")
                    if shape and kind in {"velocity_field", "vorticity_field", "forcing_field"} and len(shape) == 3:
                        metadata["shape"] = (*shape, int(metadata["components"]))
                    return kind, fmt, metadata

        if suffix == ".log":
            return "log", fmt, metadata
        if suffix == ".txt":
            return "log", fmt, metadata
        if suffix in {".json", ".jsonl"}:
            return "metadata", fmt, metadata
        if suffix in {".csv", ".dat"}:
            return "table", fmt, metadata
        if suffix in {".vti", ".pvti", ".vtu", ".pvtu", ".h5", ".hdf5"}:
            return "unclassified_field", fmt, metadata
        return "data", fmt, metadata

    def parse_step(self, path: str | Path) -> Optional[int]:
        stem = Path(path).stem
        match = self._STEP_RE.search(stem) or self._TRAILING_INTEGER_RE.search(stem)
        return int(match.group(1)) if match else None

    @staticmethod
    def sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    def _vtk_metadata(self, path: Path) -> Dict[str, object]:
        try:
            header = path.read_bytes()[:1024 * 1024].split(b"<AppendedData", 1)[0]
        except OSError:
            return {}
        metadata: Dict[str, object] = {}
        array_names: List[str] = []
        components: Optional[int] = None
        for match in self._DATA_ARRAY.finditer(header):
            attrs = {key.decode(): value.decode(errors="replace") for key, value in self._ATTR.findall(match.group(1))}
            name = attrs.get("Name")
            if name:
                array_names.append(name)
                try:
                    components = int(attrs.get("NumberOfComponents", "1"))
                except ValueError:
                    pass
        if array_names:
            metadata["array_names"] = array_names
        if components:
            metadata["components"] = components
        text = header.decode("utf-8", errors="ignore")
        for attr, key, caster in (("WholeExtent", "shape", int), ("Spacing", "spacing", float), ("Origin", "origin", float)):
            found = re.search(rf'{attr}=["\']([^"\']+)', text)
            if not found:
                continue
            values = [caster(value) for value in found.group(1).split()]
            if attr == "WholeExtent" and len(values) == 6:
                metadata[key] = (values[1] - values[0] + 1, values[3] - values[2] + 1, values[5] - values[4] + 1)
            elif len(values) >= 3:
                metadata[key] = tuple(values[:3])
        return metadata

    @staticmethod
    def _hdf5_metadata(path: Path) -> Dict[str, object]:
        try:
            import h5py  # type: ignore
        except ImportError:
            return {}
        names: List[str] = []
        metadata: Dict[str, object] = {}
        try:
            with h5py.File(path, "r") as handle:
                datasets = []
                handle.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
                names = [name for name, _ in datasets]
                if datasets:
                    dataset = datasets[0][1]
                    metadata["shape"] = tuple(int(value) for value in dataset.shape)
                    metadata["components"] = int(dataset.shape[-1]) if dataset.ndim == 4 else 1
                    if "spacing" in dataset.attrs:
                        metadata["spacing"] = tuple(float(value) for value in dataset.attrs["spacing"][:3])
                    if "origin" in dataset.attrs:
                        metadata["origin"] = tuple(float(value) for value in dataset.attrs["origin"][:3])
        except OSError:
            return {}
        if names:
            metadata["array_names"] = names
        return metadata

    @staticmethod
    def _khf_metadata(path: Path) -> Dict[str, object]:
        try:
            with path.open("rb") as handle:
                magic = handle.read(16).rstrip(b"\0")
                dims = struct.unpack("<4i", handle.read(16))
            if magic != b"KITURBHITFIELD1":
                return {"valid_magic": False}
            return {"valid_magic": True, "shape": tuple(dims[:3]), "checkpoint_step": dims[3], "components": 4}
        except (OSError, struct.error):
            return {"valid_magic": False}

    @staticmethod
    def _validate_complete(path: Path, kind: str, metadata: Dict[str, object]) -> bool:
        if path.stat().st_size <= 0:
            return False
        if kind == "checkpoint":
            return bool(metadata.get("valid_magic", True))
        if kind.endswith("_field") and path.suffix.lower() in {".vti", ".h5", ".hdf5"}:
            return bool(metadata.get("shape") or metadata.get("array_names"))
        return True

    @staticmethod
    def _read_step_times(source: Path) -> Dict[int, float]:
        result: Dict[int, float] = {}
        diagnostics = source / "diagnostics.jsonl"
        if not diagnostics.is_file():
            return result
        for line in diagnostics.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
                if "step" in payload and "physical_time" in payload:
                    result[int(payload["step"])] = float(payload["physical_time"])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False


__all__ = ["CollectionResult", "HITDataCollector"]
