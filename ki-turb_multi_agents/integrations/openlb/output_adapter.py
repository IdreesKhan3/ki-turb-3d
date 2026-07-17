"""Adapt OpenLB outputs to KI-TURB's canonical dataset representation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from agents.tools.data.hit_data_collector import CollectionResult, HITDataCollector
from postprocessing.readers import VelocitySnapshot, load_velocity_snapshots
from schemas import DatasetManifest
from schemas.unit_system import UnitSystem


class OpenLBOutputAdapter:
    def __init__(self, collector: Optional[HITDataCollector] = None) -> None:
        self.collector = collector or HITDataCollector()

    def adapt(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        *,
        source_job_id: Optional[str] = None,
        source_simulation: Optional[str] = None,
        case: Optional[dict] = None,
        provenance: Optional[dict] = None,
        expected_kinds: Optional[Sequence[str]] = None,
        unit_system: Optional[UnitSystem] = None,
    ) -> CollectionResult:
        return self.collector.collect(
            source_dir,
            target_dir,
            source_job_id=source_job_id,
            source_simulation=source_simulation,
            backend="openlb",
            case=case,
            provenance=provenance,
            expected_kinds=expected_kinds,
            unit_system=unit_system,
        )

    @staticmethod
    def velocity_snapshots(
        manifest: DatasetManifest,
        *,
        dx: float = 1.0,
        fortran_order: bool = False,
    ) -> List[VelocitySnapshot]:
        base = Path(manifest.base_dir)
        paths = [base / item.path for item in manifest.files if item.kind == "velocity_field"]
        return load_velocity_snapshots(paths, dx=dx, fortran_order=fortran_order)


__all__ = ["OpenLBOutputAdapter"]
