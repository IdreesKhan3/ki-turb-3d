"""Generate a compact scientific report for one OpenLB HIT run."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.tools.visualization_agent import VisualizationResult
from schemas import DatasetManifest, ValidationReport
from schemas.hit_analysis_products import HITAnalysisProducts
from schemas.openlb_hit import OpenLBHITConfig


class ReportResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    markdown_path: str
    html_path: str
    status: str
    included_figures: List[str] = Field(default_factory=list)


class HITReportAgent:
    def generate(
        self,
        *,
        config: OpenLBHITConfig,
        products: HITAnalysisProducts,
        validation: ValidationReport,
        output_dir: str | Path,
        manifest: Optional[DatasetManifest] = None,
        visualizations: Optional[VisualizationResult] = None,
    ) -> ReportResult:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        status = "ACCEPTED" if validation.passed else "REJECTED"
        markdown = self._markdown(
            config=config,
            products=products,
            validation=validation,
            status=status,
            manifest=manifest,
            visualizations=visualizations,
            report_dir=target,
        )
        markdown_path = target / "hit_validation_report.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        html_path = target / "hit_validation_report.html"
        html_path.write_text(self._html(markdown), encoding="utf-8")
        figures = [artifact.path for artifact in (visualizations.artifacts if visualizations else [])]
        return ReportResult(
            markdown_path=str(markdown_path),
            html_path=str(html_path),
            status=status,
            included_figures=figures,
        )

    def _markdown(
        self,
        *,
        config: OpenLBHITConfig,
        products: HITAnalysisProducts,
        validation: ValidationReport,
        status: str,
        manifest: Optional[DatasetManifest],
        visualizations: Optional[VisualizationResult],
        report_dir: Path,
    ) -> str:
        derived = config.derive_scaling()
        lines = [
            f"# KI-TURB OpenLB HIT validation report",
            "",
            f"**Case:** {config.name}",
            f"**Run ID:** {products.run_id or 'unknown'}",
            f"**Decision:** {status}",
            "",
            "## Requested and effective configuration",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Resolution | {config.domain.resolution} |",
            f"| Domain size | {config.domain.size} |",
            f"| Lattice | {config.domain.lattice.value} |",
            f"| Collision | {config.collision.model.value} |",
            f"| Forcing | {config.forcing.type.value} |",
            f"| Reynolds number | {derived.reynolds_number:.8g} |",
            f"| Physical viscosity | {derived.physical_viscosity:.8g} |",
            f"| Relaxation time | {derived.relaxation_time:.8g} |",
            f"| Actual Mach | {derived.actual_mach:.8g} |",
            f"| Lattice timestep | {derived.dt:.8g} |",
            "",
            "## Scientific checks",
            "",
            "| Check | Severity | Result | Details |",
            "|---|---|---|---|",
        ]
        for check in validation.checks:
            result = "PASS" if check.passed else "FAIL"
            details = check.message.replace("|", "\\|")
            lines.append(f"| {check.name} | {check.severity} | {result} | {details} |")

        lines.extend(["", "## Analysis products", ""])
        lines.append(f"- Energy spectra: {len(products.spectra)}")
        lines.append(f"- Spectral isotropy products: {len(products.spectral_isotropy)}")
        lines.append(f"- Reynolds-stress products: {len(products.reynolds_stress)}")
        lines.append(f"- Structure-function products: {len(products.structure_functions)}")
        lines.append(f"- PDFs: {len(products.pdfs)}")
        lines.append(f"- Flatness products: {len(products.flatness)}")
        if products.stationarity:
            lines.append(f"- Stationarity: {products.stationarity.stationary} — {products.stationarity.reason}")
        if products.resolution:
            lines.append(f"- Minimum kmax·eta: {products.resolution.kmax_eta_min}")
        if products.energy_balance:
            lines.append(f"- Mean relative energy-balance error: {products.energy_balance.relative_error_mean}")

        if manifest is not None:
            lines.extend(["", "## Dataset", ""])
            lines.append(f"- Manifest ID: `{manifest.manifest_id}`")
            lines.append(f"- Backend: `{manifest.backend}`")
            lines.append(f"- Files: {len(manifest.files)}")
            lines.append(f"- Time steps: {len(manifest.time_steps)}")

        if visualizations and visualizations.artifacts:
            lines.extend(["", "## Figures", ""])
            import os
            for artifact in visualizations.artifacts:
                relative = Path(os.path.relpath(Path(artifact.path), report_dir)).as_posix()
                lines.append(f"### {artifact.name.replace('_', ' ').title()}")
                lines.append("")
                lines.append(f"![{artifact.name}]({relative})")
                lines.append("")

        if products.warnings:
            lines.extend(["## Warnings", ""])
            lines.extend(f"- {warning}" for warning in products.warnings)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _html(markdown: str) -> str:
        try:
            import markdown as markdown_module  # type: ignore
        except ImportError:
            body = f"<pre>{html.escape(markdown)}</pre>"
        else:
            body = markdown_module.markdown(markdown, extensions=["tables", "fenced_code"])
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>KI-TURB HIT validation report</title>"
            "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:1.5rem;}"
            "table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ccc;padding:.5rem;}"
            "img{max-width:100%;height:auto;}</style></head><body>"
            + body
            + "</body></html>"
        )


__all__ = ["ReportResult", "HITReportAgent"]
