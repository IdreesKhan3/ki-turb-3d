"""End-to-end coordinator for the OpenLB HIT agent workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from agents.simulation_agent import SimulationAgent, SimulationSession
from agents.tools.physics.hit_validation_agent import HITValidationAgent
from agents.tools.report.hit_report_agent import HITReportAgent
from agents.tools.visualization_agent import HITVisualizationAgent
from postprocessing.hit_products_adapter import load_products_from_manifest
from postprocessing.pipeline import postprocess_manifest
from schemas import DatasetManifest
from schemas.hit_analysis_products import HITAnalysisProducts
from schemas.openlb_hit import OpenLBHITConfig


class HITWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    session: SimulationSession
    manifest_path: Optional[str] = None
    analysis_products_path: Optional[str] = None
    validation_path: Optional[str] = None
    visualization_dashboard: Optional[str] = None
    report_path: Optional[str] = None
    status: str


class HITMasterAgent:
    """Coordinate deterministic services; LLM orchestration delegates to these stages."""

    def __init__(
        self,
        *,
        simulation_agent: Optional[SimulationAgent] = None,
        validation_agent: Optional[HITValidationAgent] = None,
        visualization_agent: Optional[HITVisualizationAgent] = None,
        report_agent: Optional[HITReportAgent] = None,
    ) -> None:
        self.simulation_agent = simulation_agent or SimulationAgent()
        self.validation_agent = validation_agent or HITValidationAgent()
        self.visualization_agent = visualization_agent or HITVisualizationAgent()
        self.report_agent = report_agent or HITReportAgent()

    def prepare_and_compile(
        self,
        config: OpenLBHITConfig,
        *,
        run_root: str | Path,
        openlb_app_dir: str | Path,
    ) -> SimulationSession:
        session = self.simulation_agent.prepare(config, run_root)
        return self.simulation_agent.compile(session, openlb_app_dir)

    def run_and_collect(
        self,
        session: SimulationSession,
        *,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> DatasetManifest:
        self.simulation_agent.start(session)
        supervision = self.simulation_agent.supervise(
            session,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        if supervision.state != "completed":
            raise RuntimeError(f"OpenLB HIT run did not complete: {supervision.message or supervision.state}")
        return self.simulation_agent.collect(session).manifest

    def analyse(
        self,
        session: SimulationSession,
        manifest: DatasetManifest,
        *,
        requested_case_json: Optional[str | Path] = None,
    ) -> tuple[DatasetManifest, HITAnalysisProducts]:
        """Post-process a collected run and return canonical analysis products."""
        case_path = Path(requested_case_json) if requested_case_json else Path(session.case_dir) / "requested_case.json"
        processed_dir = Path(session.run_dir) / "processed"
        processed = postprocess_manifest(
            manifest,
            str(case_path),
            processed_dir=processed_dir,
        )
        products = load_products_from_manifest(processed)
        session.status = "analysed" if processed.status == "analysed" else "insufficient_data"
        session.collection_manifest = str(Path(processed.base_dir) / "dataset_manifest.json")
        session.metadata["analysis_products_path"] = processed.postprocessing.get("analysis_products_path")
        session.metadata["scientific_status"] = products.validation_status
        session.save()
        return processed, products

    def finalize(
        self,
        *,
        session: SimulationSession,
        products: Optional[HITAnalysisProducts] = None,
        manifest: Optional[DatasetManifest] = None,
    ) -> HITWorkflowResult:
        run_dir = Path(session.run_dir)
        analysis_dir = run_dir / "analysis"
        figures_dir = run_dir / "figures"
        report_dir = run_dir / "report"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        if products is None:
            if manifest is None:
                raise ValueError("manifest is required when products are not supplied")
            products = load_products_from_manifest(manifest)
        products.run_id = products.run_id or session.run_id
        products_path = products.save(analysis_dir / "hit_analysis_products.json")
        validation = self.validation_agent.validate(products, config=session.config, manifest=manifest)
        validation_path = analysis_dir / "hit_analysis_validation.json"
        validation_path.write_text(validation.model_dump_json(indent=2), encoding="utf-8")
        manifest_path = None
        if manifest is not None:
            manifest_path = Path(manifest.base_dir) / "dataset_manifest.json"
            manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        visualizations = self.visualization_agent.generate(
            products,
            figures_dir,
            manifest_path=manifest_path,
        )
        report = self.report_agent.generate(
            config=session.config,
            products=products,
            validation=validation,
            output_dir=report_dir,
            manifest=manifest,
            visualizations=visualizations,
        )
        session.status = "accepted" if validation.passed else "rejected"
        session.metadata["scientific_status"] = validation.metadata.get("status")
        session.save()
        return HITWorkflowResult(
            session=session,
            manifest_path=str(manifest_path) if manifest_path else session.collection_manifest,
            analysis_products_path=str(products_path),
            validation_path=str(validation_path),
            visualization_dashboard=visualizations.dashboard,
            report_path=report.html_path,
            status=session.status,
        )

    def execute_full_workflow(
        self,
        config: OpenLBHITConfig,
        *,
        run_root: str | Path,
        openlb_app_dir: str | Path,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> HITWorkflowResult:
        """Execute validation -> build -> run -> collect -> analyse -> visualize -> review."""
        session = self.prepare_and_compile(config, run_root=run_root, openlb_app_dir=openlb_app_dir)
        manifest = self.run_and_collect(session, poll_interval=poll_interval, timeout=timeout)
        manifest, products = self.analyse(session, manifest)
        return self.finalize(session=session, products=products, manifest=manifest)


__all__ = ["HITWorkflowResult", "HITMasterAgent"]
