"""Solver-neutral bridge from dataset manifests to agent analysis caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from analysis.product_loader import AnalysisProductLoader
from .._shared import save_to_cache


CACHE_KEY_PRODUCTS = "current_analysis_products"
CACHE_KEY_OVERVIEW = "current_overview_validation"


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "load_analysis_products",
            "description": (
                "Load the canonical analysis-product bundle for the active dataset "
                "(hit_analysis_products.json / dataset manifest). Works for any backend "
                "after fetch+postprocess. Stores products in session cache for analyst/visualizer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "manifest_path": {
                        "type": "string",
                        "description": "Optional path to dataset_manifest.json. Uses session manifest when omitted.",
                    },
                    "data_dir": {
                        "type": "string",
                        "description": "Optional run base directory when no manifest is loaded yet.",
                    },
                },
            },
        },
        {
            "name": "get_analysis_product_summary",
            "description": (
                "Summarize loaded analysis products: validation status, available spectra, "
                "isotropy, PDFs, time history, divergence, Re_lambda, kmax*eta. "
                "Informational only unless the user asks for pass/fail on divergence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_reference": {
                        "type": "string",
                        "description": "Cache key (default current_analysis_products).",
                    },
                },
            },
        },
    ]


def _loader(
    project_root: Path,
    session_context: Dict[str, Any],
    args: Dict[str, Any],
) -> AnalysisProductLoader:
    manifest_path = args.get("manifest_path") or session_context.get("manifest_path")
    if manifest_path:
        return AnalysisProductLoader.from_manifest_path(project_root, manifest_path, session_context)
    if args.get("data_dir"):
        session_context["data_directory"] = args["data_dir"]
    return AnalysisProductLoader(project_root, session_context)


def execute_tool(
    name: str,
    args: Dict[str, Any],
    project_root: Path,
    session_context: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    session_context = session_context or {}

    if name == "load_analysis_products":
        loader = _loader(project_root, session_context, args)
        products = loader.products(reload=True)
        if products is None:
            return (
                "Error: No analysis products found. Run postprocess_simulation_outputs first, "
                "or load_dataset_manifest for a completed run."
            )
        payload = products.model_dump(mode="json")
        save_to_cache(session_context, CACHE_KEY_PRODUCTS, payload)
        session_context["analysis_products"] = payload
        manifest = loader.manifest()
        summary = {
            "validation_status": products.validation_status,
            "backend": loader.backend(),
            "spectra": len(products.spectra),
            "spectral_isotropy": len(products.spectral_isotropy),
            "reynolds_stress": len(products.reynolds_stress),
            "pdfs": len(products.pdfs),
            "flatness": len(products.flatness),
            "structure_functions": len(products.structure_functions),
            "has_time_history": products.time_history is not None,
        }
        if manifest is not None:
            session_context["dataset_manifest"] = manifest.model_dump(mode="json")
        return {
            "status": "success",
            "message": f"Loaded analysis products (status={products.validation_status}).",
            "cache_key": CACHE_KEY_PRODUCTS,
            "summary": summary,
        }

    if name == "get_analysis_product_summary":
        from .._shared import get_from_cache

        cached = get_from_cache(session_context, args.get("data_reference", CACHE_KEY_PRODUCTS))
        if cached is None:
            cached = session_context.get("analysis_products")
        if cached is None:
            loader = AnalysisProductLoader(project_root, session_context)
            products = loader.products()
            if products is None:
                return "Error: No analysis products in cache. Call load_analysis_products first."
            cached = products.model_dump(mode="json")

        lines = [
            "# Analysis Product Summary",
            f"- validation_status: {cached.get('validation_status', 'unknown')}",
            f"- spectra: {len(cached.get('spectra') or [])}",
            f"- spectral_isotropy: {len(cached.get('spectral_isotropy') or [])}",
            f"- reynolds_stress: {len(cached.get('reynolds_stress') or [])}",
            f"- pdfs: {len(cached.get('pdfs') or [])}",
            f"- flatness: {len(cached.get('flatness') or [])}",
            f"- structure_functions: {len(cached.get('structure_functions') or [])}",
        ]
        history = cached.get("time_history") or {}
        if history.get("divergence_rms"):
            lines.append(f"- divergence_rms_max: {max(history['divergence_rms']):g}")
        if history.get("mach_max"):
            lines.append(f"- mach_max: {max(history['mach_max']):g}")
        if history.get("re_lambda"):
            lines.append(f"- re_lambda_last: {history['re_lambda'][-1]:g}")
        if history.get("kmax_eta"):
            lines.append(f"- kmax_eta_min: {min(history['kmax_eta']):g}")
        resolution = cached.get("resolution") or {}
        if resolution.get("kmax_eta_min") is not None:
            lines.append(f"- kmax_eta_min (resolution): {resolution['kmax_eta_min']:g}")
        return {
            "status": "success",
            "artifact_type": "markdown",
            "artifact_content": "\n".join(lines),
            "artifact_title": "Analysis Products",
        }

    return f"Error: Unknown analysis-products tool '{name}'"
