"""Guarded transitions: tool may run only when world state permits it."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from .workflow_world import WorkflowWorldState

_ANALYZE_TOOLS = frozenset({
    "compute_spectra",
    "compute_flatness",
    "compute_structure_functions",
    "compute_spectral_isotropy",
    "compute_isotropy",
    "compute_pdfs",
    "compute_overview_validation",
    "compute_volume_field",
    "plot_spectrum",
    "plot_spectral_isotropy",
    "plot_real_isotropy",
    "plot_turbulence_stats",
    "plot_flatness",
    "plot_structure_functions",
    "plot_pdf",
    "plot_volume_3d",
})


class GuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason: str = ""
    required_capability: Optional[str] = None


def guard_tool(tool_name: str, world: WorkflowWorldState) -> GuardResult:
    """Return whether ``tool_name`` is legal given current world facts."""
    name = (tool_name or "").strip()
    if not name:
        return GuardResult(allowed=False, reason="missing tool name")

    if name == "build_simulation_case":
        return GuardResult(allowed=True, reason="build creates a new job", required_capability="build")

    if name == "compile_simulation":
        if not world.has_job_record:
            return GuardResult(
                allowed=False,
                reason="compile requires an existing prepared job (build first)",
                required_capability="compile",
            )
        if not world.capability("compile"):
            return GuardResult(
                allowed=False,
                reason=(
                    f"compile not allowed from status={world.job_status!r}; "
                    "expected prepared/built (or failed retry)"
                ),
                required_capability="compile",
            )
        return GuardResult(allowed=True, required_capability="compile")

    if name == "start_simulation":
        if not world.has_job_record:
            return GuardResult(
                allowed=False,
                reason="start requires an existing job (build+compile first)",
                required_capability="start",
            )
        if world.job_status in {"running", "queued", "submitted"}:
            return GuardResult(
                allowed=False,
                reason=f"job already {world.job_status}; use supervise/status instead of start",
                required_capability="start",
            )
        if not (world.capability("start") or world.has_executable):
            return GuardResult(
                allowed=False,
                reason=(
                    f"start not allowed from status={world.job_status!r}; "
                    "expected built/compiled executable"
                ),
                required_capability="start",
            )
        return GuardResult(allowed=True, required_capability="start")

    if name == "supervise_simulation":
        if not world.has_job_record:
            return GuardResult(allowed=False, reason="supervise requires an active job")
        if not world.capability("supervise") and world.job_status not in {
            "completed", "failed", "cancelled", "fetched", "analysis_ready"
        }:
            # Allow supervise to observe terminal jobs once (poll node handles advance).
            return GuardResult(
                allowed=False,
                reason=f"supervise not allowed from status={world.job_status!r}",
                required_capability="supervise",
            )
        return GuardResult(allowed=True, required_capability="supervise")

    if name in {"check_simulation_status", "cancel_simulation"}:
        if not world.has_job_record:
            return GuardResult(
                allowed=False,
                reason=f"{name} requires an existing job_id",
                required_capability="status",
            )
        return GuardResult(allowed=True, required_capability="status")

    if name == "fetch_simulation_outputs":
        if not world.has_job_record:
            return GuardResult(allowed=False, reason="fetch requires an existing job")
        if world.job_status == "running":
            return GuardResult(
                allowed=False,
                reason="fetch blocked while job is still running; supervise until complete",
                required_capability="fetch",
            )
        if not world.capability("fetch") and world.job_status not in {"completed", "fetched"}:
            return GuardResult(
                allowed=False,
                reason=f"fetch not allowed from status={world.job_status!r}; expected completed",
                required_capability="fetch",
            )
        return GuardResult(allowed=True, required_capability="fetch")

    if name == "postprocess_simulation_outputs":
        if not (world.has_manifest or world.capability("postprocess")):
            return GuardResult(
                allowed=False,
                reason="postprocess requires a fetched manifest",
                required_capability="postprocess",
            )
        return GuardResult(allowed=True, required_capability="postprocess")

    if name in {"load_dataset_manifest", "read_dataset_manifest"}:
        if not world.capability("load") and not world.has_manifest:
            return GuardResult(
                allowed=False,
                reason="load requires simulations/<job_id>/manifest.json (or session manifest path)",
                required_capability="load",
            )
        return GuardResult(allowed=True, required_capability="load")

    if name in _ANALYZE_TOOLS:
        if not world.capability("analyze"):
            return GuardResult(
                allowed=False,
                reason=(
                    "analyze requires loaded session data or a job with manifest/processed products"
                ),
                required_capability="analyze",
            )
        return GuardResult(allowed=True, required_capability="analyze")

    # Unknown / unconstrained tools (steward filesystem, etc.)
    return GuardResult(allowed=True, reason="no solver guard for this tool")


__all__ = ["GuardResult", "guard_tool"]
