"""Verify tool outcomes against expected world-state transitions."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from agents.tools.simulation._status import tool_text_indicates_failure

from .workflow_world import WorkflowWorldState


class VerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    reason: str = ""
    expected: str = ""
    observed: str = ""


def _payload_failed(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, dict):
        status = str(raw.get("status") or "").lower()
        if status in {"error", "failed", "failure"}:
            return True
        msg = str(raw.get("message") or raw.get("error") or "")
        return tool_text_indicates_failure(msg)
    return tool_text_indicates_failure(str(raw))


def verify_step(
    tool_name: str,
    raw: Any,
    *,
    before: WorkflowWorldState,
    after: WorkflowWorldState,
    intent_action: Optional[str] = None,
) -> VerifyResult:
    """Check that a completed tool call moved the world as expected."""
    del before  # reserved for richer delta checks
    name = (tool_name or "").strip()
    if _payload_failed(raw):
        return VerifyResult(
            ok=False,
            reason="tool reported failure",
            expected="success",
            observed=str(raw)[:500],
        )

    if name == "build_simulation_case":
        if not after.has_job_record and not after.job_id:
            # Job id may only appear in tool text until session remembers it.
            text = str(raw)
            if "job_id:" not in text.lower():
                return VerifyResult(
                    ok=False,
                    reason="build did not produce a job_id",
                    expected="job_id in result",
                    observed=text[:300],
                )
        return VerifyResult(ok=True, reason="build produced a job")

    if name == "compile_simulation":
        if after.job_status not in {"built", "compiled"} and not after.has_executable:
            return VerifyResult(
                ok=False,
                reason="compile did not yield built/compiled state",
                expected="status built|compiled or executable present",
                observed=f"status={after.job_status} executable={after.has_executable}",
            )
        return VerifyResult(ok=True, reason="compile ok")

    if name == "start_simulation":
        if after.job_status not in {"running", "queued", "submitted"}:
            return VerifyResult(
                ok=False,
                reason="start did not leave job running",
                expected="status running|queued|submitted",
                observed=f"status={after.job_status}",
            )
        return VerifyResult(ok=True, reason="start ok")

    if name == "fetch_simulation_outputs":
        if not after.has_manifest:
            return VerifyResult(
                ok=False,
                reason="fetch did not produce manifest.json",
                expected="has_manifest=True",
                observed=f"has_manifest={after.has_manifest}",
            )
        return VerifyResult(ok=True, reason="fetch ok")

    if name == "postprocess_simulation_outputs":
        if after.job_status not in {
            "analysis_ready",
            "postprocessed",
            "analysed",
            "insufficient_data",
        } and not after.has_processed_products:
            return VerifyResult(
                ok=False,
                reason="postprocess did not produce analysis products",
                expected="analysis_ready or processed/",
                observed=(
                    f"status={after.job_status} processed={after.has_processed_products}"
                ),
            )
        return VerifyResult(ok=True, reason="postprocess ok")

    if name == "load_dataset_manifest":
        if intent_action in {"load", "analyze", "run"} and not (
            after.session_has_data or after.session_manifest_path or after.has_manifest
        ):
            return VerifyResult(
                ok=False,
                reason="load did not populate session data",
                expected="session data or manifest path",
                observed=(
                    f"session_has_data={after.session_has_data} "
                    f"manifest={after.session_manifest_path}"
                ),
            )
        return VerifyResult(ok=True, reason="load ok")

    return VerifyResult(ok=True, reason="no specific verifier; tool succeeded")


__all__ = ["VerifyResult", "verify_step"]
