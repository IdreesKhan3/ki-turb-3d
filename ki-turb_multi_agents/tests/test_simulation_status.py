from agents.tools.simulation._status import (
    format_job_error,
    tool_text_indicates_failure,
)
from schemas.simulation_job import JobPaths, JobStatus, SimulationJob


def _job(status: JobStatus, message: str = "") -> SimulationJob:
    return SimulationJob(
        job_id="job_test",
        backend="openlb",
        status=status,
        message=message,
        paths=JobPaths(),
    )


def test_tool_text_indicates_failure_for_rejected_status_line():
    text = "Simulation stopped.\njob_id: job_x\nstatus: rejected\nmessage: Mach too high"
    assert tool_text_indicates_failure(text)


def test_format_job_error_includes_status_and_message():
    job = _job(JobStatus.REJECTED, "simulation health rejection: Mach too high")
    err = format_job_error(job, action="fetch outputs")
    assert "rejected" in err
    assert "Mach too high" in err
    assert err.startswith("Error:")
