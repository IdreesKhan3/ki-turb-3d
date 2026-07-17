"""Remote execution interface for HPC and cloud backends.

Defines the contract for running solver jobs on a remote host (SSH, a batch
scheduler such as SLURM, or a cloud service). No transport is implemented yet;
concrete runners will subclass :class:`RemoteRunner`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from schemas import DatasetManifest, SimulationJob


class RemoteExecutionError(RuntimeError):
    """Base error for remote execution failures."""


class RemoteRunner(ABC):
    """Submit and manage solver jobs on a remote host or scheduler."""

    name: str = "remote"

    @abstractmethod
    def submit(self, job: SimulationJob) -> SimulationJob:
        """Transfer inputs and enqueue the job on the remote host."""

    @abstractmethod
    def poll(self, job: SimulationJob) -> SimulationJob:
        """Query the remote scheduler and return the updated job."""

    @abstractmethod
    def cancel(self, job: SimulationJob) -> SimulationJob:
        """Cancel the remote job."""

    @abstractmethod
    def fetch(self, job: SimulationJob) -> DatasetManifest:
        """Retrieve outputs from the remote host and return a manifest."""


class NotImplementedRemoteRunner(RemoteRunner):
    """Placeholder runner that reports remote execution is not yet available."""

    name = "unconfigured_remote"

    def submit(self, job: SimulationJob) -> SimulationJob:
        raise RemoteExecutionError("remote execution is not implemented yet")

    def poll(self, job: SimulationJob) -> SimulationJob:
        raise RemoteExecutionError("remote execution is not implemented yet")

    def cancel(self, job: SimulationJob) -> SimulationJob:
        raise RemoteExecutionError("remote execution is not implemented yet")

    def fetch(self, job: SimulationJob) -> DatasetManifest:
        raise RemoteExecutionError("remote execution is not implemented yet")


def available_runners() -> List[str]:
    return []
