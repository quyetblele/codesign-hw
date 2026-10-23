from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..core.interfaces import ResourceSnapshot
from ..core.models import Device, Job


class CpuOnlyScheduler:
    def choose_device(self, job: Job, queued_jobs, snapshot=None) -> Device:
        return Device.CPU


class GpuOnlyScheduler:
    def choose_device(self, job: Job, queued_jobs, snapshot=None) -> Device:
        return Device.GPU


class RoundRobinScheduler:
    def __init__(self, first: Device = Device.CPU) -> None:
        self._next = first

    def choose_device(self, job: Job, queued_jobs, snapshot=None) -> Device:
        chosen = self._next
        self._next = Device.GPU if chosen is Device.CPU else Device.CPU
        return chosen


@dataclass(slots=True)
class ServiceTimeProfile:
    seconds: Mapping[tuple[str, Device], float]

    def estimate(self, model_id: str, device: Device) -> float:
        try:
            value = float(self.seconds[(model_id, device)])
        except KeyError as exc:
            raise KeyError(
                f"missing service-time profile for {model_id}/{device.value}"
            ) from exc
        if value <= 0:
            raise ValueError("service time must be > 0")
        return value


class ResourceAwareScheduler:
    """Estimated-finish-delay scheduler from PLAN.md."""

    def __init__(
        self,
        profile: ServiceTimeProfile,
        tie_breaker: Device = Device.CPU,
    ) -> None:
        self.profile = profile
        self.tie_breaker = tie_breaker

    def estimated_finish_delay(
        self,
        device: Device,
        job: Job,
        queued_jobs: Mapping[Device, Sequence[Job]],
        snapshot: ResourceSnapshot | None = None,
    ) -> float:
        remaining = 0.0
        if snapshot is not None:
            worker = snapshot.workers.get(device)
            if worker is not None:
                if not worker.available and worker.estimated_remaining_s <= 0:
                    return float("inf")
                remaining = max(0.0, worker.estimated_remaining_s)

        queued = sum(
            self.profile.estimate(q.model_id, device)
            for q in queued_jobs.get(device, ())
        )
        own = self.profile.estimate(job.model_id, device)
        return remaining + queued + own

    def choose_device(
        self,
        job: Job,
        queued_jobs: Mapping[Device, Sequence[Job]],
        snapshot: ResourceSnapshot | None = None,
    ) -> Device:
        cpu = self.estimated_finish_delay(
            Device.CPU, job, queued_jobs, snapshot
        )
        gpu = self.estimated_finish_delay(
            Device.GPU, job, queued_jobs, snapshot
        )
        if cpu == gpu:
            return self.tie_breaker
        return Device.CPU if cpu < gpu else Device.GPU
