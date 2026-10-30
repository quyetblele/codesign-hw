from __future__ import annotations

from dataclasses import dataclass, field
import time
import json
import math
from pathlib import Path
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

    @classmethod
    def from_file(cls, path: str | Path) -> ServiceTimeProfile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("run_type") != "profiling" or data.get("status") != "passed":
            raise ValueError("requires a successful profiling run")
        if data.get("trace_enabled"):
            raise ValueError("traced timings are for verification, not calibration")
        entries = {}
        for row in data["profiles"]:
            key = (row["model_id"], Device(row["device"]))
            if key in entries or row.get("status") != "passed":
                raise ValueError("duplicate or incomplete profile")
            entries[key] = row["mean_service_s"]
        if not entries:
            raise ValueError("empty service-time profile")
        profile = cls(entries)
        for model, device in entries:
            profile.estimate(model, device)
        return profile

    def estimate(self, model_id: str, device: Device) -> float:
        try:
            value = float(self.seconds[(model_id, device)])
        except KeyError as exc:
            raise KeyError(
                f"missing service-time profile for {model_id}/{device.value}"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("service time must be finite and > 0")
        return value


class ResourceAwareScheduler:
    """Estimated-finish-delay scheduler from PLAN.md."""

    def __init__(
        self,
        profile: ServiceTimeProfile,
        tie_breaker: Device = Device.CPU,
        max_snapshot_age_s: float = 1.0,
    ) -> None:
        self.profile = profile
        self.tie_breaker = tie_breaker
        self.max_snapshot_age_s = max_snapshot_age_s
        self.decisions: list[dict] = []

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
                if not worker.available:
                    return float("inf")
                if worker.running_model_id is not None:
                    if worker.estimated_remaining_known:
                        remaining = max(0.0, worker.estimated_remaining_s)
                    elif worker.running_started_s is not None:
                        elapsed = max(0.0, time.monotonic() - worker.running_started_s)
                        remaining = max(0.0, self.profile.estimate(worker.running_model_id, device) - elapsed)
                    else:
                        remaining = self.profile.estimate(worker.running_model_id, device)

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
        if snapshot is not None and (not math.isfinite(snapshot.timestamp) or
            time.monotonic() - snapshot.timestamp > self.max_snapshot_age_s):
            snapshot = None
        cpu = self.estimated_finish_delay(
            Device.CPU, job, queued_jobs, snapshot
        )
        gpu = self.estimated_finish_delay(
            Device.GPU, job, queued_jobs, snapshot
        )
        if not math.isfinite(cpu) and not math.isfinite(gpu):
            raise RuntimeError("no eligible device for scheduling")
        chosen = self.tie_breaker if cpu == gpu else Device.CPU if cpu < gpu else Device.GPU
        self.decisions.append({"job_id": job.job_id, "model_id": job.model_id,
                               "cpu_eta_s": cpu if math.isfinite(cpu) else None,
                               "gpu_eta_s": gpu if math.isfinite(gpu) else None,
                               "selected_device": chosen.value,
                               "timestamp_s": time.monotonic()})
        return chosen
