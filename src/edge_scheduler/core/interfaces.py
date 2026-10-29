from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .models import Device, Job


@dataclass(frozen=True, slots=True)
class WorkerState:
    device: Device
    available: bool
    running_model_id: str | None = None
    estimated_remaining_s: float = 0.0
    memory_free_mb: float | None = None  # Decimal MB (bytes / 1_000_000).
    # T08 additions: legacy 0.0 alone does not imply a known runtime estimate.
    running_started_s: float | None = None
    estimated_remaining_known: bool = False


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    timestamp: float
    cpu_load_pct: float | None
    ram_free_mb: float | None
    gpu_memory_free_mb: float | None
    workers: Mapping[Device, WorkerState]


class Scheduler(Protocol):
    def choose_device(
        self,
        job: Job,
        queued_jobs: Mapping[Device, Sequence[Job]],
        snapshot: ResourceSnapshot | None = None,
    ) -> Device:
        ...
