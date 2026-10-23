"""Immutable telemetry and a state adapter for T07, without Job ownership."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from typing import Mapping

from ..core.interfaces import ResourceSnapshot, WorkerState
from ..core.models import Device

MIB = 1024 ** 2
SYSTEM_METRICS = (
    "cpu_load_pct", "ram_total_mb", "ram_free_mb", "ram_used_pct",
    "process_rss_mb", "process_cpu_pct",
)
GPU_METRICS = (
    "gpu_memory_total_mb", "gpu_memory_used_mb", "gpu_memory_free_mb", "gpu_utilization_pct",
)
METRICS = SYSTEM_METRICS + GPU_METRICS


@dataclass(frozen=True, slots=True)
class ProbeReading:
    values: Mapping[str, float | None]
    errors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))


@dataclass(frozen=True, slots=True)
class ResourceSample:
    sequence: int
    timestamp_s: float
    sample_start_s: float
    sample_duration_s: float
    values: Mapping[str, float | None]
    errors: Mapping[str, str] = field(default_factory=dict)
    workers: Mapping[Device, WorkerState] = field(default_factory=dict)
    queue_lengths: Mapping[str, int] = field(default_factory=dict)
    state_timestamp_s: float | None = None
    age_s: float | None = None
    stale: bool = True
    phase: str | None = None

    def __post_init__(self):
        for name in ("values", "errors", "workers", "queue_lengths"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            self.timestamp_s, self.values.get("cpu_load_pct"),
            self.values.get("ram_free_mb"), self.values.get("gpu_memory_free_mb"), self.workers,
        )


class WorkerRegistry:
    """T07/harness writes facts; monitor reads copies under one lock.

    available means ready/healthy, including busy workers with a device queue.
    No estimates are invented while running; T11 may provide a richer source.
    """
    def __init__(self):
        self._lock = Lock()
        self._states = {device: WorkerState(device, False) for device in Device}

    def set_ready(self, device: Device) -> None:
        with self._lock:
            self._states[device] = WorkerState(device, True, estimated_remaining_known=True)

    def set_running(self, device: Device, model_id: str, started_s: float) -> None:
        with self._lock:
            if not self._states[device].available:
                raise RuntimeError(f"worker {device.value} is not ready")
            self._states[device] = WorkerState(
                device, True, running_model_id=model_id, running_started_s=started_s,
            )

    def set_unavailable(self, device: Device) -> None:
        with self._lock:
            self._states[device] = WorkerState(device, False)

    def snapshot(self) -> Mapping[Device, WorkerState]:
        with self._lock:
            return MappingProxyType(dict(self._states))
