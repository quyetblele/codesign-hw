from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class JobStatus(str, Enum):
    CREATED = "CREATED"
    WAITING = "WAITING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(slots=True)
class Job:
    job_id: str
    model_id: str
    input_ref: Any
    arrival_offset_s: float = 0.0
    priority: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    dispatch_time: Optional[float] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    assigned_device: Optional[Device] = None
    status: JobStatus = JobStatus.CREATED
    result: Any = None
    error: Optional[str] = None
    scheduler_overhead_s: Optional[float] = None

    def mark_waiting(self) -> None:
        self.status = JobStatus.WAITING

    def mark_scheduled(self, device: Device, now: Optional[float] = None) -> None:
        self.assigned_device = device
        self.dispatch_time = time.monotonic() if now is None else now
        self.status = JobStatus.SCHEDULED

    def mark_running(self, now: Optional[float] = None) -> None:
        self.start_time = time.monotonic() if now is None else now
        self.status = JobStatus.RUNNING

    def mark_completed(self, result: Any = None, now: Optional[float] = None) -> None:
        self.finish_time = time.monotonic() if now is None else now
        self.result = result
        self.status = JobStatus.COMPLETED

    def mark_failed(self, error: str, now: Optional[float] = None) -> None:
        self.finish_time = time.monotonic() if now is None else now
        self.error = error
        self.status = JobStatus.FAILED

    @property
    def terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)
