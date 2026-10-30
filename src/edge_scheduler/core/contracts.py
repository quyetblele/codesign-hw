"""T04 contracts. Interfaces only: implementations belong to later tasks."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable
from .models import Device, Job
from .interfaces import ResourceSnapshot

if TYPE_CHECKING:
    from ..monitoring.monitoring_models import ResourceSample


class EventKind(str, Enum):
    ACCEPTED = "ACCEPTED"
    DISPATCHED = "DISPATCHED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class JobEvent:
    kind: EventKind
    job_id: str
    model_id: str
    timestamp_s: float
    device: Device | None = None
    top1_class: int | None = None
    error: str | None = None
    input_ref: str | None = None
    scheduler_overhead_s: float | None = None

    def __post_init__(self) -> None:
        if not self.job_id or not self.model_id:
            raise ValueError("event requires non-empty job_id and model_id")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be monotonic seconds >= 0")
        if self.kind is EventKind.FAILED and not self.error:
            raise ValueError("FAILED event requires error")
        if self.kind is EventKind.COMPLETED and self.error is not None:
            raise ValueError("COMPLETED event cannot include error")


@dataclass(frozen=True, slots=True)
class WorkerMessage:
    """WorkerLoop -> Controller. Controller alone mutates Job state.

    output contains the full inference result only for COMPLETED; JobEvent
    carries only summary fields and can be logged efficiently.
    """
    kind: EventKind
    job_id: str
    device: Device
    timestamp_s: float
    output: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in (EventKind.STARTED, EventKind.COMPLETED, EventKind.FAILED):
            raise ValueError("worker message must be STARTED/COMPLETED/FAILED")
        if not self.job_id or self.timestamp_s < 0:
            raise ValueError("invalid worker message id or timestamp")
        if self.kind is EventKind.FAILED and not self.error:
            raise ValueError("FAILED worker message requires error")
        if self.kind is EventKind.COMPLETED and self.error is not None:
            raise ValueError("COMPLETED message cannot include error")


@runtime_checkable
class InferenceBackend(Protocol):
    """One instance is pinned to a single Device. Output is a numpy ndarray.

    Model IDs: mobilenetv2, resnet50. input_ref: repository-relative image path.
    prepare performs model load/warmup; infer includes preprocessing and ONNX;
    close frees resources. Raise exceptions for errors; no silent CPU fallback.
    """
    @property
    def device(self) -> Device: ...

    def prepare(self) -> None: ...
    def infer(self, model_id: str, input_ref: str) -> Any: ...
    def close(self) -> None: ...


@runtime_checkable
class ResourceMonitor(Protocol):
    def snapshot(self) -> ResourceSnapshot: ...


@runtime_checkable
class ResourceSampleSink(Protocol):
    """Resource stream; MetricsCollector.record still takes JobEvent."""
    def record_resource(self, sample: ResourceSample) -> None: ...


@runtime_checkable
class MetricsCollector(Protocol):
    def record(self, event: JobEvent) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class ControllerPort(Protocol):
    """Entry boundary; concrete orchestration implementation belongs to T07."""
    def submit(self, job: Job) -> None: ...
    def handle_worker_message(self, message: WorkerMessage) -> None: ...
    def request_shutdown(self) -> None: ...
