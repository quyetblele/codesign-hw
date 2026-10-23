"""Thread-safe ownership and reconciliation of incoming/device Job queues."""
from __future__ import annotations

from queue import Queue
from threading import Lock
from typing import Dict

from ..core.models import Device, Job, JobStatus


class JobQueues:
    def __init__(self) -> None:
        self.incoming: Queue[Job] = Queue()
        self.device: Dict[Device, Queue[Job]] = {
            Device.CPU: Queue(),
            Device.GPU: Queue(),
        }
        self._accepted: set[str] = set()
        self._scheduled: set[str] = set()
        self._terminal: set[str] = set()
        self._lock = Lock()

    def accept(self, job: Job) -> None:
        with self._lock:
            if job.status is not JobStatus.CREATED:
                raise ValueError(f"job {job.job_id} must be CREATED before acceptance")
            if job.job_id in self._accepted:
                raise ValueError(f"duplicate job_id: {job.job_id}")
            job.mark_waiting()
            self.incoming.put_nowait(job)
            self._accepted.add(job.job_id)

    def dispatch(self, job: Job, device: Device) -> None:
        with self._lock:
            if device not in self.device:
                raise ValueError(f"unknown device: {device!r}")
            if job.job_id not in self._accepted:
                raise ValueError(f"job {job.job_id} was not accepted")
            if job.job_id in self._scheduled:
                raise ValueError(f"job {job.job_id} was already scheduled")
            if job.status is not JobStatus.WAITING:
                raise ValueError(f"job {job.job_id} must be WAITING before dispatch")
            job.mark_scheduled(device)
            self.device[device].put_nowait(job)
            self._scheduled.add(job.job_id)

    def queued_snapshot(self) -> dict[Device, tuple[Job, ...]]:
        """Immutable membership snapshots; excludes jobs currently being executed.

        Queue has no snapshot public method in Python 3.10; implementation
        details are encapsulated here rather than accessed by Controller.
        """
        result: dict[Device, tuple[Job, ...]] = {}
        for device, q in self.device.items():
            with q.mutex:
                result[device] = tuple(q.queue)
        return result

    def queue_lengths(self) -> dict[str, int]:
        """Observational counts, not an atomic snapshot with worker state."""
        return {"incoming": self.incoming.qsize(),
                **{device.value: q.qsize() for device, q in self.device.items()}}

    def record_terminal(self, job: Job) -> None:
        with self._lock:
            if job.job_id not in self._accepted:
                raise ValueError(f"unknown accepted job: {job.job_id}")
            if not job.terminal:
                raise ValueError(f"job {job.job_id} is not terminal")
            if job.job_id in self._terminal:
                raise ValueError(f"job {job.job_id} already recorded terminal")
            self._terminal.add(job.job_id)

    def reconcile(self) -> tuple[int, int]:
        with self._lock:
            return len(self._accepted), len(self._terminal)

    def is_reconciled(self) -> bool:
        accepted, terminal = self.reconcile()
        return accepted == terminal
