from __future__ import annotations

from queue import Queue
from threading import Lock
from typing import Dict

from .models import Device, Job, JobStatus


class JobQueues:
    def __init__(self) -> None:
        self.incoming: Queue[Job] = Queue()
        self.device: Dict[Device, Queue[Job]] = {
            Device.CPU: Queue(),
            Device.GPU: Queue(),
        }
        self._accepted: set[str] = set()
        self._terminal: set[str] = set()
        self._lock = Lock()

    def accept(self, job: Job) -> None:
        with self._lock:
            if job.job_id in self._accepted:
                raise ValueError(f"duplicate job_id: {job.job_id}")
            self._accepted.add(job.job_id)
        job.mark_waiting()
        self.incoming.put(job)

    def dispatch(self, job: Job, device: Device) -> None:
        if job.status is not JobStatus.WAITING:
            raise ValueError(
                f"job {job.job_id} must be WAITING before dispatch"
            )
        job.mark_scheduled(device)
        self.device[device].put(job)

    def record_terminal(self, job: Job) -> None:
        if not job.terminal:
            raise ValueError(f"job {job.job_id} is not terminal")
        with self._lock:
            if job.job_id in self._terminal:
                raise ValueError(
                    f"job {job.job_id} already recorded terminal"
                )
            self._terminal.add(job.job_id)

    def reconcile(self) -> tuple[int, int]:
        with self._lock:
            return len(self._accepted), len(self._terminal)

    def is_reconciled(self) -> bool:
        accepted, terminal = self.reconcile()
        return accepted == terminal
