"""Week 2 orchestration. Controller is the sole owner of Job lifecycle state."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

from .contracts import EventKind, JobEvent, WorkerMessage
from .interfaces import Scheduler
from .models import Device, Job, JobStatus
from .queues import JobQueues


class Controller:
    def __init__(self, scheduler: Scheduler, backends: dict[Device, Any],
                 monitor: Any = None, collector: Any = None):
        if set(backends) != {Device.CPU, Device.GPU}:
            raise ValueError("one backend per CPU/GPU required")
        for device, backend in backends.items():
            if backend.device != device:
                raise ValueError("backend assigned to wrong device")
        self.scheduler = scheduler
        self.backends = backends
        self.monitor = monitor
        self.collector = collector
        self.queues = JobQueues()
        self.jobs: dict[str, Job] = {}
        self.messages: queue.Queue[WorkerMessage] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._started = False
        self._closed = False
        self._lock = threading.RLock()
        self.events: list[JobEvent] = []

    def _emit(self, job: Job, kind: EventKind, timestamp: float,
              error: str | None = None) -> None:
        event = JobEvent(kind=kind, job_id=job.job_id, model_id=job.model_id,
                         timestamp_s=timestamp, device=job.assigned_device, error=error)
        self.events.append(event)
        if self.collector is not None:
            self.collector.record(event)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("controller has been closed")
            if self._started:
                return
            prepared = []
            try:
                for backend in self.backends.values():
                    backend.prepare()
                    prepared.append(backend)
            except Exception:
                for backend in prepared:
                    backend.close()
                raise
            self._started = True
            for device in (Device.CPU, Device.GPU):
                worker = threading.Thread(
                    target=self._worker_loop, args=(device,),
                    name=f"{device.value}-worker", daemon=False)
                worker.start()
                self._workers.append(worker)

    def submit(self, job: Job) -> None:
        with self._lock:
            if not self._started or self._stop.is_set() or self._closed:
                raise RuntimeError("controller not accepting jobs")
            if job.status is not JobStatus.CREATED:
                raise ValueError("only CREATED jobs can be submitted")
            job.arrival_time = time.monotonic()
            self.queues.accept(job)
            self.jobs[job.job_id] = job
            self._emit(job, EventKind.ACCEPTED, job.arrival_time)

    def _queued_snapshot(self) -> dict[Device, list[Job]]:
        # Queue.mutex protects the underlying deque while copying; running jobs excluded.
        result: dict[Device, list[Job]] = {}
        for device, q in self.queues.device.items():
            with q.mutex:
                result[device] = list(q.queue)
        return result

    def dispatch_pending(self) -> int:
        with self._lock:
            if not self._started:
                raise RuntimeError("controller not started")
            count = 0
            while True:
                try:
                    job = self.queues.incoming.get_nowait()
                except queue.Empty:
                    break
                try:
                    snap = self.monitor.snapshot() if self.monitor is not None else None
                    device = self.scheduler.choose_device(job, self._queued_snapshot(), snap)
                    if device not in (Device.CPU, Device.GPU):
                        raise ValueError("scheduler returned invalid device")
                    self.queues.dispatch(job, device)
                    self._emit(job, EventKind.DISPATCHED, job.dispatch_time)
                    count += 1
                except Exception as exc:
                    if not job.terminal:
                        job.mark_failed(f"dispatch: {exc}")
                        self.queues.record_terminal(job)
                        self._emit(job, EventKind.FAILED, job.finish_time, job.error)
                finally:
                    self.queues.incoming.task_done()
            return count

    def _worker_loop(self, device: Device) -> None:
        q = self.queues.device[device]
        backend = self.backends[device]
        while True:
            try:
                job = q.get(timeout=0.05)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                self.messages.put(WorkerMessage(EventKind.STARTED, job.job_id, device,
                                                time.monotonic()))
                output = backend.infer(job.model_id, job.input_ref)
                self.messages.put(WorkerMessage(EventKind.COMPLETED, job.job_id, device,
                                                time.monotonic(), output=output))
            except Exception as exc:
                self.messages.put(WorkerMessage(EventKind.FAILED, job.job_id, device,
                                                time.monotonic(), error=str(exc) or type(exc).__name__))
            finally:
                q.task_done()

    def handle_worker_message(self, message: WorkerMessage) -> None:
        with self._lock:
            job = self.jobs.get(message.job_id)
            if job is None:
                raise ValueError("unknown job")
            if job.assigned_device is not message.device:
                raise ValueError("message device mismatch")
            if job.terminal:
                raise ValueError("duplicate terminal/late worker message")
            if message.kind is EventKind.STARTED:
                if job.status is not JobStatus.SCHEDULED:
                    raise ValueError("unexpected STARTED")
                job.mark_running(message.timestamp_s)
                self._emit(job, EventKind.STARTED, message.timestamp_s)
            elif message.kind is EventKind.COMPLETED:
                if job.status is not JobStatus.RUNNING:
                    raise ValueError("unexpected COMPLETED")
                job.mark_completed(message.output, message.timestamp_s)
                self.queues.record_terminal(job)
                self._emit(job, EventKind.COMPLETED, message.timestamp_s)
            elif message.kind is EventKind.FAILED:
                if job.status not in (JobStatus.SCHEDULED, JobStatus.RUNNING):
                    raise ValueError("unexpected FAILED")
                job.mark_failed(message.error or "backend failed", message.timestamp_s)
                self.queues.record_terminal(job)
                self._emit(job, EventKind.FAILED, message.timestamp_s, job.error)
            else:
                raise ValueError("unsupported worker message")

    def drain_events(self) -> int:
        count = 0
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                return count
            try:
                self.handle_worker_message(message)
                count += 1
            finally:
                self.messages.task_done()

    def wait_all(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.dispatch_pending()
            self.drain_events()
            if self.queues.is_reconciled() and not self.messages.qsize() and self.queues.incoming.empty():
                return True
            time.sleep(0.005)
        self.drain_events()
        return self.queues.is_reconciled()

    def request_shutdown(self) -> None:
        with self._lock:
            self._stop.set()

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        complete = self.wait_all(timeout) if self._started else True
        if not complete:
            raise TimeoutError("unfinished jobs; workers retained to avoid losing results")
        self.request_shutdown()
        for worker in self._workers:
            worker.join(timeout=timeout)
        if any(w.is_alive() for w in self._workers):
            raise TimeoutError("worker did not stop")
        self.drain_events()
        for backend in self.backends.values():
            backend.close()
        if self.collector is not None:
            self.collector.flush()
            self.collector.close()
        self._closed = True
