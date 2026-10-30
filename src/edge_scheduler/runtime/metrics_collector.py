"""T10: validate lifecycle events, stream logs and expose cheap live snapshots."""
from __future__ import annotations

from collections import deque
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time

from ..core.contracts import EventKind, JobEvent
from ..core.models import Job
from .run_results import distribution, job_row, summarize_rows


class MetricsCollector:
    def __init__(self, directory: Path, *, capacity: int = 8192):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._lock = Lock()
        self._queue = Queue(maxsize=capacity)
        self._stop = Event()
        self._closed = False
        self._error = None
        self._written = self._dropped = 0
        self._rows: dict[str, dict] = {}
        self._recent = deque()
        self._accepted = self._completed = self._failed = 0
        self._first_arrival = None
        self._events_stream = (directory / "events.jsonl").open("x", encoding="utf-8")
        try:
            self._jobs_stream = (directory / "jobs.csv").open("x", encoding="utf-8", newline="")
        except BaseException:
            self._events_stream.close()
            raise
        self._thread = Thread(target=self._run, name="metrics-writer", daemon=True)
        self._thread.start()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {"written_events": self._written, "dropped_events": self._dropped,
                    "error": self._error, "closed": self._closed,
                    "running": self._thread.is_alive()}

    def record(self, event: JobEvent) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("metrics collector is closed")
            if not math.isfinite(event.timestamp_s) or event.timestamp_s < 0:
                raise ValueError("event timestamp must be finite and >= 0")
            if event.scheduler_overhead_s is not None and (
                not math.isfinite(event.scheduler_overhead_s) or event.scheduler_overhead_s < 0
            ):
                raise ValueError("scheduler overhead must be finite and >= 0")
            row = self._rows.get(event.job_id)
            kind = event.kind
            if kind is EventKind.ACCEPTED:
                if row is not None:
                    raise ValueError("duplicate accepted job")
                row = job_row(Job(event.job_id, event.model_id, event.input_ref))
                row.update(status="WAITING", arrival_time_s=event.timestamp_s)
                self._accepted += 1
                self._first_arrival = (event.timestamp_s if self._first_arrival is None
                                       else min(self._first_arrival, event.timestamp_s))
            else:
                if row is None:
                    raise ValueError("event for unknown job")
                row = dict(row)
                allowed = {EventKind.DISPATCHED: ("WAITING",),
                           EventKind.STARTED: ("SCHEDULED",),
                           EventKind.COMPLETED: ("RUNNING",),
                           EventKind.FAILED: ("WAITING", "SCHEDULED", "RUNNING")}
                if row["status"] not in allowed.get(kind, ()) or event.model_id != row["model_id"]:
                    raise ValueError("invalid or duplicate lifecycle event")
                previous = next(row[key] for key in ("start_time_s", "dispatch_time_s", "arrival_time_s")
                                if row[key] is not None)
                if event.timestamp_s < previous:
                    raise ValueError("lifecycle timestamp moved backwards")
                if row["device"] is not None and (event.device is None or event.device.value != row["device"]):
                    raise ValueError("event device mismatch")
                if kind in (EventKind.DISPATCHED, EventKind.STARTED, EventKind.COMPLETED) and event.device is None:
                    raise ValueError("device is required after dispatch")
                row["device"] = event.device.value if event.device else row["device"]
                if kind is EventKind.DISPATCHED:
                    row.update(status="SCHEDULED", dispatch_time_s=event.timestamp_s)
                elif kind is EventKind.STARTED:
                    row.update(status="RUNNING", start_time_s=event.timestamp_s)
                else:
                    row.update(status=kind.value, finish_time_s=event.timestamp_s,
                               error=event.error, top1_class=event.top1_class)
                    if kind is EventKind.COMPLETED:
                        self._completed += 1
                        self._recent.append((event.timestamp_s, event.timestamp_s - row["arrival_time_s"]))
                    else:
                        self._failed += 1
            row["scheduler_overhead_s"] = event.scheduler_overhead_s
            for name, start, finish in (
                ("dispatch_wait_s", "arrival_time_s", "dispatch_time_s"),
                ("device_wait_s", "dispatch_time_s", "start_time_s"),
                ("total_wait_s", "arrival_time_s", "start_time_s"),
                ("execution_s", "start_time_s", "finish_time_s"),
                ("latency_s", "arrival_time_s", "finish_time_s"),
            ):
                row[name] = row[finish] - row[start] if row[start] is not None and row[finish] is not None else None
            self._rows[event.job_id] = row
            if self._error:
                self._dropped += 1
                raise RuntimeError(self._error)
            try:
                self._queue.put_nowait((asdict(event), dict(row) if kind in (
                    EventKind.COMPLETED, EventKind.FAILED) else None))
            except Full:
                self._dropped += 1
                self._error = "metrics queue full; event log is incomplete"
                raise RuntimeError(self._error)

    def snapshot(self, window_s: float = 5.0, *, now: float | None = None) -> dict:
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window must be finite and > 0")
        now = time.monotonic() if now is None else now
        with self._lock:
            self._recent = deque((finished, latency) for finished, latency in self._recent
                                 if finished > now - window_s)
            recent = [latency * 1000 for finished, latency in self._recent if finished <= now]
            elapsed = max(0.0, now - self._first_arrival) if self._first_arrival is not None else 0.0
            duration = min(window_s, elapsed)
            result = {"accepted": self._accepted, "completed": self._completed, "failed": self._failed,
                      "unfinished": self._accepted - self._completed - self._failed,
                      "elapsed_s": elapsed, "window_s": window_s, "recent_duration_s": duration,
                      "recent_samples": len(recent),
                      "recent_throughput": len(recent) / duration if duration > 0 else None,
                      "error": self._error}
        result["recent_latency_ms"] = distribution(recent)
        return result

    def summary(self) -> dict:
        with self._lock:
            rows = [dict(row) for row in self._rows.values()]
            accepted, terminal = self._accepted, self._completed + self._failed
        return summarize_rows(rows, accepted, terminal)

    def _run(self) -> None:
        try:
            writer = csv.DictWriter(self._jobs_stream, fieldnames=tuple(job_row(Job("header", "model", "input"))))
            writer.writeheader()
            self._jobs_stream.flush()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                if isinstance(item, Event):
                    self._events_stream.flush()
                    self._jobs_stream.flush()
                    item.set()
                    continue
                event, row = item
                self._events_stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                if row is not None:
                    writer.writerow(row)
                # Make logs readable during the run without I/O in controller/worker.
                self._events_stream.flush()
                self._jobs_stream.flush()
                with self._lock:
                    self._written += 1
        except Exception as exc:
            with self._lock:
                self._error = self._error or f"{type(exc).__name__}: {exc}"
        finally:
            for stream in (self._events_stream, self._jobs_stream):
                try:
                    stream.close()
                except Exception as exc:
                    with self._lock:
                        self._error = self._error or f"close: {exc}"

    def flush(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            if self.stats["error"]:
                raise RuntimeError(self.stats["error"])
            return
        if self.stats["error"]:
            raise RuntimeError(self.stats["error"])
        barrier = Event()
        self._queue.put(barrier, timeout=timeout_s)
        deadline = time.monotonic() + timeout_s
        while not barrier.wait(0.05):
            if not self._thread.is_alive():
                raise RuntimeError(self.stats["error"] or "metrics writer stopped")
            if time.monotonic() >= deadline:
                raise TimeoutError("metrics flush timed out")

    def close(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("metrics writer did not stop")
        if self.stats["error"]:
            raise RuntimeError(self.stats["error"])
