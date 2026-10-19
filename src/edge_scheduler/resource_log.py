"""Bounded, asynchronous resource CSV sink; no file I/O on snapshot readers."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread

from .monitoring_models import METRICS, ResourceSample

FIELDS = ("sequence", "timestamp_s", "sample_start_s", "sample_duration_s",
          "state_timestamp_s", "age_s", "stale", "phase") + METRICS + (
              "workers_json", "queue_lengths_json", "errors_json",
          )


class ResourceCSVLogger:
    def __init__(self, path: str | Path, *, capacity: int = 1024):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._queue = Queue(maxsize=capacity)
        self._lock = Lock()
        self._stop = Event()
        self._closed = False
        self._error: str | None = None
        self._written = self._dropped = 0
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("x", newline="", encoding="utf-8")
        self._thread = Thread(target=self._run, name="resource-csv-writer", daemon=True)
        self._thread.start()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {"written_samples": self._written, "dropped_samples": self._dropped,
                    "error": self._error, "closed": self._closed,
                    "running": self._thread.is_alive()}

    def record_resource(self, sample: ResourceSample) -> None:
        with self._lock:
            if self._error:
                raise RuntimeError(f"resource log failed: {self._error}")
            if self._closed:
                raise RuntimeError("resource log is closed")
            try:
                self._queue.put_nowait(sample)
            except Full:
                self._dropped += 1

    @staticmethod
    def _row(sample):
        row = {field: getattr(sample, field) for field in FIELDS[:8]}
        row.update(sample.values)
        row["workers_json"] = json.dumps({device.value: asdict(state) for device, state in sample.workers.items()})
        row["queue_lengths_json"] = json.dumps(dict(sample.queue_lengths))
        row["errors_json"] = json.dumps(dict(sample.errors))
        return row

    def _run(self):
        try:
            writer = csv.DictWriter(self._stream, fieldnames=FIELDS)
            writer.writeheader()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    sample = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                writer.writerow(self._row(sample))
                with self._lock:
                    self._written += 1
            self._stream.flush()
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}:{exc}"
        finally:
            try:
                self._stream.close()
            except Exception as exc:
                with self._lock:
                    self._error = self._error or f"close:{exc}"

    def close(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("resource CSV writer did not stop")
        if self._error:
            raise RuntimeError(f"resource log failed: {self._error}")
