"""Background hardware sampling with a fast, immutable T04 snapshot boundary."""
from __future__ import annotations

from dataclasses import replace
import math
from threading import Event, Lock, Thread
import time
from typing import Callable, Mapping

from .contracts import ResourceSampleSink
from .gpu_probe import GPUProbe
from .interfaces import ResourceSnapshot, WorkerState
from .models import Device
from .monitoring_models import GPU_METRICS, METRICS, SYSTEM_METRICS, ProbeReading, ResourceSample
from .system_probe import SystemProbe


class ResourceMonitor:
    def __init__(self, *, interval_s: float = 0.2, max_age_s: float = 1.0,
                 system_probe=None, gpu_probe=None,
                 worker_state_provider: Callable[[], Mapping[Device, WorkerState]] | None = None,
                 queue_lengths_provider: Callable[[], Mapping[str, int]] | None = None,
                 phase_provider: Callable[[], str] | None = None,
                 sink: ResourceSampleSink | None = None, clock=time.monotonic):
        for name, value in (("interval_s", interval_s), ("max_age_s", max_age_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        self.interval_s = interval_s
        self.max_age_s = max_age_s
        self.system_probe = system_probe if system_probe is not None else SystemProbe()
        self.gpu_probe = gpu_probe if gpu_probe is not None else GPUProbe()
        self.worker_state_provider = worker_state_provider
        self.queue_lengths_provider = queue_lengths_provider
        self.phase_provider = phase_provider
        self.sink = sink
        self._clock = clock
        self._lock = Lock()
        self._lifecycle = Lock()
        self._stop = Event()
        self._sample_ready = Event()
        self._thread: Thread | None = None
        self._latest = self._empty_sample()
        self._sampler_error: str | None = None
        self._sink_error: str | None = None

    @staticmethod
    def _empty_sample() -> ResourceSample:
        return ResourceSample(0, 0.0, 0.0, 0.0, dict.fromkeys(METRICS),
                              dict.fromkeys(METRICS, "not_sampled"))

    @property
    def diagnostics(self) -> dict:
        with self._lock:
            return {"running": self._thread is not None and self._thread.is_alive(),
                    "sampler_error": self._sampler_error, "sink_error": self._sink_error,
                    "sequence": self._latest.sequence}

    def start(self) -> None:
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("previous sampler is still stopping")
                return
            self._stop.clear()
            self._sample_ready.clear()
            with self._lock:
                self._latest = self._empty_sample()
                self._sampler_error = self._sink_error = None
            self._thread = Thread(target=self._run, name="resource-sampler", daemon=True)
            self._thread.start()

    def close(self, timeout_s: float = 5.0) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        with self._lifecycle:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout_s)
                if self._thread.is_alive():
                    raise TimeoutError("resource sampler did not stop; a probe/source may be blocked")

    def wait_for_sample(self, timeout_s: float = 5.0) -> bool:
        self._sample_ready.wait(timeout_s)
        with self._lock:
            return self._latest.sequence > 0

    def read_sample(self) -> ResourceSample:
        """Hardware comes from cache; callbacks must return quick state copies."""
        with self._lock:
            sample = self._latest
            sampler_error, sink_error = self._sampler_error, self._sink_error
        errors = dict(sample.errors)
        if sampler_error:
            errors["sampler"] = sampler_error
        if sink_error:
            errors["resource_log"] = sink_error
        workers = {}
        queues = {}
        phase = None
        try:
            if self.worker_state_provider is None:
                errors["workers"] = "not_configured"
            else:
                workers = dict(self.worker_state_provider())
                for device, state in workers.items():
                    if not isinstance(device, Device) or not isinstance(state, WorkerState) or state.device is not device:
                        raise ValueError("invalid worker state mapping")
                workers = {
                    device: replace(state, memory_free_mb=sample.values.get(
                        "ram_free_mb" if device is Device.CPU else "gpu_memory_free_mb"))
                    for device, state in workers.items()
                }
        except Exception as exc:
            workers = {}
            errors["workers"] = f"source_error:{type(exc).__name__}:{exc}"
        try:
            if self.queue_lengths_provider is None:
                errors["queue_lengths"] = "not_configured"
            else:
                queues = dict(self.queue_lengths_provider())
                if any(not isinstance(key, str) or type(value) is not int or value < 0
                       for key, value in queues.items()):
                    raise ValueError("queue lengths must map names to nonnegative integers")
        except Exception as exc:
            queues = {}
            errors["queue_lengths"] = f"source_error:{type(exc).__name__}:{exc}"
        try:
            if self.phase_provider is not None:
                phase = self.phase_provider()
                if not isinstance(phase, str):
                    raise TypeError("phase must be str")
        except Exception as exc:
            phase = None
            errors["phase"] = f"source_error:{type(exc).__name__}:{exc}"
        now = self._clock()
        age = max(0.0, now - sample.timestamp_s) if sample.sequence else None
        return replace(sample, errors=errors, workers=workers, queue_lengths=queues,
                       state_timestamp_s=now, age_s=age,
                       stale=age is None or age > self.max_age_s, phase=phase)

    def snapshot(self) -> ResourceSnapshot:
        return self.read_sample().snapshot()

    @staticmethod
    def _read_probe(probe, keys, startup_error) -> ProbeReading:
        try:
            if startup_error is not None:
                raise RuntimeError(startup_error)
            reading = probe.read()
            values = {key: reading.values.get(key) for key in keys}
            errors = dict(reading.errors)
            for key, value in values.items():
                if value is None:
                    errors.setdefault(key, "unavailable")
                elif not math.isfinite(value) or value < 0:
                    values[key] = None
                    errors[key] = "invalid_value"
            return ProbeReading(values, errors)
        except Exception as exc:
            return ProbeReading(dict.fromkeys(keys), dict.fromkeys(keys, f"probe_error:{type(exc).__name__}:{exc}"))

    def _run(self) -> None:
        probes = ((self.system_probe, SYSTEM_METRICS), (self.gpu_probe, GPU_METRICS))
        startup_errors = []
        try:
            for probe, _ in probes:
                try:
                    probe.start()
                    startup_errors.append(None)
                except Exception as exc:
                    startup_errors.append(str(exc))
            # Prime CPU counters in this thread, then allow a real interval.
            deadline = self._clock() + self.interval_s
            sequence = 0
            while not self._stop.wait(max(0.0, deadline - self._clock())):
                started = self._clock()
                values, errors = {}, {}
                for (probe, keys), startup_error in zip(probes, startup_errors):
                    reading = self._read_probe(probe, keys, startup_error)
                    values.update(reading.values)
                    errors.update(reading.errors)
                finished = self._clock()
                sequence += 1
                # Conservative timestamp: an early metric must not become fresh
                # again merely because a later native query took a long time.
                sample = ResourceSample(sequence, started, started, max(0.0, finished - started), values, errors)
                with self._lock:
                    self._latest = sample
                self._sample_ready.set()
                if self.sink is not None and self._sink_error is None:
                    try:
                        self.sink.record_resource(self.read_sample())
                    except Exception as exc:
                        with self._lock:
                            self._sink_error = f"{type(exc).__name__}:{exc}"
                deadline += self.interval_s
                if deadline <= self._clock():
                    deadline = self._clock() + self.interval_s  # No catch-up spin after a slow query.
        except Exception as exc:
            with self._lock:
                self._sampler_error = f"{type(exc).__name__}:{exc}"
        finally:
            for probe, _ in reversed(probes):
                try:
                    probe.close()
                except Exception as exc:
                    with self._lock:
                        self._sampler_error = (self._sampler_error or "") + f" cleanup:{exc}"
            self._sample_ready.set()
