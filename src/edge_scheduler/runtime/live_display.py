"""Read cached metrics on a separate thread; never probe hardware or write job logs."""
import math
import shutil
import sys
import textwrap
from threading import Event, Thread

from ..monitoring.monitoring_models import MEMORY_UNIT


def number(value, suffix="") -> str:
    return "—" if value is None else f"{value:.2f}{suffix}"


class LiveDisplay:
    def __init__(self, collector, monitor, logger, phase, policy, *, window_s=5.0,
                 update_s=1.0, stream=None):
        if any(not math.isfinite(value) or value <= 0 for value in (window_s, update_s)):
            raise ValueError("window and update must be finite and > 0")
        self.collector, self.monitor, self.logger = collector, monitor, logger
        self.phase, self.policy = phase, policy
        self.window_s, self.update_s = window_s, update_s
        self.stream = sys.stdout if stream is None else stream
        self._tty = self.stream.isatty()
        self._lines = 0
        self._stop = Event()
        self._thread = None
        self.error = None

    def render(self):
        stats = self.collector.snapshot(self.window_s)
        sample = self.monitor.read_sample()
        queues = sample.queue_lengths
        workers = " | ".join(
            f"{device.value.upper()}: " + ("unavailable" if not state.available else
            f"running {state.running_model_id}" if state.running_model_id else "idle")
            for device, state in sample.workers.items()) or "workers: —"
        latency = stats["recent_latency_ms"]
        values = sample.values
        text = [
            f"T10 | {self.policy} | {self.phase.get()} | elapsed {stats['elapsed_s']:.1f}s",
            f"Jobs accepted={stats['accepted']} completed={stats['completed']} failed={stats['failed']} unfinished={stats['unfinished']}",
            "Queues " + " ".join(f"{name}={queues.get(name, '—')}" for name in ("incoming", "cpu", "gpu")),
            workers,
            f"Recent {self.window_s:g}s (n={stats['recent_samples']}): {number(stats['recent_throughput'], ' jobs/s')} | P50={number(latency['p50'], ' ms')} P95={number(latency['p95'], ' ms')}",
            f"CPU={number(values['cpu_load_pct'], '%')} RSS={number(values['process_rss_mb'], ' ' + MEMORY_UNIT)} RAM available={number(values['ram_free_mb'], ' ' + MEMORY_UNIT)}",
            f"VRAM used/free={number(values['gpu_memory_used_mb'])}/{number(values['gpu_memory_free_mb'])} {MEMORY_UNIT} | GPU={number(values['gpu_utilization_pct'], '%')}",
            f"Log dropped resources={self.logger.stats['dropped_samples']} events={self.collector.stats['dropped_events']} | sample stale={sample.stale} age={number(sample.age_s, 's')} metric_errors={len(sample.errors)} | error={stats['error'] or self.monitor.diagnostics['sink_error'] or self.monitor.diagnostics['sampler_error'] or 'none'}",
        ]
        if self._tty:
            width = max(20, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)
            text = [part for line in text for part in textwrap.wrap(line, width=width)]
            text += [""] * max(0, self._lines - len(text))
            prefix = f"\033[{self._lines}A" if self._lines else ""
            self.stream.write(prefix + "".join("\033[2K" + line + "\n" for line in text))
            self._lines = len(text)
        else:
            self.stream.write(" | ".join(text) + "\n")
        self.stream.flush()

    def _run(self):
        try:
            while not self._stop.is_set():
                self.render()
                self._stop.wait(self.update_s)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def start(self):
        self._thread = Thread(target=self._run, name="live-display", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
            if self._thread.is_alive():
                raise TimeoutError("live display did not stop")
        if self.error:
            raise RuntimeError(self.error)
        self.render()
