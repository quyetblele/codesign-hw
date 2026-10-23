from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event, get_ident
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler import Device, ResourceMonitor as MonitorProtocol, WorkerState
from edge_scheduler.monitoring.monitoring_models import GPU_METRICS, SYSTEM_METRICS, ProbeReading, WorkerRegistry
from edge_scheduler.monitoring.resource_monitor import ResourceMonitor


class FakeProbe:
    def __init__(self, keys):
        self.keys = keys
        self.started = self.reads = self.closed = 0
        self.start_thread = self.read_thread = None
        self.failure = False
        self.metadata = {}

    def start(self):
        self.started += 1
        self.start_thread = get_ident()

    def read(self):
        self.reads += 1
        self.read_thread = get_ident()
        if self.failure:
            raise RuntimeError("sensor failure")
        return ProbeReading(dict.fromkeys(self.keys, 10.0))

    def close(self):
        self.closed += 1


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.system = FakeProbe(SYSTEM_METRICS)
        self.gpu = FakeProbe(GPU_METRICS)
        self.registry = WorkerRegistry()

    def monitor(self, **kwargs):
        monitor = ResourceMonitor(interval_s=0.01, system_probe=self.system,
                                  gpu_probe=self.gpu, worker_state_provider=self.registry.snapshot, **kwargs)
        self.addCleanup(monitor.close)
        return monitor

    def start(self, monitor):
        monitor.start()
        self.assertTrue(monitor.wait_for_sample(2), monitor.diagnostics)

    def test_protocol_initial_sample_and_legacy_worker_constructor(self):
        monitor = self.monitor()
        self.assertIsInstance(monitor, MonitorProtocol)
        sample = monitor.read_sample()
        self.assertEqual(sample.sequence, 0)
        self.assertTrue(sample.stale)
        self.assertIsNone(sample.snapshot().cpu_load_pct)
        state = WorkerState(Device.CPU, True, None, 0.0, None)
        self.assertFalse(state.estimated_remaining_known)
        self.assertIsNone(state.running_started_s)

    def test_idempotent_lifecycle_restart_and_same_sampling_thread(self):
        monitor = self.monitor()
        self.start(monitor)
        monitor.start()
        self.assertEqual(self.system.started, 1)
        self.assertEqual(self.system.start_thread, self.system.read_thread)
        self.assertNotEqual(self.system.read_thread, get_ident())
        monitor.close()
        monitor.close()
        self.assertEqual(self.system.closed, 1)
        self.start(monitor)
        monitor.close()
        self.assertEqual(self.system.closed, 2)

    def test_timestamp_age_and_stale_do_not_relabel_old_data(self):
        clock = MagicMock(return_value=100.0)
        monitor = self.monitor(clock=clock)
        self.start(monitor)
        monitor.close()
        original = monitor.snapshot().timestamp
        clock.return_value = 102.0
        sample = monitor.read_sample()
        self.assertEqual(sample.timestamp_s, original)
        self.assertEqual(sample.age_s, 2)
        self.assertTrue(sample.stale)

    def test_live_worker_state_and_immutable_copies(self):
        monitor = self.monitor(queue_lengths_provider=lambda: {"cpu": 2})
        self.start(monitor)
        self.registry.set_ready(Device.CPU)
        self.registry.set_running(Device.CPU, "resnet50", 1.0)
        sample = monitor.read_sample()
        state = sample.workers[Device.CPU]
        self.assertTrue(state.available)
        self.assertFalse(state.estimated_remaining_known)
        self.assertEqual(state.memory_free_mb, 10)
        self.assertEqual(sample.queue_lengths["cpu"], 2)
        self.registry.set_unavailable(Device.CPU)
        self.assertTrue(sample.workers[Device.CPU].available)
        self.assertFalse(monitor.snapshot().workers[Device.CPU].available)
        for mapping in (sample.values, sample.workers, sample.errors, sample.queue_lengths):
            with self.assertRaises(TypeError):
                mapping["bad"] = 1

    def test_probe_failure_preserves_other_probe_and_reports_null(self):
        self.system.failure = True
        monitor = self.monitor()
        self.start(monitor)
        sample = monitor.read_sample()
        self.assertIsNone(sample.values["cpu_load_pct"])
        self.assertEqual(sample.values["gpu_memory_free_mb"], 10)
        self.assertIn("sensor failure", sample.errors["cpu_load_pct"])

    def test_slow_probe_does_not_refresh_older_metrics_timestamp(self):
        clock = MagicMock(return_value=100.0)
        original = self.gpu.read
        def advance_clock():
            clock.return_value = 102.0
            return original()
        self.gpu.read = advance_clock
        monitor = self.monitor(clock=clock)
        self.start(monitor)
        monitor.close()
        sample = monitor.read_sample()
        self.assertEqual(sample.timestamp_s, 100)
        self.assertEqual(sample.sample_duration_s, 2)
        self.assertTrue(sample.stale)

    def test_unexpected_sampler_failure_is_visible_and_probes_close(self):
        clock = MagicMock(side_effect=RuntimeError("clock failed"))
        monitor = self.monitor(clock=clock)
        monitor.start()
        self.assertFalse(monitor.wait_for_sample(2))
        monitor.close()
        self.assertIn("clock failed", monitor.diagnostics["sampler_error"])
        self.assertEqual(self.system.closed, 1)
        self.assertEqual(self.gpu.closed, 1)

    def test_failed_sources_do_not_invent_worker_or_queue_state(self):
        monitor = self.monitor(queue_lengths_provider=lambda: {"cpu": -1})
        monitor.worker_state_provider = MagicMock(side_effect=RuntimeError("registry unavailable"))
        self.start(monitor)
        sample = monitor.read_sample()
        self.assertEqual(sample.workers, {})
        self.assertEqual(sample.queue_lengths, {})
        self.assertIn("workers", sample.errors)
        self.assertIn("queue_lengths", sample.errors)
        self.assertEqual(sample.values["cpu_load_pct"], 10)

    def test_snapshot_readers_do_not_wait_for_a_blocked_probe(self):
        entered, release = Event(), Event()
        original = self.system.read
        def blocked():
            entered.set()
            release.wait(3)
            return original()
        self.system.read = blocked
        monitor = self.monitor()
        # Release runs before monitor.close in cleanup, even if an assertion fails.
        self.addCleanup(release.set)
        monitor.start()
        self.assertTrue(entered.wait(2))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(monitor.snapshot) for _ in range(20)]
            for future in futures:
                self.assertIsNone(future.result(timeout=1).cpu_load_pct)
        release.set()
        self.assertTrue(monitor.wait_for_sample(2))

    def test_sink_failure_is_visible_without_stopping_sampling(self):
        sink = MagicMock()
        failed = Event()
        def reject(sample):
            failed.set()
            raise OSError("disk full")
        sink.record_resource.side_effect = reject
        monitor = self.monitor(sink=sink)
        self.start(monitor)
        self.assertTrue(failed.wait(2))
        monitor.close()
        self.assertIn("disk full", monitor.read_sample().errors["resource_log"])
        self.assertIsNone(monitor.diagnostics["sampler_error"])

    def test_registry_requires_ready_and_idle_estimate_is_known(self):
        with self.assertRaises(RuntimeError):
            self.registry.set_running(Device.CPU, "resnet50", 1)
        self.registry.set_ready(Device.CPU)
        self.assertTrue(self.registry.snapshot()[Device.CPU].estimated_remaining_known)

    def test_bad_configuration(self):
        for options in ({"interval_s": 0}, {"max_age_s": float("nan")}):
            with self.assertRaises(ValueError):
                ResourceMonitor(**options)


if __name__ == "__main__":
    unittest.main()
