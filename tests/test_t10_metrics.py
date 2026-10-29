import csv
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.core.contracts import EventKind, JobEvent
from edge_scheduler.core.models import Device, Job
from edge_scheduler.runtime.metrics_collector import MetricsCollector
from edge_scheduler.runtime.run_results import summarize_jobs
import test_main_pipeline as fixtures


class CollectorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.collector = MetricsCollector(self.directory)
        self.addCleanup(lambda: self.collector.close() if not self.collector.stats["closed"] else None)

    def emit(self, kind, timestamp, job_id="one", **kwargs):
        self.collector.record(JobEvent(kind, job_id, "mobilenetv2", timestamp, **kwargs))

    def success(self, job_id="one", times=(1, 2, 3, 5)):
        for kind, timestamp in zip((EventKind.ACCEPTED, EventKind.DISPATCHED, EventKind.STARTED, EventKind.COMPLETED), times):
            self.emit(kind, timestamp, job_id, device=None if kind is EventKind.ACCEPTED else Device.CPU,
                      input_ref="input.jpg", top1_class=4 if kind is EventKind.COMPLETED else None,
                      scheduler_overhead_s=None if kind is EventKind.ACCEPTED else 0.001)

    def test_streaming_artifacts_and_final_formulas_match_controller_jobs(self):
        self.success()
        self.emit(EventKind.ACCEPTED, 2, "bad", input_ref="missing.jpg")
        self.emit(EventKind.FAILED, 7, "bad", error="dispatch failed", scheduler_overhead_s=0.002)
        self.collector.flush()
        events = [json.loads(line) for line in (self.directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 6)
        with (self.directory / "jobs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["top1_class"], "4")
        self.assertEqual(rows[1]["execution_s"], "")
        job = Job("one", "mobilenetv2", "input.jpg", arrival_time=1)
        job.mark_scheduled(Device.CPU, 2)
        job.mark_running(3)
        output = np.zeros((1, 1000), np.float32)
        output[0, 4] = 1
        job.mark_completed(output, 5)
        job.scheduler_overhead_s = 0.001
        bad = Job("bad", "mobilenetv2", "missing.jpg", arrival_time=2)
        bad.mark_failed("dispatch failed", 7)
        bad.scheduler_overhead_s = 0.002
        self.assertEqual(self.collector.summary(), summarize_jobs([job, bad], 2, 2))
        self.assertAlmostEqual(self.collector.summary()["throughput_jobs_s"], 1 / 6)

    def test_recent_uses_completion_time_and_startup_duration(self):
        self.success(times=(1, 1.1, 1.2, 2))
        snapshot = self.collector.snapshot(5, now=2)
        self.assertEqual(snapshot["recent_samples"], 1)
        self.assertEqual(snapshot["recent_throughput"], 1)
        self.assertEqual(snapshot["recent_latency_ms"]["p95"], 1000)
        self.assertEqual(self.collector.snapshot(1, now=3)["recent_samples"], 0)
        self.assertIsNone(self.collector.snapshot(1, now=4)["recent_latency_ms"]["p95"])

    def test_empty_window_and_invalid_lifecycle_events(self):
        self.assertIsNone(self.collector.snapshot(now=1)["recent_throughput"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.emit(EventKind.STARTED, 1, device=Device.CPU)
        self.emit(EventKind.ACCEPTED, 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.emit(EventKind.ACCEPTED, 1)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.emit(EventKind.DISPATCHED, 0.5, device=Device.CPU)
        self.emit(EventKind.DISPATCHED, 2, device=Device.CPU)
        with self.assertRaisesRegex(ValueError, "device mismatch"):
            self.emit(EventKind.STARTED, 3, device=Device.GPU)
        self.emit(EventKind.STARTED, 3, device=Device.CPU)
        self.emit(EventKind.COMPLETED, 4, device=Device.CPU)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.emit(EventKind.COMPLETED, 5, device=Device.CPU)

    def test_writer_failure_is_visible_and_close_releases_thread(self):
        with patch.object(self.collector._events_stream, "write", side_effect=OSError("disk full")):
            self.emit(EventKind.ACCEPTED, 1)
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                self.collector.flush()
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            self.collector.close()
        self.assertFalse(self.collector.stats["running"])

    def test_queue_full_fails_without_blocking_controller(self):
        gate = Event()
        original = MetricsCollector._run
        def blocked(collector):
            gate.wait(5)
            original(collector)
        other = self.directory / "bounded"
        other.mkdir()
        with patch.object(MetricsCollector, "_run", blocked):
            collector = MetricsCollector(other, capacity=1)
        try:
            collector.record(JobEvent(EventKind.ACCEPTED, "id", "mobilenetv2", 1))
            with self.assertRaisesRegex(RuntimeError, "queue full"):
                collector.record(JobEvent(EventKind.DISPATCHED, "id", "mobilenetv2", 2, Device.CPU))
            self.assertEqual(collector.stats["dropped_events"], 1)
        finally:
            gate.set()
            with self.assertRaisesRegex(RuntimeError, "queue full"):
                collector.close()
        self.assertFalse(collector.stats["running"])


class LivePipelineTest(unittest.TestCase):
    setUp = fixtures.MainPipelineTest.setUp
    run_main = fixtures.MainPipelineTest.run_main

    def test_live_plain_output_and_collector_reconciliation(self):
        code, directory, report = self.run_main("--live", "--window", "1", "--update", "0.01")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["jobs"], report["collected_jobs"])
        self.assertEqual(report["collector"]["written_events"], 24)
        self.assertFalse(report["collector"]["running"])
        self.assertEqual(report["jobs"]["scheduler_overhead_ms"]["count"], 6)


if __name__ == "__main__":
    unittest.main()
