import csv
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler import ResourceSampleSink
from edge_scheduler.monitoring_models import METRICS, ResourceSample, WorkerRegistry
from edge_scheduler.resource_log import ResourceCSVLogger


class LogTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "resources.csv"
        self.sample = ResourceSample(1, 10, 9.99, 0.01, dict.fromkeys(METRICS),
                                     {"gpu_utilization_pct": "not_supported"},
                                     workers=WorkerRegistry().snapshot())

    def test_schema_nulls_flush_close_and_no_overwrite(self):
        logger = ResourceCSVLogger(self.path)
        self.assertIsInstance(logger, ResourceSampleSink)
        logger.record_resource(self.sample)
        logger.close()
        logger.close()
        with self.path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gpu_utilization_pct"], "")
        self.assertEqual(json.loads(rows[0]["errors_json"])["gpu_utilization_pct"], "not_supported")
        self.assertFalse(json.loads(rows[0]["workers_json"])["gpu"]["available"])
        self.assertEqual(logger.stats["written_samples"], 1)
        with self.assertRaises(RuntimeError):
            logger.record_resource(self.sample)
        with self.assertRaises(FileExistsError):
            ResourceCSVLogger(self.path)

    def test_bounded_queue_reports_drops_and_drains_on_close(self):
        entered, release = Event(), Event()
        original = ResourceCSVLogger._row
        def slow(sample):
            entered.set()
            release.wait(3)
            return original(sample)
        with patch.object(ResourceCSVLogger, "_row", staticmethod(slow)):
            logger = ResourceCSVLogger(self.path, capacity=1)
            try:
                logger.record_resource(self.sample)
                self.assertTrue(entered.wait(2))
                logger.record_resource(self.sample)
                logger.record_resource(self.sample)
                self.assertEqual(logger.stats["dropped_samples"], 1)
            finally:
                release.set()
                logger.close()
        self.assertEqual(logger.stats["written_samples"], 2)

    def test_writer_failure_surfaces(self):
        with patch.object(ResourceCSVLogger, "_row", side_effect=OSError("disk full")):
            logger = ResourceCSVLogger(self.path)
            logger.record_resource(self.sample)
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                logger.close()
            self.assertIsNotNone(logger.stats["error"])


if __name__ == "__main__":
    unittest.main()
