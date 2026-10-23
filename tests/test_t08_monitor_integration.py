import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.monitoring.gpu_probe import GPUProbe
from edge_scheduler.monitoring.resource_monitor import ResourceMonitor


class MonitorIntegrationTest(unittest.TestCase):
    def test_real_cpu_ram_and_process_with_gpu_disabled(self):
        monitor = ResourceMonitor(interval_s=0.1, gpu_probe=GPUProbe(enabled=False))
        try:
            monitor.start()
            self.assertTrue(monitor.wait_for_sample(5))
            sample = monitor.read_sample()
            self.assertGreater(sample.values["ram_free_mb"], 0)
            self.assertGreater(sample.values["process_rss_mb"], 0)
            self.assertTrue(0 <= sample.values["cpu_load_pct"] <= 100)
            self.assertEqual(sample.errors["gpu_memory_free_mb"], "disabled")
        finally:
            monitor.close()
        self.assertFalse(monitor.diagnostics["running"])
        self.assertIsNone(monitor.diagnostics["sampler_error"])

    @unittest.skipUnless(os.environ.get("T08_TEST_GPU", os.environ.get("T06_TEST_GPU")) == "1",
                         "set T08_TEST_GPU=1 (or T06_TEST_GPU=1) to require real GPU memory telemetry")
    def test_real_gpu_memory_utilization_optional_on_wsl(self):
        monitor = ResourceMonitor(interval_s=0.1)
        try:
            monitor.start()
            self.assertTrue(monitor.wait_for_sample(5))
            sample = monitor.read_sample()
            self.assertIsNotNone(sample.values["gpu_memory_total_mb"], sample.errors)
            self.assertGreater(sample.values["gpu_memory_total_mb"], 0)
            utilization = sample.values["gpu_utilization_pct"]
            if utilization is None:
                self.assertIn("gpu_utilization_pct", sample.errors)
            else:
                self.assertTrue(0 <= utilization <= 100)
        finally:
            monitor.close()


if __name__ == "__main__":
    unittest.main()
