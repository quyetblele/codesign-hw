from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.monitoring.gpu_probe import GPUProbe
from edge_scheduler.monitoring.monitoring_models import MIB
from edge_scheduler.monitoring.system_probe import SystemProbe


class SystemProbeTest(unittest.TestCase):
    def setUp(self):
        self.api = MagicMock()
        self.api.cpu_count.return_value = 8
        self.api.cpu_percent.return_value = 25.0
        self.api.virtual_memory.return_value = SimpleNamespace(
            total=8000 * MIB, available=3000 * MIB, free=100 * MIB, percent=62.5)
        self.process = self.api.Process.return_value
        self.process.memory_info.return_value = SimpleNamespace(rss=120 * MIB)
        self.process.cpu_percent.return_value = 240.0
        self.probe = SystemProbe(self.api)

    def test_units_available_ram_and_multicore_process_cpu(self):
        self.probe.start()
        sample = self.probe.read()
        self.assertEqual(sample.values["ram_free_mb"], 3000)
        self.assertEqual(sample.values["process_rss_mb"], 120)
        self.assertEqual(sample.values["process_cpu_pct"], 240)
        self.assertEqual(sample.errors, {})
        self.assertTrue(all(call.kwargs == {"interval": None}
                            for call in self.api.cpu_percent.call_args_list))

    def test_failed_prime_does_not_publish_first_zero(self):
        self.api.cpu_percent.side_effect = [OSError("initial failure"), 0.0, 20.0]
        self.probe.start()
        first = self.probe.read()
        self.assertIsNone(first.values["cpu_load_pct"])
        self.assertEqual(first.errors["cpu_load_pct"], "warming_up")
        self.assertEqual(self.probe.read().values["cpu_load_pct"], 20)

    def test_memory_failure_preserves_cpu(self):
        self.probe.start()
        self.api.virtual_memory.side_effect = PermissionError()
        sample = self.probe.read()
        self.assertEqual(sample.values["cpu_load_pct"], 25)
        self.assertIsNone(sample.values["ram_free_mb"])
        self.assertEqual(sample.errors["ram_free_mb"], "permission_denied")

    def test_process_permission_and_exit_are_field_errors(self):
        self.probe.start()
        self.process.memory_info.side_effect = psutil.AccessDenied(123)
        self.process.cpu_percent.side_effect = psutil.NoSuchProcess(123)
        sample = self.probe.read()
        self.assertEqual(sample.errors["process_rss_mb"], "permission_denied")
        self.assertEqual(sample.errors["process_cpu_pct"], "process_unavailable")
        self.assertEqual(sample.values["ram_free_mb"], 3000)

    def test_invalid_values_are_not_fabricated_zero(self):
        self.probe.start()
        self.api.cpu_percent.return_value = float("nan")
        self.process.cpu_percent.return_value = -1
        sample = self.probe.read()
        self.assertIsNone(sample.values["cpu_load_pct"])
        self.assertEqual(sample.errors["process_cpu_pct"], "invalid_value")


class GPUProbeTest(unittest.TestCase):
    def setUp(self):
        self.api = MagicMock()
        self.api.nvmlDeviceGetName.return_value = b"Test GPU"
        self.api.nvmlDeviceGetUUID.return_value = "GPU-test"
        self.api.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(
            total=4096 * MIB, used=1024 * MIB, free=3072 * MIB)
        self.api.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=50)
        self.probe = GPUProbe(api=self.api)

    def test_read_units_metadata_and_cleanup(self):
        self.probe.start()
        self.probe.start()
        sample = self.probe.read()
        self.assertEqual(sample.values["gpu_memory_free_mb"], 3072)
        self.assertEqual(sample.values["gpu_utilization_pct"], 50)
        self.assertEqual(self.probe.metadata["name"], "Test GPU")
        self.probe.close()
        self.probe.close()
        self.api.nvmlInit.assert_called_once()
        self.api.nvmlShutdown.assert_called_once()

    def test_unsupported_utilization_keeps_memory(self):
        self.api.nvmlDeviceGetUtilizationRates.side_effect = type("NVMLError_NotSupported", (Exception,), {})()
        self.probe.start()
        sample = self.probe.read()
        self.assertEqual(sample.values["gpu_memory_free_mb"], 3072)
        self.assertIsNone(sample.values["gpu_utilization_pct"])
        self.assertEqual(sample.errors["gpu_utilization_pct"], "not_supported")
        self.probe.close()

    def test_memory_failure_keeps_utilization(self):
        self.api.nvmlDeviceGetMemoryInfo.side_effect = type("NVMLError_NoPermission", (Exception,), {})()
        self.probe.start()
        sample = self.probe.read()
        self.assertEqual(sample.errors["gpu_memory_free_mb"], "permission_denied")
        self.assertEqual(sample.values["gpu_utilization_pct"], 50)
        self.probe.close()

    def test_driver_or_library_missing_does_not_raise(self):
        for name, reason in (("NVMLError_DriverNotLoaded", "driver_unavailable"),
                             ("NVMLError_LibraryNotFound", "library_unavailable")):
            self.api.nvmlInit.side_effect = type(name, (Exception,), {})()
            self.probe.start()
            sample = self.probe.read()
            self.assertTrue(all(value is None for value in sample.values.values()))
            self.assertEqual(set(sample.errors.values()), {reason})
            self.probe.close()
        self.api.nvmlShutdown.assert_not_called()

    def test_uuid_selection_and_handle_failure_cleanup(self):
        probe = GPUProbe(api=self.api, device_uuid="GPU-selected")
        self.api.nvmlDeviceGetHandleByUUID.side_effect = type("NVMLError_NotFound", (Exception,), {})()
        probe.start()
        self.assertEqual(probe.read().errors["gpu_memory_free_mb"], "device_not_found")
        probe.close()
        self.api.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-selected")
        self.api.nvmlShutdown.assert_called_once()

    def test_disabled_does_not_initialize(self):
        probe = GPUProbe(api=self.api, enabled=False)
        probe.start()
        self.assertEqual(set(probe.read().errors.values()), {"disabled"})
        probe.close()
        self.api.nvmlInit.assert_not_called()

    def test_invalid_memory_sentinel_is_unavailable(self):
        self.probe.start()
        self.api.nvmlDeviceGetMemoryInfo.return_value.free = 2 ** 64 - 1
        sample = self.probe.read()
        self.assertIsNone(sample.values["gpu_memory_free_mb"])
        self.assertIn("gpu_memory_free_mb", sample.errors)
        self.probe.close()


if __name__ == "__main__":
    unittest.main()
