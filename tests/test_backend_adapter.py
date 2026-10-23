import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.core.models import Device
from edge_scheduler.monitoring.monitoring_models import WorkerRegistry
from edge_scheduler.runtime.backend_adapter import MonitoredBackend


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.registry = WorkerRegistry()
        registry = self.registry

        class Backend:
            device = Device.CPU
            error = None
            closed = False

            def prepare(self):
                pass

            def infer(self, model_id, input_ref):
                state = registry.snapshot()[self.device]
                self.running_state = state
                if self.error:
                    raise self.error
                return "output"

            def close(self):
                self.closed = True

        self.backend = Backend()
        self.adapter = MonitoredBackend(self.backend, self.registry)

    def test_lifecycle_publishes_facts_without_inventing_remaining_time(self):
        self.assertFalse(self.registry.snapshot()[Device.CPU].available)
        self.adapter.prepare()
        self.assertTrue(self.registry.snapshot()[Device.CPU].available)
        self.assertEqual(self.adapter.infer("mobilenetv2", "image"), "output")
        running = self.backend.running_state
        self.assertEqual(running.running_model_id, "mobilenetv2")
        self.assertIsNotNone(running.running_started_s)
        self.assertFalse(running.estimated_remaining_known)
        self.assertIsNone(self.registry.snapshot()[Device.CPU].running_model_id)
        self.adapter.close()
        self.assertTrue(self.backend.closed)
        self.assertFalse(self.registry.snapshot()[Device.CPU].available)

    def test_bad_input_keeps_worker_healthy_and_next_request_succeeds(self):
        self.adapter.prepare()
        self.backend.error = FileNotFoundError("missing image")
        with self.assertRaises(FileNotFoundError):
            self.adapter.infer("mobilenetv2", "missing")
        self.assertTrue(self.registry.snapshot()[Device.CPU].available)
        self.backend.error = None
        self.assertEqual(self.adapter.infer("resnet50", "image"), "output")

    def test_runtime_failure_marks_device_unavailable(self):
        self.adapter.prepare()
        self.backend.error = RuntimeError("device lost")
        with self.assertRaisesRegex(RuntimeError, "device lost"):
            self.adapter.infer("mobilenetv2", "image")
        self.assertFalse(self.registry.snapshot()[Device.CPU].available)
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            self.adapter.infer("mobilenetv2", "image")
        self.adapter.close()


if __name__ == "__main__":
    unittest.main()
