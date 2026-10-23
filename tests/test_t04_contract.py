"""Week 1 T04 contract tests; stdlib unittest, no real Worker dependencies."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_scheduler import (
    Job, JobStatus, Device, Scheduler, EventKind,
    JobEvent, WorkerMessage, InferenceBackend,
)
from edge_scheduler.core.contract_validation import (
    validate_job_request, validate_backend, validate_backend_output,
)


class FakeBackend:
    def __init__(self, device):
        self._device = device
        self.ready = False

    @property
    def device(self):
        return self._device

    def prepare(self):
        self.ready = True

    def infer(self, model_id, input_ref):
        if not self.ready:
            raise RuntimeError("not prepared")
        import numpy as np
        return np.zeros((1, 1000), dtype=np.float32)

    def close(self):
        self.ready = False


class FakeScheduler:
    def choose_device(self, job, queued_jobs, snapshot=None):
        return Device.CPU


class ContractTest(unittest.TestCase):
    def test_job_valid(self):
        validate_job_request(Job("j1", "mobilenetv2", "inputs/test.jpg"))

    def test_job_invalid_model(self):
        with self.assertRaises(ValueError):
            validate_job_request(Job("j1", "unexpected", "inputs/test.jpg"))

    def test_job_invalid_input(self):
        with self.assertRaises(ValueError):
            validate_job_request(Job("j1", "resnet50", ""))

    def test_status_lifecycle(self):
        job = Job("j1", "mobilenetv2", "inputs/test.jpg")
        self.assertEqual(job.status, JobStatus.CREATED)
        job.mark_waiting()
        self.assertEqual(job.status, JobStatus.WAITING)
        job.mark_scheduled(Device.CPU, now=1.0)
        job.mark_running(now=2.0)
        job.mark_completed("result", now=3.0)
        self.assertTrue(job.terminal)

    def test_scheduler_interface(self):
        scheduler = FakeScheduler()
        self.assertEqual(scheduler.choose_device(Job("j1", "resnet50", "image.jpg"), {}), Device.CPU)

    def test_backend_protocol(self):
        for device in Device:
            with self.subTest(device=device):
                backend = FakeBackend(device)
                self.assertIsInstance(backend, InferenceBackend)
                validate_backend(backend, device)
                backend.prepare()
                validate_backend_output(backend.infer("mobilenetv2", "inputs/test.jpg"))
                backend.close()

    def test_backend_wrong_device(self):
        with self.assertRaises(ValueError):
            validate_backend(FakeBackend(Device.CPU), Device.GPU)

    def test_worker_events(self):
        msg = WorkerMessage(EventKind.STARTED, "j1", Device.CPU, 1.0)
        self.assertEqual(msg.job_id, "j1")
        ev = JobEvent(EventKind.COMPLETED, "j1", "mobilenetv2", 2.0, Device.CPU)
        self.assertEqual(ev.kind, EventKind.COMPLETED)

    def test_failure_error_mandatory(self):
        with self.assertRaises(ValueError):
            WorkerMessage(EventKind.FAILED, "j1", Device.CPU, 1.0)
        with self.assertRaises(ValueError):
            JobEvent(EventKind.FAILED, "j1", "resnet50", 1.0)


if __name__ == "__main__":
    unittest.main()
