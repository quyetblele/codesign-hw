"""Week 2 acceptance/stress checks using deterministic FakeBackend, not real ONNX."""
import threading
import time
import unittest

from edge_scheduler.core.contracts import EventKind, WorkerMessage
from edge_scheduler.runtime.controller import Controller
from edge_scheduler.core.interfaces import ResourceSnapshot, WorkerState
from edge_scheduler.core.models import Device, Job, JobStatus
from edge_scheduler.runtime.queues import JobQueues
from edge_scheduler.runtime.schedulers import (
    RoundRobinScheduler, CpuOnlyScheduler,
    ResourceAwareScheduler, ServiceTimeProfile,
)


class FakeBackend:
    def __init__(self, device, delay=0.0, prepare_error=False, close_error=False):
        self.device = device
        self.delay = delay
        self.prepare_error = prepare_error
        self.close_error = close_error
        self.closed = False
    def prepare(self):
        if self.prepare_error:
            raise RuntimeError("prepare error")
    def infer(self, model_id, input_ref):
        if self.delay:
            time.sleep(self.delay)
        if input_ref == "error":
            raise ValueError("inference error")
        return [model_id, input_ref, self.device.value]
    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError("backend close error")


class FailingCollector:
    def __init__(self):
        self.flushed = False
        self.closed = False
    def record(self, event):
        raise IOError("collector record error")
    def flush(self):
        self.flushed = True
    def close(self):
        self.closed = True


def controller(delay=0.0, scheduler=None, monitor=None, collector=None,
               prepare_error=False, close_error=False):
    cpu = FakeBackend(Device.CPU, delay, prepare_error=prepare_error, close_error=close_error)
    gpu = FakeBackend(Device.GPU, delay)
    result = Controller(scheduler or RoundRobinScheduler(),
                        {Device.CPU: cpu, Device.GPU: gpu},
                        monitor=monitor, collector=collector)
    return result, cpu, gpu


class Week2HardeningTests(unittest.TestCase):
    def test_graceful_shutdown_with_pending_jobs(self):
        c, cpu, gpu = controller(delay=.002)
        c.start()
        jobs = [Job(f"q-{i}", "mobilenetv2", "image") for i in range(50)]
        for j in jobs:
            c.submit(j)
        c.request_shutdown()
        self.assertFalse(c.accepting_jobs)
        with self.assertRaises(RuntimeError):
            c.submit(Job("late", "mobilenetv2", "image"))
        c.close(timeout=10)
        self.assertEqual(c.queues.reconcile(), (50, 50))
        self.assertTrue(all(j.status is JobStatus.COMPLETED for j in jobs))
        self.assertTrue(cpu.closed and gpu.closed)
        self.assertTrue(all(not t.is_alive() for t in c._workers))

    def test_graceful_shutdown_midflight(self):
        c, _, _ = controller(delay=.02)
        c.start()
        jobs = [Job(f"m-{i}", "resnet50", "image") for i in range(12)]
        for j in jobs:
            c.submit(j)
        c.dispatch_pending()
        time.sleep(.003)
        c.request_shutdown()
        c.close(timeout=5)
        self.assertEqual(c.queues.reconcile(), (12, 12))
        self.assertTrue(all(j.status is JobStatus.COMPLETED for j in jobs))

    def test_close_timeout_is_retryable(self):
        c, _, _ = controller(delay=.06)
        c.start()
        jobs = [Job(str(i), "mobilenetv2", "image") for i in range(12)]
        for j in jobs: c.submit(j)
        with self.assertRaises(TimeoutError):
            c.close(timeout=.001)
        self.assertFalse(c.accepting_jobs)
        self.assertTrue(any(t.is_alive() for t in c._workers))
        c.close(timeout=5)
        self.assertEqual(c.queues.reconcile(), (12, 12))
        self.assertTrue(all(not t.is_alive() for t in c._workers))

    def test_duplicate_terminal_message_is_rejected(self):
        c, _, _ = controller()
        c.start()
        job = Job("single", "mobilenetv2", "image")
        c.submit(job)
        self.assertTrue(c.wait_all())
        before = c.queues.reconcile()
        with self.assertRaises(ValueError):
            c.handle_worker_message(WorkerMessage(
                EventKind.COMPLETED, "single", job.assigned_device,
                time.monotonic(), output=123))
        self.assertEqual(c.queues.reconcile(), before)
        c.close()

    def test_event_collector_failure_never_loses_job(self):
        coll = FailingCollector()
        c, cpu, gpu = controller(collector=coll)
        c.start()
        jobs = [Job(str(i), "mobilenetv2", "image") for i in range(8)]
        for j in jobs:
            c.submit(j)
        self.assertTrue(c.wait_all())
        with self.assertRaisesRegex(RuntimeError, "event collection errors"):
            c.close()
        self.assertEqual(c.queues.reconcile(), (8, 8))
        self.assertTrue(coll.flushed and coll.closed and cpu.closed and gpu.closed)
        self.assertEqual(len(c.collector_errors), 8 * 4)
        self.assertTrue(all(j.status is JobStatus.COMPLETED for j in jobs))

    def test_queue_snapshot_is_safe_and_excludes_running(self):
        qs = JobQueues()
        j1, j2 = Job("a", "mobilenetv2", "x"), Job("b", "resnet50", "x")
        qs.accept(j1); qs.accept(j2)
        qs.dispatch(j1, Device.CPU); qs.dispatch(j2, Device.GPU)
        snap = qs.queued_snapshot()
        self.assertEqual(snap[Device.CPU], (j1,))
        self.assertEqual(snap[Device.GPU], (j2,))
        picked = qs.device[Device.CPU].get_nowait()
        self.assertIs(picked, j1)
        qs.device[Device.CPU].task_done()
        self.assertEqual(qs.queued_snapshot()[Device.CPU], ())
        with self.assertRaises(ValueError):
            qs.dispatch(j1, Device.CPU)

    def test_queue_unauthorized_terminal_rejected(self):
        qs = JobQueues()
        j = Job("unknown", "mobilenetv2", "image")
        j.mark_failed("some error")
        with self.assertRaises(ValueError): qs.record_terminal(j)

    def test_shutdown_race_multi_producers(self):
        c, _, _ = controller(delay=.0003)
        c.start()
        accepted = []
        rejected = []
        lock = threading.Lock()
        def producer(index):
            for i in range(80):
                job = Job(f"{index}-{i}", "mobilenetv2", "image")
                try:
                    c.submit(job)
                except RuntimeError:
                    with lock: rejected.append(job)
                else:
                    with lock: accepted.append(job)
                if i % 9 == 0:
                    time.sleep(.0001)
        producers = [threading.Thread(target=producer, args=(i,)) for i in range(5)]
        for t in producers: t.start()
        time.sleep(.002)
        c.request_shutdown()
        for t in producers: t.join()
        c.close(timeout=10)
        self.assertEqual(c.queues.reconcile(), (len(accepted), len(accepted)))
        self.assertEqual(len(accepted) + len(rejected), 400)
        self.assertTrue(all(j.status is JobStatus.COMPLETED for j in accepted))

    def test_failing_prepare_does_not_start_workers(self):
        c, _, _ = controller(prepare_error=True)
        with self.assertRaisesRegex(RuntimeError, "prepare error"):
            c.start()
        self.assertFalse(c._workers)
        self.assertFalse(c._started)

    def test_cleanup_failure_still_closes_all_components(self):
        class Collector:
            def __init__(self): self.flushed=False; self.closed=False
            def record(self, event): pass
            def flush(self): self.flushed=True
            def close(self): self.closed=True
        coll = Collector()
        c, cpu, gpu = controller(close_error=True, collector=coll)
        c.start()
        c.submit(Job("a", "mobilenetv2", "image"))
        with self.assertRaisesRegex(RuntimeError, "cleanup errors"):
            c.close()
        self.assertTrue(cpu.closed and gpu.closed and coll.flushed and coll.closed)

    def test_monitor_unavailable_device_is_not_dispatched(self):
        class Monitor:
            def snapshot(self):
                return ResourceSnapshot(
                    timestamp=time.monotonic(), cpu_load_pct=None,
                    ram_free_mb=None, gpu_memory_free_mb=None,
                    workers={Device.CPU:WorkerState(
                        device=Device.CPU, available=False,
                        estimated_remaining_s=0.0)})
        c, _, _ = controller(scheduler=CpuOnlyScheduler(), monitor=Monitor())
        c.start()
        j = Job("off", "mobilenetv2", "image")
        c.submit(j)
        self.assertTrue(c.wait_all())
        self.assertIs(j.status, JobStatus.FAILED)
        self.assertIn("unavailable", j.error)
        c.close()

    def test_resource_aware_profile_with_fake_times(self):
        fake_profile = ServiceTimeProfile({
            ("mobilenetv2", Device.CPU): .030,
            ("mobilenetv2", Device.GPU): .005,
        })
        c, _, _ = controller(scheduler=ResourceAwareScheduler(fake_profile))
        c.start()
        j=Job("r1", "mobilenetv2", "image")
        c.submit(j)
        self.assertTrue(c.wait_all())
        self.assertIs(j.assigned_device, Device.GPU)
        c.close()

    def test_submit_validates_schema_before_queueing(self):
        c, _, _ = controller()
        c.start()
        invalid = [
            Job("", "mobilenetv2", "image"),
            Job("bad-model", "unknown", "image"),
            Job("bad-input", "mobilenetv2", ""),
        ]
        for j in invalid:
            with self.assertRaises(ValueError):
                c.submit(j)
        self.assertEqual(c.queues.reconcile(), (0, 0))
        c.close()

    def test_second_backend_prepare_failure_cleans_up_both(self):
        cpu = FakeBackend(Device.CPU)
        gpu = FakeBackend(Device.GPU, prepare_error=True)
        c = Controller(RoundRobinScheduler(), {Device.CPU:cpu, Device.GPU:gpu})
        with self.assertRaisesRegex(RuntimeError, "prepare error"):
            c.start()
        self.assertTrue(cpu.closed and gpu.closed)
        self.assertFalse(c._workers)

    def test_stress_mixed_success_and_failure(self):
        c, _, _ = controller(delay=.00015)
        c.start()
        jobs = [Job(str(i), "resnet50" if i%3 else "mobilenetv2",
                    "error" if i%17==0 else "image") for i in range(300)]
        for j in jobs:
            c.submit(j)
        c.close(timeout=10)
        self.assertEqual(c.queues.reconcile(), (300,300))
        self.assertEqual(sum(j.status is JobStatus.FAILED for j in jobs),18)
        self.assertEqual(sum(j.status is JobStatus.COMPLETED for j in jobs),282)
        self.assertTrue(all(not t.is_alive() for t in c._workers))


if __name__ == "__main__":
    unittest.main()
