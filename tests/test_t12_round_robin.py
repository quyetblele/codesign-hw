"""T12: scheduling/replay contract tests using real worker threads and fake backends."""
import unittest
import numpy as np
from edge_scheduler.core.models import Device, JobStatus
from edge_scheduler.runtime.schedulers import RoundRobinScheduler
from edge_scheduler.runtime.controller import Controller
from edge_scheduler.workload.generator import generate_workload
from edge_scheduler.workload.manifest import WorkloadManifest, WorkloadItem
from edge_scheduler.workload.replay import replay_workload

class FakeBackend:
    def __init__(self,device,fail=()):
        self.device = device
        self.fail = set(fail)
        self.closed = False
        self.calls = []
    def prepare(self):
        pass
    def infer(self,model_id,input_ref):
        self.calls.append((model_id,input_ref))
        if input_ref in self.fail:
            raise ValueError("injected failure")
        return np.zeros((1,1000),dtype=np.float32)
    def close(self):
        self.closed = True

class RoundRobinTests(unittest.TestCase):
    def test_policy_distribution_for_one_ten_eleven(self):
        for count in (1,10,11):
            with self.subTest(count=count):
                p = RoundRobinScheduler()
                m = generate_workload(count=count,models=("mobilenetv2","resnet50"),
                                      input_ref="inputs/test.jpg")
                choices = [p.choose_device(item,{Device.CPU:(),Device.GPU:()}) for item in m.items]
                self.assertEqual(choices,[Device.CPU if i%2==0 else Device.GPU for i in range(count)])

    def test_replay_burst_through_controller_workers(self):
        cpu,gpu = FakeBackend(Device.CPU),FakeBackend(Device.GPU)
        c = Controller(RoundRobinScheduler(),{Device.CPU:cpu,Device.GPU:gpu})
        m = generate_workload(count=11,models=("mobilenetv2","resnet50"),
                              input_ref="inputs/test.jpg",pattern="burst",seed=7,
                              burst_size=4,burst_gap_s=.008)
        try:
            c.start()
            rows = replay_workload(m,c.submit,lambda:(c.dispatch_pending(),c.drain_events()),
                                   poll_interval_s=.001)
            self.assertTrue(c.wait_all(timeout=5))
            self.assertEqual(len(rows),11)
            self.assertEqual(c.queues.reconcile(),(11,11))
            for i,item in enumerate(m.items):
                job = c.jobs[item.job_id]
                self.assertEqual(job.status,JobStatus.COMPLETED)
                self.assertEqual(job.assigned_device,Device.CPU if i%2==0 else Device.GPU)
            self.assertEqual(len(cpu.calls),6)
            self.assertEqual(len(gpu.calls),5)
        finally:
            c.close(timeout=5)
        self.assertTrue(cpu.closed and gpu.closed)

    def test_failed_job_does_not_lose_remaining(self):
        cpu = FakeBackend(Device.CPU,fail=("inputs/fail.jpg",))
        gpu = FakeBackend(Device.GPU)
        c = Controller(RoundRobinScheduler(),{Device.CPU:cpu,Device.GPU:gpu})
        m = WorkloadManifest("batch",0,(
            WorkloadItem("j1","mobilenetv2","inputs/fail.jpg",0),
            WorkloadItem("j2","resnet50","inputs/test.jpg",0),
            WorkloadItem("j3","mobilenetv2","inputs/test.jpg",0)))
        try:
            c.start()
            replay_workload(m,c.submit,lambda:(c.dispatch_pending(),c.drain_events()))
            self.assertTrue(c.wait_all(timeout=4))
            self.assertEqual(c.queues.reconcile(),(3,3))
            self.assertEqual(c.jobs["j1"].status,JobStatus.FAILED)
            self.assertEqual(c.jobs["j2"].status,JobStatus.COMPLETED)
            self.assertEqual(c.jobs["j3"].status,JobStatus.COMPLETED)
        finally:
            c.close(timeout=4)

if __name__ == "__main__":
    unittest.main()
