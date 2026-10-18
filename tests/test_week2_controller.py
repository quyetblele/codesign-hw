import time
import unittest
from threading import Thread
from edge_scheduler.controller import Controller
from edge_scheduler.models import Device, Job, JobStatus
from edge_scheduler.schedulers import RoundRobinScheduler, CpuOnlyScheduler, GpuOnlyScheduler

class FakeBackend:
    def __init__(self, device, delay=0):
        self.device = device
        self.delay = delay
        self.prepared = False
        self.closed = False
    def prepare(self): self.prepared = True
    def infer(self, model_id, input_ref):
        time.sleep(self.delay)
        if input_ref == 'broken':
            raise ValueError('test backend error')
        return (model_id, self.device.value, input_ref)
    def close(self): self.closed = True

class Collector:
    def __init__(self): self.events = []; self.flushed = False; self.closed = False
    def record(self, event): self.events.append(event)
    def flush(self): self.flushed = True
    def close(self): self.closed = True

def create(policy=None, delay=0):
    cpu, gpu = FakeBackend(Device.CPU, delay), FakeBackend(Device.GPU, delay)
    collector = Collector()
    controller = Controller(policy or RoundRobinScheduler(), {Device.CPU:cpu, Device.GPU:gpu}, collector=collector)
    controller.start()
    return controller, cpu, gpu, collector

class Week2Tests(unittest.TestCase):
    def test_many_jobs_round_robin(self):
        c, cpu, gpu, coll = create(delay=.001)
        jobs = [Job(str(i), 'mobilenetv2', 'x') for i in range(30)]
        for job in jobs: c.submit(job)
        self.assertTrue(c.wait_all())
        self.assertTrue(c.queues.is_reconciled())
        self.assertEqual([j.status for j in jobs], [JobStatus.COMPLETED]*30)
        self.assertEqual([j.assigned_device for j in jobs], [Device.CPU, Device.GPU]*15)
        self.assertEqual(len([e for e in coll.events if e.kind.value == 'COMPLETED']),30)
        c.close()
        self.assertTrue(cpu.closed and gpu.closed and coll.flushed and coll.closed)

    def test_failure_does_not_lose_job(self):
        c,*_ = create()
        ok = Job('ok', 'mobilenetv2', 'x')
        fail = Job('fail', 'resnet50', 'broken')
        c.submit(ok); c.submit(fail)
        self.assertTrue(c.wait_all())
        self.assertEqual(ok.status, JobStatus.COMPLETED)
        self.assertEqual(fail.status, JobStatus.FAILED)
        self.assertIn('test backend error', fail.error)
        self.assertEqual(c.queues.reconcile(), (2,2))
        c.close()

    def test_duplicates_and_invalid_resubmit(self):
        c,*_ = create()
        c.submit(Job('same','mobilenetv2','x'))
        with self.assertRaises(ValueError): c.submit(Job('same','mobilenetv2','x'))
        with self.assertRaises(ValueError): c.submit(Job('again','mobilenetv2','x', status=JobStatus.RUNNING))
        self.assertTrue(c.wait_all())
        c.close()
        with self.assertRaises(RuntimeError): c.submit(Job('late','mobilenetv2','x'))

    def test_wrong_message_rejected(self):
        from edge_scheduler.contracts import WorkerMessage, EventKind
        c,*_ = create(policy=CpuOnlyScheduler(), delay=.02)
        j=Job('one','mobilenetv2','x')
        c.submit(j); c.dispatch_pending()
        with self.assertRaises(ValueError):
            c.handle_worker_message(WorkerMessage(EventKind.STARTED,'one',Device.GPU,time.monotonic()))
        self.assertTrue(c.wait_all())
        c.close()

    def test_dispatch_error_recorded(self):
        class BadPolicy:
            def choose_device(self, job, queued_jobs, snapshot=None):
                raise ValueError('no capacity')
        c,*_ = create(policy=BadPolicy())
        job=Job('bad','mobilenetv2','x')
        c.submit(job)
        self.assertTrue(c.wait_all())
        self.assertEqual(job.status,JobStatus.FAILED)
        self.assertIn('no capacity',job.error)
        c.close()

    def test_parallel_submit(self):
        c,*_ = create(delay=.001)
        jobs=[Job(str(i),'resnet50','x') for i in range(40)]
        threads=[Thread(target=c.submit,args=(job,)) for job in jobs]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertTrue(c.wait_all())
        self.assertEqual(c.queues.reconcile(), (40,40))
        c.close()

if __name__ == '__main__':
    unittest.main()
