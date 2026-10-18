"""Run: PYTHONPATH=src python3 scripts/demo_week2.py"""
from edge_scheduler.controller import Controller
from edge_scheduler.models import Device, Job
from edge_scheduler.schedulers import RoundRobinScheduler

class FakeBackend:
    def __init__(self, device): self.device = device
    def prepare(self): pass
    def infer(self, model_id, input_ref):
        if input_ref == "broken":
            raise ValueError("synthetic input failure")
        return {"model": model_id, "on": self.device.value}
    def close(self): pass

controller = Controller(RoundRobinScheduler(), {d: FakeBackend(d) for d in (Device.CPU, Device.GPU)})
controller.start()
jobs = [
    Job("job-001", "mobilenetv2", "image1.jpg"),
    Job("job-002", "resnet50", "image2.jpg"),
    Job("job-003", "mobilenetv2", "broken"),
    Job("job-004", "resnet50", "image4.jpg"),
]
for job in jobs:
    controller.submit(job)
finished = controller.wait_all(timeout=5)
for job in jobs:
    print(f"{job.job_id}: device={job.assigned_device.value if job.assigned_device else 'NONE'}, status={job.status.value}, error={job.error}")
print("RECONCILE", controller.queues.reconcile(), "finished", finished)
controller.close()
if not finished:
    raise SystemExit("INCOMPLETE")
