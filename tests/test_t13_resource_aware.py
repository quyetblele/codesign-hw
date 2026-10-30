import unittest
from unittest.mock import patch
import math
from edge_scheduler.core.models import Device, Job
from edge_scheduler.core.interfaces import ResourceSnapshot, WorkerState
from edge_scheduler.runtime.schedulers import ResourceAwareScheduler, ServiceTimeProfile

class T13Tests(unittest.TestCase):
    def setUp(self):
        self.profile=ServiceTimeProfile({("mobilenetv2",Device.CPU):0.05,("mobilenetv2",Device.GPU):0.01})
        self.scheduler=ResourceAwareScheduler(self.profile)
        self.job=Job("test","mobilenetv2","inputs/test.jpg")
    def test_choose_gpu_idle(self):
        self.assertEqual(self.scheduler.choose_device(self.job, {Device.CPU:(),Device.GPU:()}),Device.GPU)
    def test_cpu_when_gpu_congested(self):
        queued=[Job(str(i),"mobilenetv2","inputs/test.jpg") for i in range(6)]
        self.assertEqual(self.scheduler.choose_device(self.job,{Device.CPU:(),Device.GPU:queued}),Device.CPU)
    def test_unavailable_gpu(self):
        snap=ResourceSnapshot(100,0,500,500,{Device.GPU:WorkerState(Device.GPU,False)})
        with patch("edge_scheduler.runtime.schedulers.time.monotonic",return_value=100):
            self.assertEqual(self.scheduler.choose_device(self.job,{},snap),Device.CPU)
    def test_unknown_running_not_zero(self):
        snap=ResourceSnapshot(100,0,500,500,{Device.GPU:WorkerState(Device.GPU,True,running_model_id="mobilenetv2")})
        with patch("edge_scheduler.runtime.schedulers.time.monotonic",return_value=100):
            self.assertAlmostEqual(self.scheduler.estimated_finish_delay(Device.GPU,self.job,{},snap),.02)
    def test_nonfinite_profile(self):
        for value in (float("nan"),float("inf"),0,-1):
            with self.assertRaises(ValueError):
                ServiceTimeProfile({("mobilenetv2",Device.GPU):value}).estimate("mobilenetv2",Device.GPU)
    def test_decision_log(self):
        self.scheduler.choose_device(self.job,{})
        self.assertEqual(self.scheduler.decisions[-1]["selected_device"],"gpu")
    def test_both_unavailable(self):
        snap=ResourceSnapshot(100,0,500,500,{d:WorkerState(d,False) for d in Device})
        with patch("edge_scheduler.runtime.schedulers.time.monotonic",return_value=100):
            with self.assertRaises(RuntimeError):
                self.scheduler.choose_device(self.job,{},snap)
if __name__=="__main__":
    unittest.main()
