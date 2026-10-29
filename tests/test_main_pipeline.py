import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import edge_scheduler.main as app
from edge_scheduler.core.models import Device, Job
from edge_scheduler.runtime.controller import Controller
from edge_scheduler.runtime.run_results import job_row, summarize_jobs


class FakeBackend:
    def __init__(self, device):
        self.device = device
        self.prepared = False
        self.closed = False
        self.prepare_error = False
        self.session_providers = {}

    def prepare(self):
        if self.prepare_error:
            raise RuntimeError("GPU initialization failed")
        self.prepared = True

    def infer(self, model_id, input_ref):
        time.sleep(0.01)
        if input_ref == "missing.jpg":
            raise FileNotFoundError(input_ref)
        output = np.zeros((1, 1000), np.float32)
        output[0, 1 if self.device is Device.CPU else 2] = 1
        return output

    def close(self):
        self.closed = True


class MainPipelineTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.cpu, self.gpu = FakeBackend(Device.CPU), FakeBackend(Device.GPU)
        for name, backend in (("CPUBackend", self.cpu), ("GPUBackend", self.gpu)):
            patcher = patch.object(app, name, return_value=backend)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def run_main(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = app.main([
                "--jobs", "6", "--warmup", "1", "--monitor-interval", "0.01",
                "--no-gpu-telemetry", "--output-dir", str(self.directory),
                *arguments,
            ])
        directory = next(path for path in self.directory.iterdir() if path.is_dir())
        report = json.loads((directory / "summary.json").read_text())
        return code, directory, report

    def assert_closed(self, report):
        self.assertTrue(report["controller"]["closed"])
        self.assertEqual(report["controller"]["running_workers"], [])
        self.assertFalse(report["monitor"]["running"])
        self.assertFalse(report["logger"]["running"])
        self.assertTrue(self.cpu.closed and self.gpu.closed)

    def test_round_robin_routes_jobs_and_writes_all_artifacts(self):
        code, directory, report = self.run_main()
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["jobs"]["completed_jobs"], 6)
        self.assertTrue(report["jobs"]["reconciled"])
        with (directory / "jobs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["device"] for row in rows], ["cpu", "gpu"] * 3)
        self.assertEqual([row["top1_class"] for row in rows], ["1", "2"] * 3)
        self.assertTrue(all(float(row["latency_s"]) >= float(row["execution_s"]) for row in rows))
        self.assertTrue((directory / "config.json").is_file())
        with (directory / "resources.csv").open() as stream:
            samples = list(csv.DictReader(stream))
        self.assertGreater(len(samples), 0)
        self.assertTrue(any(json.loads(row["workers_json"])["cpu"]["available"] for row in samples))
        self.assertTrue(all(set(json.loads(row["queue_lengths_json"])) == {"incoming", "cpu", "gpu"}
                            for row in samples))
        self.assertEqual(report["logger"]["dropped_samples"], 0)
        self.assert_closed(report)

    def test_cpu_only_routes_all_jobs_to_cpu_but_prepares_both_backends(self):
        code, _, report = self.run_main("--scheduler", "cpu-only")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["jobs"]["devices"]["cpu"]["completed"], 6)
        self.assertEqual(report["jobs"]["devices"]["gpu"]["assigned"], 0)
        self.assertTrue(self.cpu.prepared and self.gpu.prepared)
        self.assert_closed(report)

    def test_gpu_only_routes_all_jobs_to_gpu(self):
        code, _, report = self.run_main("--scheduler", "gpu-only")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["jobs"]["devices"]["gpu"]["completed"], 6)
        self.assertEqual(report["jobs"]["devices"]["cpu"]["assigned"], 0)
        self.assert_closed(report)

    def test_bad_input_is_logged_as_failed_jobs_and_nonzero_exit(self):
        code, directory, report = self.run_main("--input", "missing.jpg")
        self.assertEqual(code, 1)
        self.assertEqual(report["jobs"]["failed_jobs"], 6)
        self.assertEqual(report["jobs"]["completed_jobs"], 0)
        self.assertTrue(report["jobs"]["reconciled"])
        with (directory / "jobs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row["error"] == "missing.jpg" for row in rows))
        self.assert_closed(report)

    def test_prepare_failure_cleans_partial_startup_and_reports_zero_acceptance(self):
        self.gpu.prepare_error = True
        code, directory, report = self.run_main()
        self.assertEqual(code, 1)
        self.assertTrue(any("initialization failed" in error for error in report["errors"]))
        self.assertEqual(report["jobs"]["accepted_jobs"], 0)
        self.assertTrue((directory / "jobs.csv").read_text().startswith("job_id,"))
        self.assert_closed(report)

    def test_workload_timeout_still_drains_all_accepted_jobs(self):
        original = Controller.wait_all
        calls = []

        def wait(controller, timeout):
            calls.append(timeout)
            return False if len(calls) == 1 else original(controller, timeout)

        with patch.object(Controller, "wait_all", wait):
            code, _, report = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(report["jobs"]["completed_jobs"], 6)
        self.assertTrue(any("workload timeout" in error for error in report["errors"]))
        self.assert_closed(report)

    def test_shutdown_timeout_saves_pending_jobs_and_retries_cleanup(self):
        original_close = Controller.close
        original_wait = Controller.wait_all
        checkpoints, waits, closes = [], [], []

        def wait(controller, timeout):
            waits.append(timeout)
            return False if len(waits) == 1 else original_wait(controller, timeout)

        def close(controller, timeout):
            closes.append(timeout)
            if len(closes) == 1:
                raise TimeoutError("test drain timeout")
            checkpoints.append(json.loads(next(self.directory.glob("*/summary.json")).read_text()))
            return original_close(controller, timeout)

        with patch.object(Controller, "wait_all", wait), patch.object(Controller, "close", close):
            code, _, report = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(checkpoints[0]["status"], "draining")
        self.assertEqual(len(checkpoints[0]["jobs"]["unfinished_job_ids"]), 6)
        self.assertTrue(report["jobs"]["reconciled"])
        self.assert_closed(report)

    def test_interrupt_stops_acceptance_and_drains_already_accepted_job(self):
        original = Controller.submit
        calls = []

        def submit(controller, job):
            calls.append(job)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return original(controller, job)

        with patch.object(Controller, "submit", submit):
            code, _, report = self.run_main()
        self.assertEqual(code, 130)
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["jobs"]["accepted_jobs"], 1)
        self.assertEqual(report["jobs"]["completed_jobs"], 1)
        self.assert_closed(report)

    def test_resource_sink_failure_is_not_reported_as_success(self):
        with patch.object(app.ResourceCSVLogger, "record_resource", side_effect=OSError("disk full")):
            code, _, report = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(report["jobs"]["completed_jobs"], 6)
        self.assertIn("disk full", report["monitor"]["sink_error"])
        self.assert_closed(report)

    def test_invalid_cli_values_fail_before_creating_output(self):
        for arguments in (["--jobs", "0"], ["--monitor-interval", "nan"],
                          ["--shutdown-timeout", "inf"], ["--models", "unknown"],
                          ["--input", "/absolute/image.jpg"],
                          ["--no-gpu-telemetry", "--require-gpu-metrics"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    app.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)
        self.assertEqual(list(self.directory.iterdir()), [])


class RunResultsTest(unittest.TestCase):
    def test_metrics_use_successful_jobs_but_throughput_includes_last_failure(self):
        jobs = []
        for index, (arrival, dispatch, started, finished) in enumerate(((1, 2, 3, 5), (2, 2.5, 3, 6))):
            job = Job(str(index), "mobilenetv2", "image", arrival_time=arrival)
            job.mark_scheduled(Device.CPU, dispatch)
            job.mark_running(started)
            job.mark_completed(np.zeros((1, 1000), np.float32), finished)
            jobs.append(job)
        failed = Job("failed", "mobilenetv2", "image", arrival_time=3)
        failed.mark_failed("dispatch failure", 7)
        jobs.append(failed)
        summary = summarize_jobs(jobs, 3, 3)
        self.assertAlmostEqual(summary["throughput_jobs_s"], 2 / 6)
        self.assertAlmostEqual(summary["failure_rate"], 1 / 3)
        self.assertEqual(summary["successful_job_metrics"]["latency_ms"]["p95"], 4000)
        self.assertEqual(summary["successful_job_metrics"]["execution_ms"]["mean"], 2500)
        self.assertIsNone(job_row(failed)["execution_s"])
        self.assertTrue(summary["reconciled"])


@unittest.skipUnless(os.environ.get("T06_TEST_GPU") == "1" and os.environ.get("T08_TEST_GPU", "1") == "1",
                     "set T06_TEST_GPU=1 T08_TEST_GPU=1 for the real controller/ONNX/NVML pipeline")
class RealPipelineIntegrationTest(unittest.TestCase):
    def test_three_policies_with_real_backends_resources_and_cuda_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            for policy in ("cpu-only", "gpu-only", "round-robin"):
                with self.subTest(policy=policy):
                    output = Path(temporary) / policy
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        code = app.main(["--scheduler", policy, "--jobs", "4", "--warmup", "1",
                                         "--monitor-interval", "0.05", "--require-gpu-metrics",
                                         "--profile", "--output-dir", str(output)])
                    directory = next(path for path in output.iterdir() if path.is_dir())
                    report = json.loads((directory / "summary.json").read_text())
                    self.assertEqual(code, 0, report)
                    self.assertEqual(report["jobs"]["completed_jobs"], 4)
                    self.assertTrue(report["jobs"]["reconciled"])
                    self.assertEqual(report["controller"]["running_workers"], [])
                    self.assertFalse(report["monitor"]["running"])
                    self.assertFalse(report["logger"]["running"])
                    expected_cpu = 4 if policy == "cpu-only" else 0 if policy == "gpu-only" else 2
                    self.assertEqual(report["jobs"]["devices"]["cpu"]["assigned"], expected_cpu)
                    self.assertGreater(report["resources"]["metrics"]["gpu_memory_total_mb"]["valid_samples"], 0)
                    for model in ("mobilenetv2", "resnet50"):
                        self.assertTrue(report["profiles"]["gpu"][model]["cuda_compute_verified"])


if __name__ == "__main__":
    unittest.main()
