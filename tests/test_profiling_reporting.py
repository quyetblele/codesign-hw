import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.artifacts import mark_latest
from edge_scheduler.inference import profiling
from edge_scheduler.inference.cpu_backend import CPUBackend
from edge_scheduler.core.models import Device
from edge_scheduler.reporting import generate_report, main as report_main, memory_unit, select_run
from edge_scheduler.runtime.schedulers import ServiceTimeProfile
import test_main_pipeline as fixtures


class ProfilingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_real_cpu_profiles_are_isolated_exported_and_loadable(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = profiling.main(["--device", "cpu", "--runs", "3", "--repeats", "2", "--warmup", "1",
                                   "--interval", "0.01", "--output-dir", str(self.directory)])
        self.assertEqual(code, 0)
        pointer = json.loads((self.directory / "latest.json").read_text())
        run = Path(pointer["path"])
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["memory_unit"], "MB")
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(json.loads((run / "config.json").read_text())["memory_unit"], "MB")
        self.assertEqual(len(summary["profiles"]), 2)
        pids = set()
        for trial in summary["trials"]:
            self.assertEqual(list(trial["session_providers"]), [trial["model_id"]])
            self.assertFalse(trial["monitor"]["running"])
            self.assertFalse(trial["logger"]["running"])
            pids.add(trial["resource_metadata"]["system"]["pid"])
        self.assertEqual(len(pids), 4)
        profile = ServiceTimeProfile.from_file(run / "profile.json")
        self.assertGreater(profile.estimate("mobilenetv2", Device.CPU), 0)
        with (run / "measurements.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)
        for row in summary["profiles"]:
            self.assertEqual(row["samples"], 6)
            self.assertEqual(row["repeats"], 2)
            self.assertIsNone(row["delta_gpu_memory_used_mb"])
            self.assertGreater(row["ready_process_rss_mb"], 0)
        # Metadata, rather than folder name, selects report type after relocation.
        renamed = self.directory / "descriptive-folder"
        shutil.move(str(run), str(renamed))
        report = generate_report(renamed)
        self.assertTrue(report.exists())
        self.assertIn("Model/device profiling", report.read_text())
        self.assertIn("RSS delta MB", report.read_text())
        self.assertIn("Memory unit: **MB**", report.read_text())
        self.assertEqual(len(list((renamed / "report" / "figures").glob("*.png"))), 2)

    def test_bad_input_is_failed_and_not_loadable_for_calibration(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = profiling.main(["--device", "cpu", "--models", "mobilenetv2", "--input", "missing.jpg",
                                   "--runs", "1", "--repeats", "1", "--warmup", "1", "--interval", "0.01",
                                   "--output-dir", str(self.directory)])
        self.assertEqual(code, 1)
        run = Path(json.loads((self.directory / "latest.json").read_text())["path"])
        with self.assertRaisesRegex(ValueError, "successful"):
            ServiceTimeProfile.from_file(run / "profile.json")
        self.assertTrue(generate_report(run).exists())

    def test_invalid_profile_options_fail_before_creating_runs(self):
        for arguments in (["--runs", "0"], ["--interval", "nan"], ["--repeats", "0"],
                          ["--device", "cpu", "--require-gpu-metrics"], ["--models", "resnet50", "resnet50"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    profiling.main([*arguments, "--output-dir", str(self.directory)])
                self.assertEqual(raised.exception.code, 2)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_nonfinite_and_traced_profiles_are_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ServiceTimeProfile({("model", Device.CPU): value}).estimate("model", Device.CPU)
        path = self.directory / "profile.json"
        path.write_text(json.dumps({"run_type": "profiling", "status": "passed", "trace_enabled": True}))
        with self.assertRaisesRegex(ValueError, "traced"):
            ServiceTimeProfile.from_file(path)

    def test_selected_backend_does_not_load_other_model(self):
        backend = CPUBackend(model_ids=("mobilenetv2",), warmup_runs=1)
        try:
            backend.prepare()
            self.assertEqual(list(backend.session_providers), ["mobilenetv2"])
            with self.assertRaises(ValueError):
                backend.infer("resnet50", "inputs/test.jpg")
        finally:
            backend.close()


class ReportTest(unittest.TestCase):
    setUp = fixtures.MainPipelineTest.setUp
    run_main = fixtures.MainPipelineTest.run_main

    def test_explicit_run_and_env_run_override_latest_and_generate_four_pngs(self):
        code, directory, report = self.run_main()
        self.assertEqual(code, 0, report)
        with patch.dict(os.environ, {"RUN": str(directory)}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(report_main([]), 0)
        self.assertEqual(len(list((directory / "report" / "figures").glob("*.png"))), 4)
        self.assertIn("Scheduler overhead", (directory / "report" / "report.md").read_text())
        self.assertEqual(report["memory_unit"], "MB")
        self.assertIn("Memory unit: **MB**", (directory / "report" / "report.md").read_text())
        self.assertEqual(select_run(None, self.directory), directory)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            select_run(str(self.directory / "missing"), self.directory)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(report_main(["--run", str(self.directory / "missing")]), 1)

    def test_missing_required_artifact_fails_without_using_latest(self):
        _, directory, _ = self.run_main()
        (directory / "jobs.csv").unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(report_main(["--run", str(directory)]), 1)
        self.assertFalse((directory / "report" / "report.md").exists())

    def test_latest_uses_start_order_not_order_of_completion(self):
        first = self.directory / "first"
        second = self.directory / "second"
        for directory, started in ((first, "2026-10-29T01:00:00+00:00"), (second, "2026-10-29T02:00:00+00:00")):
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({"started_at": started}))
            mark_latest(directory, "pipeline", "running")
        mark_latest(first, "pipeline", "passed")
        self.assertEqual(select_run(None, self.directory), second)

    def test_historical_mib_report_preserves_saved_unit_and_values(self):
        _, directory, _ = self.run_main()
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config.update(memory_unit="MiB", schema_version=1)
        config_path.write_text(json.dumps(config))
        saved = (directory / "resources.csv").read_bytes()
        report = generate_report(directory)
        self.assertIn("Memory unit: **MiB** (1 MiB = 1,048,576 bytes)", report.read_text())
        self.assertEqual((directory / "resources.csv").read_bytes(), saved)
        self.assertEqual(memory_unit({}), "MiB")

    def test_unknown_memory_unit_fails_instead_of_mislabeling_data(self):
        with self.assertRaisesRegex(ValueError, "unsupported memory unit"):
            memory_unit({"memory_unit": "unknown"})
        with self.assertRaisesRegex(ValueError, "memory_unit is required"):
            memory_unit({"schema_version": 2})


if __name__ == "__main__":
    unittest.main()
