import contextlib
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler import benchmark as bench
from edge_scheduler import benchmark_report as report
from edge_scheduler import main as app
from edge_scheduler.core.models import Device
from edge_scheduler.reporting import generate_report
from edge_scheduler.runtime.schedulers import ServiceTimeProfile
from edge_scheduler.workload.generator import generate_workload
from test_main_pipeline import FakeBackend


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def fixture(self):
        manifest = bench.mixed_manifest(count=12, pattern="burst", seed=42,
                                        input_ref="inputs/test.jpg", burst_size=4, burst_gap=.01)
        manifest.save(self.directory / "burst.json")
        parent = self.directory / "runs"
        with patch.object(app, "CPUBackend", return_value=FakeBackend(Device.CPU)), \
             patch.object(app, "GPUBackend", return_value=FakeBackend(Device.GPU)), \
             contextlib.redirect_stdout(io.StringIO()):
            code = app.main(["--scheduler", "round-robin", "--manifest", str(self.directory / "burst.json"),
                             "--warmup", "1", "--monitor-interval", "0.01", "--no-gpu-telemetry",
                             "--output-dir", str(parent)])
        self.assertEqual(code, 0)
        child = bench.saved_run(parent)
        config = json.loads((child / "config.json").read_text())
        config.update(jobs=12, repeats=3, cooldown=2, output_dir=str(self.directory.parent),
                      run_id=self.directory.name, hardware=dict(platform="synthetic test fixture", physical_cores=1,
                      logical_cpus=2, ram_total_mb=100, gpu={"error": "disabled for test"}))
        experiment = dict(run_type="benchmark", status="passed", errors=[], config=config,
                          schedule=bench.execution_order(3),
                          manifests={"burst": dict(path="burst.json", sha256=manifest.sha256)},
                          runs=[dict(index=1, repeat=1, workload="burst", scheduler="round-robin", status="passed",
                                     run_path=str(child.relative_to(self.directory)), validation_errors=[])])
        (self.directory / "config.json").write_text(json.dumps(dict(run_type="benchmark")))
        (self.directory / "experiment.json").write_text(json.dumps(experiment))
        return manifest, child, experiment

    def test_matrix_has_48_unique_paired_slots_and_rotates_order(self):
        schedule = bench.execution_order(3)
        self.assertEqual([r["index"] for r in schedule], list(range(1, 49)))
        keys = [(r["workload"], r["repeat"], r["scheduler"]) for r in schedule]
        self.assertEqual(len(set(keys)), 48)
        self.assertEqual(set(Counter((r["workload"], r["scheduler"]) for r in schedule).values()), {3})
        for pattern in bench.PATTERNS:
            orders = [tuple(r["scheduler"] for r in schedule if r["workload"] == pattern and r["repeat"] == repeat)
                      for repeat in (1, 2, 3)]
            self.assertEqual(len(set(orders)), 3)

    def test_shuffled_mix_preserves_arrivals_and_counts(self):
        args = dict(count=100, pattern="high", seed=42, input_ref="inputs/test.jpg", rate=80)
        manifest = bench.mixed_manifest(**args)
        original = generate_workload(count=100, models=("mobilenetv2", "resnet50"),
                                     pattern="high", seed=42, input_ref="inputs/test.jpg", rate_jobs_s=80)
        self.assertEqual([i.arrival_offset_s for i in manifest.items], [i.arrival_offset_s for i in original.items])
        self.assertEqual(Counter(i.model_id for i in manifest.items), {"mobilenetv2": 50, "resnet50": 50})
        self.assertEqual(manifest.sha256, bench.mixed_manifest(**args).sha256)
        self.assertNotEqual([i.model_id for i in manifest.items], [i.model_id for i in original.items])
        for parity in (0, 1):
            self.assertEqual(set(i.model_id for i in manifest.items[parity::2]), {"mobilenetv2", "resnet50"})

    def test_calibration_requires_both_models_and_devices(self):
        with self.assertRaises(KeyError):
            bench.calibrated_rates(ServiceTimeProfile({("mobilenetv2", Device.CPU): .01}))
        profile = ServiceTimeProfile({(m, d): .02 if d is Device.CPU else .01
                                      for m in ("mobilenetv2", "resnet50") for d in Device})
        rates, capacities = bench.calibrated_rates(profile)
        self.assertEqual(capacities, {"cpu": 50, "gpu": 100})
        self.assertLess(rates["low"], rates["medium"])
        self.assertGreater(rates["high"], sum(capacities.values()))

    def test_invalid_cli_rejected_before_creating_experiment(self):
        for args in (["--repeats", "2"], ["--jobs", "50001"], ["--cooldown", "nan"],
                     ["--seed", "-1"], ["--no-gpu-telemetry", "--require-gpu-metrics"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.main([*args, "--output-dir", str(self.directory)])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_raw_identity_guard_detects_duplicate_terminal_and_wrong_hash(self):
        manifest, child, _ = self.fixture()
        self.assertEqual(bench.validate_run(child, manifest, "round-robin"), [])
        path = child / "jobs.csv"
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[-1]["job_id"] = rows[0]["job_id"]
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        issues = bench.validate_run(child, manifest, "round-robin")
        self.assertIn("missing/duplicate/unexpected terminal job IDs", issues)
        other = bench.mixed_manifest(count=12, pattern="burst", seed=43, input_ref="inputs/test.jpg")
        self.assertIn("manifest hash mismatch", bench.validate_run(child, other, "round-robin"))

    def test_report_recomputes_metrics_and_keeps_missing_gpu_unavailable(self):
        _, child, experiment = self.fixture()
        rows, _ = report.load_measurements(self.directory, experiment)
        self.assertEqual(rows[0]["status"], "passed", rows[0]["validation_errors"])
        self.assertIsNone(rows[0]["gpu_memory_used_mb_mean"])
        summary = json.loads((child / "summary.json").read_text())
        summary["jobs"]["throughput_jobs_s"] *= 2
        (child / "summary.json").write_text(json.dumps(summary))
        rows, _ = report.load_measurements(self.directory, experiment)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("raw job CSV metrics differ from summary", rows[0]["validation_errors"])

    def test_partial_matrix_generates_eight_figures_and_explicit_incomplete_report(self):
        self.fixture()
        path = generate_report(self.directory)
        self.assertEqual(path.name, "index.html")
        self.assertIn("data:image/png;base64,", path.read_text())
        analysis = json.loads((path.parent / "analysis.json").read_text())
        self.assertEqual(analysis["status"], "incomplete")
        self.assertEqual(sum(g["missing_repeats"] for g in analysis["groups"]), 47)
        self.assertEqual(len(list((path.parent / "figures").glob("*.png"))), 8)
        self.assertIn("INCOMPLETE", (path.parent / "report.md").read_text())

    def test_failed_and_unpaired_runs_are_not_used_for_performance_claims(self):
        rows = [dict(index=1, workload="high", repeat=1, scheduler="resource-aware", status="passed",
                     manifest_sha256="a", p95_ms=50, throughput_jobs_s=100),
                dict(index=2, workload="high", repeat=1, scheduler="gpu-only", status="passed",
                     manifest_sha256="a", p95_ms=100, throughput_jobs_s=80),
                dict(index=3, workload="high", repeat=2, scheduler="gpu-only", status="failed",
                     manifest_sha256="a", p95_ms=1, throughput_jobs_s=999)]
        pairs = report.paired_comparisons(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["p95_reduction_pct"], 50)
        self.assertEqual(pairs[0]["throughput_gain_pct"], 25)
        groups = report.compare(rows, 3)
        gpu = next(g for g in groups if g["workload"] == "high" and g["scheduler"] == "gpu-only")
        self.assertEqual(gpu["p95_ms_mean"], 100)
        self.assertEqual(gpu["failed_repeats"], 1)
        self.assertFalse(gpu["complete"])

    def test_hard_timeout_preserves_log_and_stops_child(self):
        result = bench.run_command([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
                                   self.directory / "process.log", .1)
        self.assertIn("timeout", result["error"])
        self.assertIsNotNone(result["returncode"])
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("started", (self.directory / "process.log").read_text())


if __name__ == "__main__":
    unittest.main()
