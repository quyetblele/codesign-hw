"""T09: replay determinism, validation and timing acceptance tests."""
import csv
from pathlib import Path
import tempfile
import unittest
from edge_scheduler.workload.generator import generate_workload
from edge_scheduler.workload.manifest import WorkloadManifest, WorkloadItem
from edge_scheduler.workload.replay import replay_workload, summarize_replay, write_replay_csv

class Clock:
    def __init__(self):
        self.now = 100.
    def time(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds

class WorkloadAcceptance(unittest.TestCase):
    def test_seed_reproducibility(self):
        opts = dict(count=40, models=("mobilenetv2","resnet50"),
                    input_ref="inputs/test.jpg", pattern="high", seed=42)
        a = generate_workload(**opts)
        self.assertEqual(a.canonical_json(), generate_workload(**opts).canonical_json())
        self.assertEqual(a.sha256, generate_workload(**opts).sha256)
        self.assertNotEqual(a.sha256, generate_workload(**{**opts, "seed": 43}).sha256)

    def test_manifest_round_trip(self):
        m = generate_workload(count=9, models=("mobilenetv2",), input_ref="inputs/test.jpg", pattern="burst", seed=7)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"workload.json"
            m.save(p)
            self.assertEqual(m, WorkloadManifest.load(p))
            self.assertEqual(m.sha256, WorkloadManifest.load(p).sha256)
            with self.assertRaises(FileExistsError):
                m.save(p)

    def test_patterns_monotonic_and_bursts(self):
        for pattern in ("batch","low","medium","high","burst"):
            with self.subTest(pattern=pattern):
                m = generate_workload(count=13, models=("mobilenetv2",), input_ref="inputs/test.jpg",
                                      pattern=pattern, seed=8)
                times = [x.arrival_offset_s for x in m.items]
                self.assertEqual(times, sorted(times))
                if pattern == "batch":
                    self.assertEqual(set(times), {0})
                if pattern == "burst":
                    self.assertEqual(times[:5], [0.0]*5)
                    self.assertEqual(times[5:10], [0.25]*5)

    def test_duplicate_and_invalid_path(self):
        j = WorkloadItem("j1","mobilenetv2","inputs/test.jpg",0)
        with self.assertRaises(ValueError):
            WorkloadManifest("batch",0,(j,j))
        for path in ("../secret.jpg", "/tmp/absolute.jpg", "inputs\\bad.jpg", ""):
            with self.assertRaises(ValueError):
                WorkloadItem("job","mobilenetv2",path,0)

    def test_reject_invalid_numbers(self):
        for num in (-0.1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):
                WorkloadItem("j","resnet50","inputs/test.jpg",num)
        with self.assertRaises(ValueError):
            generate_workload(count=0,models=("mobilenetv2",), input_ref="inputs/test.jpg")

    def test_replay_does_not_wait_for_previous_completion(self):
        m = WorkloadManifest("burst",8,(
            WorkloadItem("j1","mobilenetv2","inputs/test.jpg",0),
            WorkloadItem("j2","resnet50","inputs/test.jpg",0.01),
            WorkloadItem("j3","mobilenetv2","inputs/test.jpg",0.01),
            WorkloadItem("j4","resnet50","inputs/test.jpg",0.02)))
        c = Clock()
        calls, pumps = [], []
        def submit(job):
            calls.append((job.job_id,c.now))
        rows = replay_workload(m,submit,lambda:pumps.append(c.now),
                               clock=c.time,sleep=c.sleep,poll_interval_s=.004)
        self.assertEqual([x[0] for x in calls],["j1","j2","j3","j4"])
        self.assertAlmostEqual(calls[-1][1],100.02)
        self.assertGreater(len(pumps),4)
        self.assertAlmostEqual(max(abs(x.jitter_s) for x in rows),0,places=7)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"replay.csv"
            write_replay_csv(p,rows)
            with p.open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),4)

    def test_replay_measures_submission_delay(self):
        c = Clock()
        m = WorkloadManifest("batch",0,(WorkloadItem("j","mobilenetv2","inputs/test.jpg",0),))
        def slow_submit(job):
            c.now += .025
        rows = replay_workload(m,slow_submit,lambda:None,clock=c.time,sleep=c.sleep)
        self.assertAlmostEqual(rows[0].jitter_s,.025)
        self.assertAlmostEqual(summarize_replay(rows)["max_jitter_ms"],25)

if __name__ == "__main__":
    unittest.main()
