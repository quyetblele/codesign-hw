# T10/T11 implementation and acceptance — 29/10/2026

## Delivered

- `make run`: T10 live terminal, configurable `WINDOW=5` and `UPDATE=1`, resource sampling independent of display refresh. Redirected output uses plain lines.
- Streaming MetricsCollector with lifecycle guards, a bounded asynchronous writer, `events.jsonl`, terminal `jobs.csv`, final metrics shared with controller reporting, and scheduler-overhead measurement around snapshot/selection/enqueue.
- Recent throughput and latency cover successful completions in the selected window; startup divides throughput by actual elapsed workload time. Empty latency is unavailable, never zero.
- `make profile`: fresh subprocess for every model/device/repeat, single-model backend selection, baseline/ready/inference memory observations, raw CSV, profile summary CSV and machine-readable JSON in `logs/profiling/<run_id>/`.
- `ServiceTimeProfile.from_file`: validates successful untraced calibration, completeness and finite positive service time. This does not activate or complete Resource-aware scheduling.
- `make report`: selected `RUN=...` or latest run; detects type from metadata, supports historical pipeline configs, writes Markdown and standalone PNG files under the selected run's `report/` directory.
- UTC timestamp plus eight UUID characters identifies each run. Atomic latest pointers include path, type, start time and status; completing an older run does not replace a newer run's pointer.

## Commands

Run from the repository root; Make selects `.venv/bin/python` and sets `PYTHONPATH`.

```bash
make run
make run WINDOW=1 UPDATE=0.5
make profile
make profile DEVICE=cpu RUNS=100 REPEATS=3
make report
make report RUN=logs/pipeline/20261029T034929Z_6660920a
make report RUN=logs/profiling/20261029T035002Z_5305ba8b
make test
```

Defaults: run has 1000 jobs, Round Robin, Recent 5 seconds and refresh 1 second; profiling has 100 inferences per pair per repeat, 3 repeats, 5 warm-ups, CPU and GPU, both models. Extra CLI options can be passed with `ARGS="..."`.

## Verification

- Full suite outside the sandbox: **113/113 PASS**, no skips, 18.264 seconds. CUDA inference, output comparisons, compute traces and NVML VRAM tests all passed.
- Default regression before the additional tests: 99 cases, 96 pass and 3 GPU skips. The 14 added tests cover collector formulas/lifecycle/Recent, disk failure, queue saturation, live integration, isolated CPU profiling, failed profile rejection, selected model loading and report selection/artifacts.
- GPU pipeline tests inside the sandbox failed with CUDA failure 35 / unavailable NVML. These did not recur outside the sandbox; no CPU fallback was introduced.
- `pip check`: no broken requirements. Added `matplotlib==3.10.7` for headless PNG export.
- `git diff --check`: clean.

### T10 sample

Command: `make run JOBS=100 WINDOW=1 UPDATE=0.2 ARGS='--warmup 1 --require-gpu-metrics'`.

[Summary](../logs/pipeline/20261029T034929Z_6660920a/summary.json), [report](../logs/pipeline/20261029T034929Z_6660920a/report/report.md).

100 accepted = 100 completed, failed=0, unfinished=0. Exactly 400 events written,
dropped events=0, no writer errors. Collector and controller final metrics agree.
Live terminal exercised prepare/workload/drain/shutdown. Workers, sampler and
writers stopped after completion.

### T11 sample

Command: `make profile RUNS=20 REPEATS=2 ARGS='--warmup 1 --require-gpu-metrics'`.

[Summary](../logs/profiling/20261029T035002Z_5305ba8b/summary.json), [profile table](../logs/profiling/20261029T035002Z_5305ba8b/profile_summary.csv), [report](../logs/profiling/20261029T035002Z_5305ba8b/report/report.md).

All four model/device pairs passed, 40 measured inferences per pair across two
repeats. Eight trials have eight distinct PIDs. Each trial only loaded its selected
model, recorded real RSS/VRAM when applicable, dropped no resource samples and
stopped its sampler/writer. This is command acceptance data, not a full benchmark.

Both explicit-run and latest report commands succeeded. Pipeline report contains
four charts (queues, latency, waiting/execution, resources); profiling report
contains two charts (execution distribution and memory deltas). PNGs were visually
inspected. Reporting does not rerun inference.

## Limits

Pipeline still prepares CPU and GPU backends for every policy. Job/event history
is retained for a finite run; this is not an indefinite-load memory guarantee.
Profiling RSS covers the process including runtime libraries; VRAM covers the
whole selected GPU. Baseline deltas are signed estimates, maxima are sampled
observations, and short-lived peaks may be missed. ORT-traced profiles are for
verification and rejected as calibration. Failed/incomplete trials are not used
as successful calibration. Thread settings and memory scope are saved in config.

Workload manifest/replay, remaining-time estimation, complete Resource-aware
resource/freshness decisions and the four-scheduler benchmark remain outside this
change. The T11 profile loader supplies data but does not claim these tasks complete.
