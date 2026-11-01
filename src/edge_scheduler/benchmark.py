"""T15: paired, sequential CPU/CUDA experiments with preserved raw artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import time

import psutil

from .artifacts import mark_latest, new_run
from .core.models import Device
from .inference.model_registry import MODEL_REGISTRY, PROJECT_ROOT, repository_file
from .runtime.run_results import write_json
from .runtime.schedulers import ServiceTimeProfile
from .workload.generator import generate_workload
from .workload.manifest import MAX_JOBS, WorkloadManifest

POLICIES = ("cpu-only", "gpu-only", "round-robin", "resource-aware")
PATTERNS = ("low", "medium", "high", "burst")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    paths += [PROJECT_ROOT / "requirements.txt", PROJECT_ROOT / "Makefile"]
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in paths}


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def hardware_metadata():
    process = psutil.Process()
    return {"platform": platform.platform(), "python": sys.version,
            "physical_cores": psutil.cpu_count(logical=False), "logical_cpus": os.cpu_count(),
            "cpu_affinity": process.cpu_affinity() if hasattr(process, "cpu_affinity") else None,
            "ram_total_mb": psutil.virtual_memory().total / 1e6,
            "lscpu": command_output(["lscpu", "--json"]),
            "gpu": command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,power.draw",
                                   "--format=csv,noheader,nounits"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": {name: version(name) for name in
                         ("numpy", "onnxruntime-gpu", "psutil", "nvidia-ml-py", "matplotlib")},
            "git_commit": command_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "-C", str(PROJECT_ROOT), "status", "--short"])}


def calibrated_rates(profile):
    # Equal model counts; these isolated capacities are estimates, not measured
    # concurrent capacity. Pilot results and actual queue growth remain visible.
    capacities = {d.value: 1 / statistics.fmean(profile.estimate(m, d) for m in MODEL_REGISTRY)
                  for d in Device}
    cpu, gpu = capacities["cpu"], capacities["gpu"]
    low = 0.5 * min(cpu, gpu)
    medium = max(low * 1.5, 0.85 * max(cpu, gpu))
    high = max(medium * 1.5, 1.25 * (cpu + gpu))
    return {"low": low, "medium": medium, "high": high}, capacities


def mixed_manifest(*, count, pattern, seed, input_ref, rate=None, burst_size=100, burst_gap=1.0):
    base = generate_workload(count=count, models=tuple(MODEL_REGISTRY), input_ref=input_ref,
                             pattern=pattern, seed=seed, rate_jobs_s=rate,
                             burst_size=burst_size, burst_gap_s=burst_gap)
    models = [row.model_id for row in base.items]
    # Separate RNG stream: model order does not alter the arrival schedule.
    random.Random(seed ^ 0x5EED).shuffle(models)
    return replace(base, items=tuple(replace(row, model_id=model) for row, model in zip(base.items, models)))


def execution_order(repeats):
    rows = []
    for repeat in range(repeats):
        patterns = PATTERNS[repeat % 4:] + PATTERNS[:repeat % 4]
        for pattern in patterns:
            shift = (repeat + PATTERNS.index(pattern)) % 4
            policies = POLICIES[shift:] + POLICIES[:shift]
            first = len(rows) + 1
            rows.extend({"index": first + i, "repeat": repeat + 1,
                         "workload": pattern, "scheduler": policy}
                        for i, policy in enumerate(policies))
    return rows


def stop_child(process):
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def run_command(command, log_path, timeout, extra_env=None):
    started = time.monotonic()
    record = {"command": command, "timeout_s": timeout,
              "started_at": datetime.now(timezone.utc).isoformat()}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as stream:
        environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        environment.update(extra_env or {})
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=PROJECT_ROOT, start_new_session=True,
                                   env=environment)
        try:
            record["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record["error"] = "hard process timeout; SIGINT/TERM/KILL cleanup attempted"
            stop_child(process)
            record["returncode"] = process.returncode
        except BaseException:
            stop_child(process)
            raise
    record["wall_s"] = time.monotonic() - started
    return record


def saved_run(parent):
    candidates = sorted(parent.glob("*/summary.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one child run in {parent}, found {len(candidates)}")
    return candidates[0].parent


def validate_run(directory, manifest, policy):
    """Check raw identity/outcomes, not only a child's exit status."""
    import csv
    issues = []
    summary = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    replayed = WorkloadManifest.load(directory / "workload.json")
    if replayed.sha256 != manifest.sha256 or summary.get("manifest_sha256") != manifest.sha256:
        issues.append("manifest hash mismatch")
    if config["scheduler"] != policy:
        issues.append("scheduler mismatch")
    with (directory / "jobs.csv").open() as stream:
        jobs = list(csv.DictReader(stream))
    expected = {item.job_id: item for item in manifest.items}
    if Counter(row["job_id"] for row in jobs) != Counter(expected.keys()):
        issues.append("missing/duplicate/unexpected terminal job IDs")
    for row in jobs:
        item = expected.get(row["job_id"])
        if item is None or row["model_id"] != item.model_id or row["input_ref"] != item.input_ref:
            issues.append("job content mismatch")
            break
    stats = summary["jobs"]
    if (summary["status"] != "passed" or not stats["reconciled"] or stats["failed_jobs"]
            or stats["unfinished_job_ids"] or stats["completed_jobs"] != len(expected)
            or any(row["status"] != "COMPLETED" for row in jobs)):
        issues.append("failed/incomplete/unreconciled workload")
    with (directory / "replay.csv").open() as stream:
        replay = list(csv.DictReader(stream))
    if [row["job_id"] for row in replay] != [item.job_id for item in manifest.items]:
        issues.append("replay order mismatch")
    if any(abs(float(row["planned_offset_s"]) - item.arrival_offset_s) > 1e-6
           for row, item in zip(replay, manifest.items)):
        issues.append("replay arrival mismatch")
    if policy == "resource-aware":
        decisions = [json.loads(line) for line in (directory / "scheduler_decisions.jsonl").read_text().splitlines()]
        if Counter(row["job_id"] for row in decisions) != Counter(expected.keys()):
            issues.append("decision log does not cover each job exactly once")
        devices = {row["job_id"]: row["device"] for row in jobs}
        if any(row["selected_device"] != devices.get(row["job_id"]) for row in decisions):
            issues.append("decision/job device mismatch")
    return issues


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=positive_int, default=1000)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-runs", type=positive_int, default=100)
    parser.add_argument("--profile-repeats", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--monitor-interval", type=positive_float, default=0.1)
    parser.add_argument("--cooldown", type=positive_float, default=2.0)
    parser.add_argument("--burst-size", type=positive_int, default=100)
    parser.add_argument("--input", default="inputs/test.jpg")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--no-gpu-telemetry", action="store_true")
    parser.add_argument("--require-gpu-metrics", action="store_true")
    parser.add_argument("--verify-tests", action="store_true", help="archive full regression suite with required real CUDA/NVML")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "benchmark")
    args = parser.parse_args(argv)
    if not 2 <= args.jobs <= MAX_JOBS or args.repeats < 3:
        parser.error(f"jobs must be 2..{MAX_JOBS} and repeats >= 3")
    if not 0 <= args.seed < 2**64 or args.device_id < 0 or args.gpu_index < 0:
        parser.error("invalid seed/device index")
    if args.no_gpu_telemetry and args.require_gpu_metrics:
        parser.error("conflicting GPU telemetry options")
    if os.environ.get("CUDA_VISIBLE_DEVICES") and not args.no_gpu_telemetry and not args.gpu_uuid:
        parser.error("set --gpu-uuid to map CUDA_VISIBLE_DEVICES, or --no-gpu-telemetry")
    try:
        repository_file(PROJECT_ROOT, args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir = args.output_dir.resolve()
    return args


def run_experiment(args, directory):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(run_type="benchmark", schema_version=1, memory_unit="MB", run_id=directory.name,
                  started_at=datetime.now(timezone.utc).isoformat(), batch_size=1,
                  models=list(MODEL_REGISTRY), model_mix="balanced counts, seeded shuffle; one fixed manifest per workload",
                  prepared_devices=["cpu", "gpu"], timed_tracing=False, live_display=False,
                  hardware=hardware_metadata(), source_sha256=source_hashes(),
                  asset_sha256={spec.filename: sha256_file(PROJECT_ROOT / spec.filename) for spec in MODEL_REGISTRY.values()})
    config["asset_sha256"][args.input] = sha256_file(PROJECT_ROOT / args.input)
    write_json(directory / "config.json", config)
    experiment = {"run_type": "benchmark", "status": "running", "errors": [], "config": config,
                  "schedule": execution_order(args.repeats), "stages": [], "runs": [], "manifests": {}}

    def checkpoint():
        write_json(directory / "experiment.json", experiment)
        write_json(directory / "summary.json", {key: experiment[key] for key in ("status", "errors")})
        mark_latest(directory, "benchmark", experiment["status"])

    checkpoint()
    gpu_flags = ["--device-id", str(args.device_id), "--gpu-index", str(args.gpu_index)]
    if args.gpu_uuid:
        gpu_flags += ["--gpu-uuid", args.gpu_uuid]
    if args.no_gpu_telemetry:
        gpu_flags.append("--no-gpu-telemetry")
    if args.require_gpu_metrics:
        gpu_flags.append("--require-gpu-metrics")
    shared = ["--warmup", str(args.warmup), "--threads", str(args.threads), *gpu_flags]

    def pipeline(policy, manifest_path, parent, timeout, trace=False):
        cmd = [sys.executable, "-m", "edge_scheduler.main", "--scheduler", policy,
               "--manifest", str(manifest_path), "--output-dir", str(parent),
               "--monitor-interval", str(args.monitor_interval), "--timeout", str(timeout),
               "--shutdown-timeout", str(timeout), *shared]
        if policy == "resource-aware":
            cmd += ["--service-profile", str(profile_path)]
        if trace:
            cmd.append("--profile")
        return run_command(cmd, parent / "process.log", timeout * 2 + 120)

    try:
        if args.verify_tests:
            print("Verification: full regression suite, real CUDA/NVML required", flush=True)
            stage = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                                directory / "tests.log", 600, {"T06_TEST_GPU": "1", "T08_TEST_GPU": "1"})
            experiment["stages"].append(dict(stage, name="regression_tests"))
            checkpoint()
            if stage["returncode"] or stage.get("error") or "skipped=" in (directory / "tests.log").read_text():
                raise RuntimeError("regression suite failed or skipped tests; inspect tests.log")
        print("Calibration: fresh processes, untraced timings", flush=True)
        calibration = directory / "calibration"
        stage = run_command([sys.executable, "-m", "edge_scheduler.inference.profiling",
                             "--device", "both", "--runs", str(args.profile_runs),
                             "--repeats", str(args.profile_repeats), "--interval", str(args.monitor_interval),
                             "--input", args.input, "--output-dir", str(calibration), *shared],
                            directory / "calibration.log", 300 * 4 * args.profile_repeats + 60)
        experiment["stages"].append(dict(stage, name="calibration"))
        checkpoint()
        if stage["returncode"] or stage.get("error"):
            raise RuntimeError("calibration failed; see calibration.log and calibration/*/summary.json")
        profile_path = saved_run(calibration) / "profile.json"
        profile = ServiceTimeProfile.from_file(profile_path)
        rates, capacities = calibrated_rates(profile)
        experiment.update(profile_path=str(profile_path.relative_to(directory)),
                          profile_sha256=sha256_file(profile_path), estimated_capacities_jobs_s=capacities,
                          rates_jobs_s=rates)
        manifests = directory / "manifests"
        manifests.mkdir()
        gap = args.burst_size / (0.65 * sum(capacities.values()))
        for index, pattern in enumerate(PATTERNS):
            manifest = mixed_manifest(count=args.jobs, pattern=pattern, seed=(args.seed + index) % 2**64,
                                      input_ref=args.input, rate=rates.get(pattern),
                                      burst_size=args.burst_size, burst_gap=gap)
            manifest.save(manifests / f"{pattern}.json")
            experiment["manifests"][pattern] = {"path": f"manifests/{pattern}.json", "sha256": manifest.sha256,
                "jobs": len(manifest.items), "model_counts": dict(Counter(x.model_id for x in manifest.items)),
                "last_arrival_s": manifest.items[-1].arrival_offset_s,
                "burst_size": args.burst_size if pattern == "burst" else None,
                "burst_gap_s": gap if pattern == "burst" else None}
        pilot = mixed_manifest(count=12, pattern="burst", seed=args.seed,
                               input_ref=args.input, burst_size=4, burst_gap=0.05)
        pilot_path = manifests / "pilot.json"
        pilot.save(pilot_path)
        checkpoint()
        # Trace evidence is kept separate from all four untraced policy pilots.
        for policy, trace in [("gpu-only", True), *[(p, False) for p in POLICIES]]:
            name = "cuda_verification" if trace else f"pilot_{policy}"
            print(f"Preflight: {name}", flush=True)
            parent = directory / "preflight" / name
            stage = pipeline(policy, pilot_path, parent, 60, trace)
            experiment["stages"].append(dict(stage, name=name))
            child = saved_run(parent)
            stage["validation_errors"] = validate_run(child, pilot, policy)
            experiment["stages"][-1].update(stage)
            checkpoint()
            if stage["returncode"] or stage.get("error") or stage["validation_errors"]:
                raise RuntimeError(f"preflight failed: {name}; inspect {parent}")
        cpu_mean = 1 / capacities["cpu"]
        for plan in experiment["schedule"]:
            pattern, policy = plan["workload"], plan["scheduler"]
            manifest = WorkloadManifest.load(manifests / f"{pattern}.json")
            timeout = max(60, manifest.items[-1].arrival_offset_s + args.jobs * cpu_mean * 5)
            parent = directory / "runs" / pattern / f"repeat-{plan['repeat']}" / policy
            print(f"[{plan['index']:02d}/{len(experiment['schedule'])}] {pattern} / {policy} / repeat {plan['repeat']}", flush=True)
            result = dict(plan, status="running", output_parent=str(parent.relative_to(directory)))
            experiment["runs"].append(result)
            checkpoint()
            result["gpu_before"] = command_output(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
                                                    "--format=csv,noheader,nounits"])
            result.update(pipeline(policy, manifests / f"{pattern}.json", parent, timeout))
            result["gpu_after"] = command_output(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
                                                   "--format=csv,noheader,nounits"])
            try:
                child = saved_run(parent)
                result["run_path"] = str(child.relative_to(directory))
                result["validation_errors"] = validate_run(child, manifest, policy)
            except (OSError, ValueError, KeyError) as exc:
                result["validation_errors"] = [f"{type(exc).__name__}: {exc}"]
            result["status"] = "passed" if (result["returncode"] == 0 and not result.get("error")
                                                      and not result["validation_errors"]) else "failed"
            if result["status"] == "failed":
                experiment["errors"].append(f"run {plan['index']} failed: {pattern}/{policy}/repeat {plan['repeat']}")
            checkpoint()
            if plan["index"] < len(experiment["schedule"]):
                time.sleep(args.cooldown)
        if source_hashes() != config["source_sha256"]:
            experiment["errors"].append("source files changed during experiment")
        if sha256_file(profile_path) != experiment["profile_sha256"]:
            experiment["errors"].append("service-time profile changed during experiment")
        if any(sha256_file(PROJECT_ROOT / path) != digest for path, digest in config["asset_sha256"].items()):
            experiment["errors"].append("model/input assets changed during experiment")
        experiment["status"] = "failed" if experiment["errors"] else "passed"
    except KeyboardInterrupt:
        experiment["status"] = "interrupted"
        experiment["errors"].append("KeyboardInterrupt; current child stopped, partial artifacts retained")
    except Exception as exc:
        experiment["status"] = "failed"
        experiment["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        experiment["finished_at"] = datetime.now(timezone.utc).isoformat()
        checkpoint()
    return experiment


def main(argv=None):
    args = parse_args(argv)
    directory = new_run(args.output_dir)
    print(f"Benchmark experiment: {directory}", flush=True)
    experiment = run_experiment(args, directory)
    from .benchmark_report import generate_benchmark_report
    try:
        report = generate_benchmark_report(directory)
        print(f"Report: {report}", flush=True)
    except Exception as exc:
        print(f"Reporting failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{experiment['status'].upper()}: {len(experiment['runs'])}/{len(experiment['schedule'])} runs", flush=True)
    for error in experiment["errors"]:
        print(error, file=sys.stderr)
    return 0 if experiment["status"] == "passed" else 130 if experiment["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
