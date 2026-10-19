"""T08 acceptance harness: idle telemetry, real T06 workload and paired overhead.

This adapter publishes worker facts, not Job state or a T07 queue/controller.
GPU inference errors fail the run; unsupported GPU telemetry stays unavailable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import statistics
import sys
from threading import Event, Lock
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_scheduler.gpu_probe import GPUProbe
from edge_scheduler.model_registry import MODEL_REGISTRY, PROJECT_ROOT
from edge_scheduler.models import Device
from edge_scheduler.monitoring_models import METRICS, WorkerRegistry
from edge_scheduler.resource_log import ResourceCSVLogger
from edge_scheduler.resource_monitor import ResourceMonitor


class Phase:
    def __init__(self):
        self._lock = Lock()
        self._value = "idle"

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


def summarize_csv(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row[metric] != ""]
        reasons = {}
        for row in rows:
            reason = json.loads(row["errors_json"]).get(metric)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        metrics[metric] = {
            "valid_samples": len(values), "unavailable_samples": len(rows) - len(values),
            "mean": statistics.fmean(values) if values else None,
            "observed_max": max(values) if values else None, "errors": reasons,
        }
    return {"samples": len(rows), "metrics": metrics,
            "max_sample_duration_s": max((float(row["sample_duration_s"]) for row in rows), default=None)}


def monitored(args, directory, registry, phase, callback):
    logger = ResourceCSVLogger(directory / "resources.csv")
    gpu = GPUProbe(device_index=args.gpu_index, device_uuid=args.gpu_uuid, enabled=not args.no_gpu)
    monitor = ResourceMonitor(interval_s=args.interval, max_age_s=args.max_age,
                              gpu_probe=gpu, worker_state_provider=registry.snapshot,
                              phase_provider=phase.get, sink=logger)
    errors = []
    result = None
    try:
        monitor.start()
        if not monitor.wait_for_sample(max(5.0, args.interval * 3)):
            raise RuntimeError(f"monitor did not produce samples: {monitor.diagnostics}")
        result = callback()
        phase.set("cooldown")
        Event().wait(args.interval * 1.1)
    finally:
        for component in (monitor, logger):
            try:
                component.close()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        summary = summarize_csv(directory / "resources.csv")
        summary.update(logger=logger.stats, monitor=monitor.diagnostics,
                       system_metadata=monitor.system_probe.metadata, gpu_metadata=gpu.metadata,
                       cleanup_errors=errors)
        (directory / "monitor_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or summary["monitor"]["sampler_error"] or summary["monitor"]["sink_error"]:
        raise RuntimeError(f"monitor/log failure: {summary}")
    if summary["logger"]["dropped_samples"]:
        raise RuntimeError("resource samples were dropped; see monitor_summary.json")
    for metric in ("cpu_load_pct", "ram_free_mb", "process_rss_mb"):
        if not summary["metrics"][metric]["valid_samples"]:
            raise RuntimeError(f"no valid {metric} samples")
    if args.require_gpu_metrics and not summary["metrics"]["gpu_memory_total_mb"]["valid_samples"]:
        raise RuntimeError("GPU memory telemetry required but unavailable; see monitor_summary.json")
    return result, summary


def jobs(args, backends, registry, phase):
    phase.set("inference")

    def execute(backend):
        for _ in range(args.runs):
            for model in MODEL_REGISTRY:
                registry.set_running(backend.device, model, time.monotonic())
                try:
                    backend.infer(model, args.input)
                finally:
                    registry.set_ready(backend.device)
        return args.runs * len(MODEL_REGISTRY)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(backends)) as pool:
        futures = [pool.submit(execute, backend) for backend in backends.values()]
        count = sum(future.result() for future in futures)
    return {"completed_jobs": count, "elapsed_s": time.monotonic() - started}


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=positive, default=10.0)
    parser.add_argument("--interval", type=positive, default=0.2)
    parser.add_argument("--max-age", type=positive, default=1.0)
    parser.add_argument("--workload", choices=("none", "cpu", "gpu", "both"), default="none")
    parser.add_argument("--runs", type=int, default=100, help="jobs/model/device")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device-id", type=int, default=0, help="CUDA backend ordinal")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVML device index")
    parser.add_argument("--gpu-uuid", help="explicit NVML GPU UUID, overrides NVML index")
    parser.add_argument("--no-gpu", action="store_true", help="disable GPU telemetry only")
    parser.add_argument("--require-gpu-metrics", action="store_true", help="require VRAM; utilization may be unavailable")
    parser.add_argument("--overhead-repeats", type=int, default=0, help="paired, alternating monitored/baseline passes")
    parser.add_argument("--input", default="inputs/test.jpg")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "t08")
    args = parser.parse_args()
    if min(args.runs, args.warmup, args.threads) < 1 or min(args.device_id, args.gpu_index, args.overhead_repeats) < 0:
        parser.error("runs/warmup/threads must be >= 1; device indexes/repeats must be >= 0")
    if args.no_gpu and args.require_gpu_metrics:
        parser.error("--no-gpu conflicts with --require-gpu-metrics")
    if args.overhead_repeats and args.workload == "none":
        parser.error("overhead measurement needs a workload")
    if args.workload in ("gpu", "both") and os.environ.get("CUDA_VISIBLE_DEVICES") and not args.gpu_uuid and not args.no_gpu:
        parser.error("set --gpu-uuid to map telemetry explicitly when CUDA_VISIBLE_DEVICES is set")
    directory = args.output_dir.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(memory_unit="MiB", clock="time.monotonic", rss_scope="entire application process",
                  gpu_scope="whole selected GPU, including other applications",
                  packages=dict(sorted((d.metadata["Name"], d.version) for d in distributions())))
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    registry, phase, backends = WorkerRegistry(), Phase(), {}
    report = {"status": "running", "run_directory": str(directory), "overhead_pairs": []}
    cleanup_errors = []
    try:
        def prepare_and_run():
            if args.workload == "none":
                Event().wait(args.duration)
                return {"completed_jobs": 0}
            from edge_scheduler.cpu_backend import CPUBackend
            from edge_scheduler.gpu_backend import GPUBackend
            devices = ("cpu", "gpu") if args.workload == "both" else (args.workload,)
            for name in devices:
                phase.set("prepare_" + name)
                cls = CPUBackend if name == "cpu" else GPUBackend
                backend = cls(root=args.root, warmup_runs=args.warmup,
                              intra_op_num_threads=args.threads, device_id=args.device_id)
                backends[name] = backend
                backend.prepare()
                registry.set_ready(backend.device)
            return jobs(args, backends, registry, phase)

        report["workload"], report["monitor"] = monitored(
            args, directory, registry, phase, prepare_and_run)
        for iteration in range(args.overhead_repeats):
            pair = {"iteration": iteration, "order": ["baseline", "monitored"] if iteration % 2 == 0 else ["monitored", "baseline"]}
            for mode in pair["order"]:
                if mode == "baseline":
                    pair[mode] = jobs(args, backends, registry, phase)
                else:
                    subdirectory = directory / "overhead" / str(iteration)
                    subdirectory.mkdir(parents=True)
                    pair[mode], _ = monitored(args, subdirectory, registry, phase,
                                             lambda: jobs(args, backends, registry, phase))
            pair["relative_elapsed_change_pct"] = (
                pair["monitored"]["elapsed_s"] / pair["baseline"]["elapsed_s"] - 1) * 100
            report["overhead_pairs"].append(pair)
        if report["overhead_pairs"]:
            report["median_relative_elapsed_change_pct"] = statistics.median(
                pair["relative_elapsed_change_pct"] for pair in report["overhead_pairs"])
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr)
    finally:
        phase.set("shutdown")
        for backend in backends.values():
            registry.set_unavailable(backend.device)
            try:
                backend.close()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"T08 {report['status'].upper()}: {directory}")
    if "monitor" in report:
        print(f"  samples={report['monitor']['samples']}; workload={report['workload']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
