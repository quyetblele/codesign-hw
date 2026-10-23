"""Run a finite batch through the controller, real ONNX backends and monitor.

Run from the project root with PYTHONPATH=src python -m edge_scheduler.main.
Both backends are prepared, including for CPU-only/GPU-only scheduling.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import math
import os
from pathlib import Path
import platform
import sys
from threading import Lock
import time
from uuid import uuid4

from .core.models import Device, Job
from .inference.cpu_backend import CPUBackend
from .inference.gpu_backend import GPUBackend
from .inference.model_registry import MODEL_REGISTRY, PROJECT_ROOT
from .inference.verification import summarize_profile
from .monitoring.gpu_probe import GPUProbe
from .monitoring.monitoring_models import WorkerRegistry
from .monitoring.resource_log import ResourceCSVLogger
from .monitoring.resource_monitor import ResourceMonitor
from .runtime.backend_adapter import MonitoredBackend
from .runtime.controller import Controller
from .runtime.run_results import summarize_jobs, summarize_resources, write_job_csv, write_json
from .runtime.schedulers import CpuOnlyScheduler, GpuOnlyScheduler, RoundRobinScheduler


class Phase:
    def __init__(self):
        self._value = "initializing"
        self._lock = Lock()

    def set(self, value: str) -> None:
        with self._lock:
            self._value = value

    def get(self) -> str:
        with self._lock:
            return self._value


@dataclass
class Pipeline:
    controller: Controller
    monitor: ResourceMonitor
    logger: ResourceCSVLogger
    backends: dict[Device, MonitoredBackend]
    phase: Phase


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=("cpu-only", "gpu-only", "round-robin"),
                        default="round-robin")
    parser.add_argument("--jobs", type=_positive_integer, default=20, help="total jobs in one batch")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    parser.add_argument("--input", default="inputs/test.jpg", help="image path relative to project root")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "pipeline")
    parser.add_argument("--warmup", type=_positive_integer, default=5)
    parser.add_argument("--threads", type=_positive_integer, default=2, help="ORT intra-op threads/session")
    parser.add_argument("--monitor-interval", type=_positive_seconds, default=0.2)
    parser.add_argument("--max-sample-age", type=_positive_seconds, default=1.0)
    parser.add_argument("--timeout", type=_positive_seconds, default=60.0, help="workload wait timeout")
    parser.add_argument("--shutdown-timeout", type=_positive_seconds, default=60.0,
                        help="timeout per drain attempt; timed-out cleanup is retried without cancelling inference")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVML device index")
    parser.add_argument("--gpu-uuid", help="explicit NVML UUID, overriding GPU index")
    parser.add_argument("--no-gpu-telemetry", action="store_true", help="disable NVML only; GPU inference is still required")
    parser.add_argument("--require-gpu-metrics", action="store_true", help="require VRAM telemetry; utilization remains optional")
    parser.add_argument("--profile", action="store_true", help="save ORT traces and require CUDA compute for both models")
    args = parser.parse_args(argv)
    if args.device_id < 0 or args.gpu_index < 0:
        parser.error("device-id and gpu-index must be >= 0")
    if not args.input.strip() or Path(args.input).is_absolute():
        parser.error("input must be a non-empty project-relative path")
    if args.no_gpu_telemetry and args.require_gpu_metrics:
        parser.error("no-gpu-telemetry conflicts with require-gpu-metrics")
    if os.environ.get("CUDA_VISIBLE_DEVICES") and not args.no_gpu_telemetry and not args.gpu_uuid:
        parser.error("set --gpu-uuid to map NVML to CUDA_VISIBLE_DEVICES, or disable GPU telemetry")
    args.root = args.root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def create_run_directory(args) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    directory = args.output_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(run_id=run_id, workload="finite batch, alternating selected models",
                  prepared_devices=[device.value for device in Device], clock="time.monotonic",
                  memory_unit="MiB", resource_scope="OS-visible CPU/RAM; whole-process RSS; whole selected GPU",
                  timing_scope="job timestamps exclude prepare/warmup; execution includes preprocessing, transfer and validation",
                  python=sys.version, platform=platform.platform(),
                  packages={name: version(name) for name in ("numpy", "onnxruntime-gpu", "psutil", "nvidia-ml-py")})
    write_json(directory / "config.json", config)
    return directory


def build_pipeline(args, directory: Path) -> Pipeline:
    registry, phase = WorkerRegistry(), Phase()
    options = {"root": args.root, "warmup_runs": args.warmup,
               "intra_op_num_threads": args.threads, "device_id": args.device_id,
               "profile_dir": directory / "profiles" if args.profile else None}
    backends = {device: MonitoredBackend(cls(**options), registry)
                for device, cls in ((Device.CPU, CPUBackend), (Device.GPU, GPUBackend))}
    policies = {"cpu-only": CpuOnlyScheduler, "gpu-only": GpuOnlyScheduler,
                "round-robin": RoundRobinScheduler}
    controller = Controller(policies[args.scheduler](), backends)
    logger = ResourceCSVLogger(directory / "resources.csv")
    try:
        monitor = ResourceMonitor(
            interval_s=args.monitor_interval, max_age_s=args.max_sample_age,
            gpu_probe=GPUProbe(device_index=args.gpu_index, device_uuid=args.gpu_uuid,
                               enabled=not args.no_gpu_telemetry),
            worker_state_provider=registry.snapshot,
            queue_lengths_provider=controller.queues.queue_lengths,
            phase_provider=phase.get, sink=logger,
        )
    except BaseException:
        logger.close()
        raise
    controller.monitor = monitor
    return Pipeline(controller, monitor, logger, backends, phase)


def create_jobs(args) -> list[Job]:
    return [Job(f"job-{index + 1:06d}", args.models[index % len(args.models)], args.input)
            for index in range(args.jobs)]


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def write_results(directory: Path, pipeline: Pipeline | None, report: dict) -> None:
    jobs = list(pipeline.controller.jobs.values()) if pipeline else []
    accepted, terminal = pipeline.controller.queues.reconcile() if pipeline else (0, 0)
    report["jobs"] = summarize_jobs(jobs, accepted, terminal)
    if pipeline:
        report["controller"] = pipeline.controller.diagnostics
        report["monitor"] = pipeline.monitor.diagnostics
        report["logger"] = pipeline.logger.stats
    write_job_csv(directory / "jobs.csv", jobs)
    write_json(directory / "summary.json", report)


def run_pipeline(args, directory: Path) -> dict:
    report = {"status": "running", "run_directory": str(directory),
              "requested_jobs": args.jobs, "errors": [], "interrupted": False}
    pipeline = None
    try:
        pipeline = build_pipeline(args, directory)
        pipeline.phase.set("prepare")
        pipeline.monitor.start()
        if not pipeline.monitor.wait_for_sample(max(5.0, args.monitor_interval * 3)):
            raise RuntimeError(f"monitor produced no sample: {pipeline.monitor.diagnostics}")
        started = time.monotonic()
        pipeline.controller.start()
        report["prepare_s"] = time.monotonic() - started
        report["session_providers"] = {
            device.value: getattr(adapter.backend, "session_providers", {})
            for device, adapter in pipeline.backends.items()
        }
        if args.require_gpu_metrics:
            sample = pipeline.monitor.read_sample()
            if sample.values["gpu_memory_total_mb"] is None:
                raise RuntimeError(f"VRAM telemetry required: {sample.errors}")
        pipeline.phase.set("workload")
        for job in create_jobs(args):
            pipeline.controller.submit(job)
        if not pipeline.controller.wait_all(args.timeout):
            raise TimeoutError("workload timeout; draining accepted jobs before releasing resources")
    except KeyboardInterrupt:
        report["interrupted"] = True
        report["errors"].append("KeyboardInterrupt: stop requested")
    except Exception as exc:
        report["errors"].append(_error(exc))
    finally:
        if pipeline is not None:
            pipeline.phase.set("drain")
            pipeline.controller.request_shutdown()
            while True:
                try:
                    pipeline.controller.close(args.shutdown_timeout)
                    break
                except KeyboardInterrupt:
                    report["interrupted"] = True
                    report["errors"].append("KeyboardInterrupt: continuing graceful drain")
                except TimeoutError as exc:
                    message = _error(exc)
                    if message not in report["errors"]:
                        report["errors"].append(message)
                    report["status"] = "draining"
                    try:
                        write_results(directory, pipeline, report)
                    except Exception as write_error:
                        print(f"Could not save drain checkpoint: {_error(write_error)}", file=sys.stderr)
                    print("Shutdown timed out; saved unfinished jobs and retrying drain. "
                          "Native inference is not cancelled.", file=sys.stderr)
                except Exception as exc:
                    report["errors"].append(_error(exc))
                    break
            pipeline.phase.set("shutdown")
            for component in (pipeline.monitor, pipeline.logger):
                try:
                    component.close()
                except Exception as exc:
                    report["errors"].append(_error(exc))
            if args.profile:
                try:
                    report["profiles"] = {}
                    for device, adapter in pipeline.backends.items():
                        paths = getattr(adapter.backend, "profile_paths", {})
                        report["profiles"][device.value] = {
                            model: {"path": str(Path(path).relative_to(directory)),
                                    **summarize_profile(path)} for model, path in paths.items()
                        }
                    for model in MODEL_REGISTRY:
                        if not report["profiles"]["gpu"].get(model, {}).get("cuda_compute_verified"):
                            raise RuntimeError(f"no CUDA compute trace for {model}")
                except Exception as exc:
                    report["errors"].append(_error(exc))
            try:
                report["resources"] = summarize_resources(directory / "resources.csv")
                report["resource_metadata"] = {"system": pipeline.monitor.system_probe.metadata,
                                               "gpu": pipeline.monitor.gpu_probe.metadata}
                if pipeline.monitor.diagnostics["sampler_error"] or pipeline.monitor.diagnostics["sink_error"]:
                    report["errors"].append("resource sampler or sink failed")
                if pipeline.logger.stats["error"] or pipeline.logger.stats["dropped_samples"]:
                    report["errors"].append("resource writer failed or dropped samples")
                for metric in ("cpu_load_pct", "ram_free_mb", "process_rss_mb"):
                    if not report["resources"]["metrics"].get(metric, {}).get("valid_samples"):
                        report["errors"].append(f"no valid resource samples: {metric}")
            except Exception as exc:
                report["errors"].append(_error(exc))
        write_results(directory, pipeline, report)
        stats = report["jobs"]
        if not stats["reconciled"] or stats["failed_jobs"]:
            report["errors"].append("jobs failed or terminal reconciliation failed")
        if not report["interrupted"] and stats["accepted_jobs"] != args.jobs:
            report["errors"].append("not all requested jobs were accepted")
        report["status"] = "interrupted" if report["interrupted"] else "failed" if report["errors"] else "passed"
        write_json(directory / "summary.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        directory = create_run_directory(args)
        print(f"Pipeline run: {directory}", flush=True)
        report = run_pipeline(args, directory)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(_error(exc), file=sys.stderr)
        return 1
    stats = report["jobs"]
    print(f"{report['status'].upper()}: accepted={stats['accepted_jobs']} "
          f"completed={stats['completed_jobs']} failed={stats['failed_jobs']} "
          f"unfinished={len(stats['unfinished_job_ids'])}")
    for message in report["errors"]:
        print(message, file=sys.stderr)
    return 130 if report["interrupted"] else 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
