"""T11: isolated, repeatable model/device calibration with raw measurements."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from ..artifacts import mark_latest, new_run
from ..monitoring.monitoring_models import MEMORY_UNIT
from .model_registry import MODEL_REGISTRY, PROJECT_ROOT
from ..runtime.run_results import distribution, summarize_resources, write_json


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return number


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fresh_sample(monitor, timeout=10.0):
    previous = monitor.read_sample().sequence
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = monitor.read_sample()
        if sample.sequence > previous and not sample.stale:
            return sample
        if monitor.diagnostics["sampler_error"]:
            raise RuntimeError(monitor.diagnostics["sampler_error"])
        time.sleep(0.01)
    raise TimeoutError("no fresh resource sample")


def run_trial(config: dict, directory: Path) -> int:
    # Called only by a fresh subprocess: one selected model and one device.
    from .cpu_backend import CPUBackend
    from .gpu_backend import GPUBackend
    from .verification import summarize_profile
    from ..main import Phase
    from ..monitoring.gpu_probe import GPUProbe
    from ..monitoring.resource_log import ResourceCSVLogger
    from ..monitoring.resource_monitor import ResourceMonitor

    model, device = config["model_id"], config["device"]
    result = {"model_id": model, "device": device, "repeat": config["repeat"],
              "status": "running", "errors": [], "memory_unit": MEMORY_UNIT}
    phase = Phase()
    logger = ResourceCSVLogger(directory / "resources.csv")
    monitor = ResourceMonitor(interval_s=config["interval"], sink=logger, phase_provider=phase.get,
                              gpu_probe=GPUProbe(enabled=device == "gpu" and not config["no_gpu_telemetry"],
                                                 device_index=config["gpu_index"], device_uuid=config["gpu_uuid"]))
    backend = None
    try:
        phase.set("baseline")
        monitor.start()
        if not monitor.wait_for_sample(10):
            raise RuntimeError("monitor produced no baseline sample")
        baseline = fresh_sample(monitor)
        for key in ("process_rss_mb", "gpu_memory_used_mb"):
            result["baseline_" + key] = baseline.values[key]
        if config["require_gpu_metrics"] and device == "gpu" and baseline.values["gpu_memory_used_mb"] is None:
            raise RuntimeError(f"GPU memory required: {dict(baseline.errors)}")
        backend_type = CPUBackend if device == "cpu" else GPUBackend
        backend = backend_type(root=config["root"], model_ids=(model,), warmup_runs=config["warmup"],
                               intra_op_num_threads=config["threads"], device_id=config["device_id"],
                               profile_dir=directory / "traces" if config["verify_cuda"] else None)
        phase.set("prepare_warmup")
        started = time.monotonic()
        backend.prepare()
        result["prepare_s"] = time.monotonic() - started
        result["session_providers"] = backend.session_providers
        phase.set("ready")
        ready = fresh_sample(monitor)
        for key in ("process_rss_mb", "gpu_memory_used_mb"):
            value, before = ready.values[key], baseline.values[key]
            result["ready_" + key] = value
            result["delta_" + key] = value - before if value is not None and before is not None else None
        phase.set("inference")
        with (directory / "measurements.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MEASUREMENT_FIELDS)
            writer.writeheader()
            for index in range(config["runs"]):
                started = time.monotonic()
                output = backend.infer(model, config["input"])
                finished = time.monotonic()
                writer.writerow({"model_id": model, "device": device, "repeat": config["repeat"],
                                 "iteration": index, "start_time_s": started, "finish_time_s": finished,
                                 "execution_ms": (finished - started) * 1000,
                                 "top1_class": int(output[0].argmax())})
            stream.flush()
        # Allow a final sampled observation, outside the timed inference calls.
        fresh_sample(monitor)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["errors"].append("KeyboardInterrupt")
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        phase.set("shutdown")
        for component in (backend, monitor, logger):
            if component is not None:
                try:
                    component.close()
                except Exception as exc:
                    result["errors"].append(f"cleanup {type(exc).__name__}: {exc}")
        if backend is not None and config["verify_cuda"]:
            try:
                result["traces"] = {name: summarize_profile(path) for name, path in backend.profile_paths.items()}
                if device == "gpu" and not result["traces"].get(model, {}).get("cuda_compute_verified"):
                    raise RuntimeError("CUDA compute trace not verified")
            except Exception as exc:
                result["errors"].append(str(exc))
        result["monitor"], result["logger"] = monitor.diagnostics, logger.stats
        result["resource_metadata"] = {"system": monitor.system_probe.metadata, "gpu": monitor.gpu_probe.metadata}
        result["resources"] = summarize_resources(directory / "resources.csv")
        if any((monitor.diagnostics["sampler_error"], monitor.diagnostics["sink_error"],
                logger.stats["error"], logger.stats["dropped_samples"])):
            result["errors"].append("resource logging failed or dropped samples")
        if not result["resources"]["metrics"].get("process_rss_mb", {}).get("valid_samples"):
            result["errors"].append("no valid process RSS samples")
        if result["errors"] and result["status"] != "interrupted":
            result["status"] = "failed"
        write_json(directory / "trial.json", result)
    return 0 if result["status"] == "passed" else 130 if result["status"] == "interrupted" else 1


MEASUREMENT_FIELDS = ("model_id", "device", "repeat", "iteration", "start_time_s", "finish_time_s",
                      "execution_ms", "top1_class")


def aggregate(directory: Path, config: dict, status: str, errors: list[str]) -> dict:
    measurements, resources, trials = [], [], []
    resource_fields = None
    for trial_dir in sorted((directory / "trials").iterdir()):
        trial_config = json.loads((trial_dir / "config.json").read_text())
        if (trial_dir / "trial.json").exists():
            trials.append(json.loads((trial_dir / "trial.json").read_text()))
        for filename, destination in (("measurements.csv", measurements), ("resources.csv", resources)):
            path = trial_dir / filename
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if filename == "resources.csv":
                    resource_fields = ["model_id", "device", "repeat"] + list(reader.fieldnames or [])
                for row in reader:
                    if filename == "resources.csv":
                        row.update({key: trial_config[key] for key in ("model_id", "device", "repeat")})
                    destination.append(row)
    write_csv(directory / "measurements.csv", measurements, MEASUREMENT_FIELDS)
    if resource_fields:
        write_csv(directory / "resources.csv", resources, resource_fields)
    else:
        from ..monitoring.resource_log import FIELDS
        write_csv(directory / "resources.csv", [], ["model_id", "device", "repeat", *FIELDS])
    rows = []
    for model in config["models"]:
        for device in config["devices"]:
            matching = [trial for trial in trials if trial["model_id"] == model and trial["device"] == device]
            passed = {trial["repeat"] for trial in matching if trial["status"] == "passed"}
            values = [float(row["execution_ms"]) for row in measurements
                      if row["model_id"] == model and row["device"] == device and int(row["repeat"]) in passed]
            stats = distribution(values)
            means = [statistics.fmean(float(row["execution_ms"]) for row in measurements
                                     if row["model_id"] == model and row["device"] == device and int(row["repeat"]) == repeat)
                     for repeat in sorted(passed)]
            row = {"model_id": model, "device": device, "samples": len(values), "repeats": len(passed),
                   "status": "passed" if len(passed) == config["repeats"] and len(values) == config["runs"] * config["repeats"] else "incomplete",
                   "mean_ms": stats["mean"], "p50_ms": stats["p50"], "p95_ms": stats["p95"], "p99_ms": stats["p99"],
                   "repeat_mean_std_ms": statistics.pstdev(means) if means else None,
                   "mean_service_s": stats["mean"] / 1000 if stats["mean"] is not None else None}
            for key in ("baseline_process_rss_mb", "ready_process_rss_mb", "delta_process_rss_mb",
                        "baseline_gpu_memory_used_mb", "ready_gpu_memory_used_mb", "delta_gpu_memory_used_mb"):
                observed = [trial[key] for trial in matching if trial.get(key) is not None and trial["status"] == "passed"]
                row[key] = statistics.median(observed) if observed else None
            for metric in ("process_rss_mb", "gpu_memory_used_mb"):
                observed = [float(sample[metric]) for sample in resources if sample["model_id"] == model
                            and sample["device"] == device and int(sample["repeat"]) in passed
                            and sample["phase"] == "inference" and sample[metric] != ""]
                row["observed_max_" + metric] = max(observed) if observed else None
            rows.append(row)
    write_csv(directory / "profile_summary.csv", rows, list(rows[0]))
    if status == "passed" and any(row["status"] != "passed" for row in rows):
        status = "failed"
        errors.append("incomplete model/device measurements")
    report = {"schema_version": 2, "run_type": "profiling", "run_id": directory.name,
              "memory_unit": config["memory_unit"],
              "status": status, "errors": errors, "profiles": rows, "trials": trials,
              "timing_scope": config["timing_scope"], "memory_scope": config["memory_scope"],
              "trace_enabled": config["verify_cuda"]}
    write_json(directory / "profile.json", {key: value for key, value in report.items() if key != "trials"})
    write_json(directory / "summary.json", report)
    mark_latest(directory, "profiling", status)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    parser.add_argument("--runs", type=positive_int, default=100)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--interval", type=positive_seconds, default=0.2)
    parser.add_argument("--trial-timeout", type=positive_seconds, default=300)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--no-gpu-telemetry", action="store_true")
    parser.add_argument("--require-gpu-metrics", action="store_true")
    parser.add_argument("--verify-cuda", action="store_true", help="enable ORT traces; adds timing overhead")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", default="inputs/test.jpg")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "profiling")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config:
        return run_trial(json.loads(args.worker_config.read_text()), args.worker_config.parent)
    if args.device_id < 0 or args.gpu_index < 0:
        parser.error("device-id and gpu-index must be >= 0")
    if len(set(args.models)) != len(args.models):
        parser.error("models must be distinct")
    if Path(args.input).is_absolute() or not args.input.strip():
        parser.error("input must be a nonempty project-relative image path")
    if args.require_gpu_metrics and (args.device == "cpu" or args.no_gpu_telemetry):
        parser.error("require-gpu-metrics requires GPU with telemetry enabled")
    if args.device != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") and not args.gpu_uuid and not args.no_gpu_telemetry:
        parser.error("set --gpu-uuid to map CUDA_VISIBLE_DEVICES, or disable GPU telemetry")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(run_type="profiling", schema_version=2, started_at=datetime.now(timezone.utc).isoformat(),
                  devices=["cpu", "gpu"] if args.device == "both" else [args.device],
                  timing_scope="image load + preprocessing + synchronous ONNX run + validation; excludes prepare/warmup, sampling waits and file I/O",
                  memory_scope="one model/device per fresh process; RSS whole process; VRAM whole GPU; baseline deltas are signed estimates; maxima are sampled observations, not absolute peaks",
                  memory_unit=MEMORY_UNIT, clock="time.monotonic", python=sys.version, platform=platform.platform(),
                  cpu=platform.processor(), logical_cpus=os.cpu_count(), batch_size=1,
                  packages={name: version(name) for name in ("numpy", "onnxruntime-gpu", "psutil", "nvidia-ml-py")})
    directory = new_run(args.output_dir)
    config["run_id"] = directory.name
    write_json(directory / "config.json", config)
    (directory / "trials").mkdir()
    mark_latest(directory, "profiling", "running")
    print(f"Profiling run: {directory}", flush=True)
    status, errors = "passed", []
    try:
        for repeat in range(args.repeats):
            # Alternate order across repeats to reduce fixed order effects.
            pairs = [(model, device) for model in args.models for device in config["devices"]]
            for model, device in pairs if repeat % 2 == 0 else reversed(pairs):
                trial_dir = directory / "trials" / f"{model}_{device}_{repeat + 1}"
                trial_dir.mkdir()
                trial_config = dict(config, model_id=model, device=device, repeat=repeat + 1)
                write_json(trial_dir / "config.json", trial_config)
                print(f"  {model}/{device}, repeat {repeat + 1}/{args.repeats}", flush=True)
                result = subprocess.run([sys.executable, "-m", "edge_scheduler.inference.profiling",
                                         "--worker-config", str(trial_dir / "config.json")],
                                        capture_output=True, text=True, timeout=args.trial_timeout,
                                        env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src")))
                if result.returncode:
                    status = "failed"
                    detail = result.stderr.strip() or result.stdout.strip()
                    if (trial_dir / "trial.json").exists():
                        detail = "; ".join(json.loads((trial_dir / "trial.json").read_text())["errors"])
                    errors.append(f"{model}/{device} repeat {repeat + 1}: {detail or 'worker failed'}")
    except KeyboardInterrupt:
        status = "interrupted"
        errors.append("KeyboardInterrupt: profiling stopped")
    except Exception as exc:
        status = "failed"
        errors.append(f"{type(exc).__name__}: {exc}")
    report = aggregate(directory, config, status, errors)
    print(f"Model        Device  Samples  Mean ms   P50 ms   P95 ms   RSS delta {MEMORY_UNIT}  VRAM delta {MEMORY_UNIT}")
    for row in report["profiles"]:
        def show(key):
            return "—" if row[key] is None else f"{row[key]:.3f}"
        print(f"{row['model_id']:12} {row['device']:6} {row['samples']:7}  {show('mean_ms'):>7}  {show('p50_ms'):>7}  {show('p95_ms'):>7}  {show('delta_process_rss_mb'):>13}  {show('delta_gpu_memory_used_mb'):>14}")
    print(f"{report['status'].upper()}: {directory}")
    for error in errors:
        print(error, file=sys.stderr)
    return 0 if report["status"] == "passed" else 130 if report["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
