"""Run real T06 backends and save a reproducible acceptance bundle.

GPU runs never skip or substitute CPU. --profile additionally requires CUDA
compute evidence for each model. This is not the T07 queued worker pipeline.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import platform
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import onnxruntime as ort

from edge_scheduler.inference.cpu_backend import CPUBackend
from edge_scheduler.inference.gpu_backend import GPUBackend
from edge_scheduler.inference.model_registry import MODEL_REGISTRY, PROJECT_ROOT
from edge_scheduler.inference.verification import summarize_profile


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def run(args, directory: Path, report: dict) -> None:
    rows = []
    outputs = {}
    report["devices"] = {}
    devices = ("cpu", "gpu") if args.device == "both" else (args.device,)
    try:
        for device in devices:
            backend_type = CPUBackend if device == "cpu" else GPUBackend
            backend = backend_type(
                root=args.root, warmup_runs=args.warmup,
                intra_op_num_threads=args.threads, device_id=args.device_id,
                profile_dir=directory / "profiles" if args.profile else None,
            )
            details = {"models": {}}
            report["devices"][device] = details
            try:
                started = time.monotonic()
                backend.prepare()
                details["prepare_s"] = time.monotonic() - started
                details["session_providers"] = backend.session_providers
                # Alternate models to exercise session reuse under mixed jobs.
                for iteration in range(args.runs):
                    for model in MODEL_REGISTRY:
                        started = time.monotonic()
                        output = backend.infer(model, args.input)
                        elapsed_ms = (time.monotonic() - started) * 1000
                        outputs.setdefault(f"{device}_{model}", output.copy())
                        rows.append({
                            "job_id": f"{device}-{iteration}-{model}",
                            "device": device, "model_id": model,
                            "iteration": iteration, "input_ref": args.input,
                            "execution_ms": elapsed_ms,
                            "top1_class": int(output[0].argmax()),
                        })
                for model in MODEL_REGISTRY:
                    times = [row["execution_ms"] for row in rows
                             if row["device"] == device and row["model_id"] == model]
                    details["models"][model] = {
                        "runs": len(times), "output_shape": list(outputs[f"{device}_{model}"].shape),
                        "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)),
                    }
            finally:
                backend.close()
                details["profile_paths"] = {
                    model: str(Path(path).relative_to(directory))
                    for model, path in backend.profile_paths.items()
                }
            if args.profile:
                details["profiles"] = {
                    model: summarize_profile(directory / path)
                    for model, path in details["profile_paths"].items()
                }
                if device == "gpu":
                    for model in MODEL_REGISTRY:
                        if not details["profiles"][model]["cuda_compute_verified"]:
                            raise RuntimeError(f"no CUDA compute evidence for {model}")
        if args.device == "both":
            report["comparison"] = {}
            for model in MODEL_REGISTRY:
                cpu, gpu = outputs[f"cpu_{model}"], outputs[f"gpu_{model}"]
                report["comparison"][model] = {
                    "max_abs_difference": float(np.max(np.abs(cpu - gpu))),
                    "rtol": args.rtol, "atol": args.atol,
                }
                np.testing.assert_allclose(cpu, gpu, rtol=args.rtol, atol=args.atol)
    finally:
        if rows:
            with (directory / "jobs.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if outputs:
            np.savez_compressed(directory / "outputs.npz", **outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument("--require-gpu", action="store_true",
                        help="assert this run includes GPU; GPU requests always fail on errors")
    parser.add_argument("--profile", action="store_true", help="save ORT traces; require CUDA compute for GPU")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", default="inputs/test.jpg")
    parser.add_argument("--runs", type=positive_integer, default=10, help="jobs per model per device")
    parser.add_argument("--warmup", type=positive_integer, default=5)
    parser.add_argument("--threads", type=positive_integer, default=2)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "t06")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.require_gpu and args.device == "cpu":
        parser.error("--require-gpu requires --device gpu or both")
    if args.device_id < 0:
        parser.error("--device-id must be >= 0")
    if not all(np.isfinite(value) and value >= 0 for value in (args.rtol, args.atol)):
        parser.error("rtol/atol must be finite and >= 0")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    directory = args.output_dir.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "run_id": run_id,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "onnxruntime": ort.__version__, "available_providers": ort.get_available_providers(),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
        },
        "timing_scope": "image load + preprocessing + synchronous ONNX run + output validation; excludes prepare",
        "profile_note": "operator event counts include warmup; profiling adds overhead",
    }
    try:
        run(args, directory, report)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"T06 {report['status'].upper()}: {directory}")
    for device, details in report.get("devices", {}).items():
        for model, metrics in details["models"].items():
            print(f"  {device:3} {model:11} mean={metrics['mean_ms']:.3f} ms p95={metrics['p95_ms']:.3f} ms")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
