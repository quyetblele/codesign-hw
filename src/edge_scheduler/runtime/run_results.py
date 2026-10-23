"""Final reports for finite pipeline runs; not a streaming MetricsCollector."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from ..core.models import Device, Job, JobStatus
from ..monitoring.monitoring_models import METRICS


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _duration(start, finish):
    return finish - start if start is not None and finish is not None else None


def job_row(job: Job) -> dict:
    return {
        "job_id": job.job_id, "model_id": job.model_id, "input_ref": job.input_ref,
        "device": job.assigned_device.value if job.assigned_device else None,
        "status": job.status.value, "arrival_time_s": job.arrival_time,
        "dispatch_time_s": job.dispatch_time, "start_time_s": job.start_time,
        "finish_time_s": job.finish_time,
        "dispatch_wait_s": _duration(job.arrival_time, job.dispatch_time),
        "device_wait_s": _duration(job.dispatch_time, job.start_time),
        "total_wait_s": _duration(job.arrival_time, job.start_time),
        "execution_s": _duration(job.start_time, job.finish_time),
        "latency_s": _duration(job.arrival_time, job.finish_time),
        "top1_class": int(np.asarray(job.result)[0].argmax())
        if job.status is JobStatus.COMPLETED else None,
        "error": job.error,
    }


def summarize_jobs(jobs: list[Job], accepted: int, terminal: int) -> dict:
    completed = [job for job in jobs if job.status is JobStatus.COMPLETED]
    failed = [job for job in jobs if job.status is JobStatus.FAILED]
    unfinished = [job.job_id for job in jobs if not job.terminal]
    errors = []
    if accepted != len(jobs) or terminal != len(completed) + len(failed):
        errors.append("queue counters do not match Job outcomes")
    for job in completed:
        times = (job.arrival_time, job.dispatch_time, job.start_time, job.finish_time)
        if any(value is None for value in times) or list(times) != sorted(times):
            errors.append(f"invalid lifecycle timestamps: {job.job_id}")
    elapsed = None
    if jobs and not unfinished:
        elapsed = max(job.finish_time for job in jobs) - min(job.arrival_time for job in jobs)
    metrics = {}
    rows = [job_row(job) for job in completed]
    for name in ("dispatch_wait_s", "device_wait_s", "total_wait_s", "execution_s", "latency_s"):
        values = [row[name] * 1000 for row in rows if row[name] is not None]
        metrics[name.removesuffix("_s") + "_ms"] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            **{f"p{percentile}": float(np.percentile(values, percentile)) if values else None
               for percentile in (50, 95, 99)},
        }
    return {
        "accepted_jobs": accepted, "completed_jobs": len(completed),
        "failed_jobs": len(failed), "terminal_jobs": terminal,
        "unfinished_job_ids": unfinished,
        "reconciled": accepted == terminal and not unfinished and not errors,
        "invariant_errors": errors,
        "failure_rate": len(failed) / accepted if accepted else None,
        "workload_elapsed_s": elapsed,
        "throughput_jobs_s": len(completed) / elapsed if elapsed and elapsed > 0 else None,
        "devices": {
            device.value: {
                "assigned": sum(job.assigned_device is device for job in jobs),
                "completed": sum(job.assigned_device is device for job in completed),
                "failed": sum(job.assigned_device is device for job in failed),
            } for device in Device
        },
        "successful_job_metrics": metrics,
    }


def write_job_csv(path: Path, jobs: list[Job]) -> None:
    # A header is also written when startup failed before the first acceptance.
    fields = tuple(job_row(Job("header", "mobilenetv2", "unused")))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(job_row(job) for job in jobs)


def summarize_resources(path: Path) -> dict:
    if not path.exists():
        return {"samples": 0, "metrics": {}, "phases": {}}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row[metric] != ""]
        reasons = Counter(json.loads(row["errors_json"]).get(metric) for row in rows)
        metrics[metric] = {
            "valid_samples": len(values), "unavailable_samples": len(rows) - len(values),
            "mean": statistics.fmean(values) if values else None,
            "observed_max": max(values) if values else None,
            "errors": {reason: count for reason, count in reasons.items() if reason},
        }
    return {"samples": len(rows), "metrics": metrics,
            "stale_samples": sum(row["stale"] == "True" for row in rows),
            "phases": dict(Counter(row["phase"] for row in rows)),
            "max_sample_duration_s": max((float(row["sample_duration_s"]) for row in rows),
                                         default=None),
            "scope": "all phases; arithmetic sample means; maxima are observed, not absolute peaks"}
