"""Shared row format and metric definitions for streaming and final reports."""
from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np

from ..core.models import Device, Job, JobStatus
from ..monitoring.monitoring_models import METRICS


def write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + "_", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


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
        "scheduler_overhead_s": job.scheduler_overhead_s,
    }


def summarize_jobs(jobs: list[Job], accepted: int, terminal: int) -> dict:
    return summarize_rows([job_row(job) for job in jobs], accepted, terminal)


def summarize_rows(rows: list[dict], accepted: int, terminal: int) -> dict:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    failed = [row for row in rows if row["status"] == "FAILED"]
    unfinished = [row["job_id"] for row in rows if row["status"] not in ("COMPLETED", "FAILED")]
    errors = []
    if accepted != len(rows) or terminal != len(completed) + len(failed):
        errors.append("queue counters do not match Job outcomes")
    for row in completed:
        times = tuple(row[key + "_time_s"] for key in ("arrival", "dispatch", "start", "finish"))
        if any(value is None for value in times) or list(times) != sorted(times):
            errors.append(f"invalid lifecycle timestamps: {row['job_id']}")
    elapsed = None
    if rows and not unfinished:
        elapsed = max(row["finish_time_s"] for row in rows) - min(row["arrival_time_s"] for row in rows)
    metrics = {}
    for name in ("dispatch_wait_s", "device_wait_s", "total_wait_s", "execution_s", "latency_s"):
        values = [row[name] * 1000 for row in completed if row[name] is not None]
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
                "assigned": sum(row["device"] == device.value for row in rows),
                "completed": sum(row["device"] == device.value for row in completed),
                "failed": sum(row["device"] == device.value for row in failed),
            } for device in Device
        },
        "successful_job_metrics": metrics,
        "scheduler_overhead_ms": distribution([
            row["scheduler_overhead_s"] * 1000 for row in rows
            if row.get("scheduler_overhead_s") is not None
        ]),
    }


def distribution(values: list[float]) -> dict:
    return {"count": len(values), "mean": statistics.fmean(values) if values else None,
            "total": sum(values),
            **{f"p{p}": float(np.percentile(values, p)) if values else None for p in (50, 95, 99)}}


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
