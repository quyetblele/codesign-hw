"""Deadline-based replay. Pump dispatch/events while awaiting future arrivals."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import csv
import math
import time
from .manifest import WorkloadManifest

@dataclass(frozen=True, slots=True)
class ReplayRecord:
    job_id: str
    planned_offset_s: float
    actual_offset_s: float
    jitter_s: float

def replay_workload(manifest: WorkloadManifest, submit: Callable, pump: Callable,
                    *, clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep,
                    poll_interval_s: float = 0.005,
                    on_record: Callable[[ReplayRecord], None] | None = None):
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("invalid replay poll interval")
    from ..core.models import Job
    begin = clock()
    records = []
    for index, item in enumerate(manifest.items):
        deadline = begin + item.arrival_offset_s
        while True:
            now = clock()
            if now >= deadline:
                break
            pump()
            remaining = deadline - clock()
            if remaining > 0:
                sleep(min(poll_interval_s, remaining))
        job = Job(item.job_id, item.model_id, item.input_ref,
                  arrival_offset_s=item.arrival_offset_s)
        submit(job)
        actual = clock() - begin
        record = ReplayRecord(item.job_id, item.arrival_offset_s, actual,
                              actual - item.arrival_offset_s)
        records.append(record)
        if on_record is not None:
            on_record(record)
        if index + 1 == len(manifest.items) or manifest.items[index+1].arrival_offset_s > item.arrival_offset_s:
            pump()
    return records

def write_replay_csv(path: str | Path, records: list[ReplayRecord]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("job_id","planned_offset_s","actual_offset_s","jitter_ms"))
        for r in records:
            w.writerow((r.job_id, f"{r.planned_offset_s:.6f}", f"{r.actual_offset_s:.6f}",
                        f"{r.jitter_s*1000:.3f}"))

def summarize_replay(records: list[ReplayRecord]):
    if not records:
        return dict(submitted_jobs=0, max_jitter_ms=None, mean_jitter_ms=None)
    js = [max(0, x.jitter_s)*1000 for x in records]
    return dict(submitted_jobs=len(records), max_jitter_ms=round(max(js),3),
                mean_jitter_ms=round(sum(js)/len(js),3))
