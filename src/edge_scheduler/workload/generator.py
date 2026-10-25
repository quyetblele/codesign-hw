"""Generate seeded batch, Poisson arrival and burst workloads."""
import math
import random
from .manifest import WorkloadManifest, WorkloadItem, MODEL_IDS, PATTERNS, MAX_JOBS

RATES = {"low": 2.0, "medium": 8.0, "high": 20.0}

def generate_workload(*, count: int, models: tuple[str, ...], input_ref: str,
                      pattern: str = "batch", seed: int = 0,
                      rate_jobs_s: float | None = None,
                      burst_size: int = 5, burst_gap_s: float = 0.25):
    if type(count) is not int or not 1 <= count <= MAX_JOBS:
        raise ValueError("count out of bounds")
    if not isinstance(models, tuple) or not models or any(m not in MODEL_IDS for m in models):
        raise ValueError("invalid models")
    if pattern not in PATTERNS:
        raise ValueError("invalid pattern")
    if rate_jobs_s is not None:
        if isinstance(rate_jobs_s, bool) or not isinstance(rate_jobs_s, (float,int)) or not math.isfinite(rate_jobs_s) or rate_jobs_s <= 0:
            raise ValueError("invalid arrival rate")
        if pattern in ("batch", "burst"):
            raise ValueError("rate applies only to Poisson presets")
    if type(burst_size) is not int or burst_size < 1:
        raise ValueError("invalid burst size")
    if not isinstance(burst_gap_s, (int,float)) or not math.isfinite(burst_gap_s) or burst_gap_s <= 0:
        raise ValueError("invalid burst gap")
    WorkloadManifest("batch", seed, (WorkloadItem("validate", models[0], input_ref, 0),))
    rng = random.Random(seed)
    current = 0.0
    results = []
    rate = float(rate_jobs_s or RATES.get(pattern, 1.0))
    for i in range(count):
        if pattern == "batch":
            current = 0.0
        elif pattern == "burst":
            current = (i // burst_size) * burst_gap_s
        elif i:
            current += rng.expovariate(rate)
        results.append(WorkloadItem(f"job-{i+1:06d}", models[i % len(models)],
                                    input_ref, round(current, 6)))
    return WorkloadManifest(pattern, seed, tuple(results))
