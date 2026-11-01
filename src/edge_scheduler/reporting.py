"""Generate offline Markdown and PNG reports from a selected saved run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from .inference.model_registry import PROJECT_ROOT


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(value):
    if value is None or value == "":
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite value in run data")
    return value


def show(value):
    if value is None or value == "":
        return "—"
    return f"{value:.3f}" if isinstance(value, float) else str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(show(value) for value in row) + " |" for row in rows], ""]


def memory_unit(config: dict) -> str:
    # Original saved runs used MiB, including configs predating unit metadata.
    if "memory_unit" not in config and config.get("schema_version", 1) >= 2:
        raise ValueError("memory_unit is required for schema version 2 or newer")
    unit = config.get("memory_unit", "MiB")
    if unit not in ("MB", "MiB"):
        raise ValueError(f"unsupported memory unit: {unit}")
    return unit


def select_run(run: str | None, logs_dir: Path) -> Path:
    if run:
        directory = Path(run).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"run directory does not exist: {directory}")
        return directory
    pointer_path = logs_dir / "latest.json"
    if not pointer_path.exists():
        candidates = [logs_dir / name / "latest.json" for name in ("pipeline", "profiling", "benchmark")
                      if (logs_dir / name / "latest.json").exists()]
        if not candidates:
            raise ValueError("no latest run; use make run, make profile, or specify RUN=...")
        pointer_path = max(candidates, key=lambda path: json.loads(path.read_text()).get("started_at", ""))
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    directory = Path(pointer["path"]).resolve()
    if not directory.is_dir():
        raise ValueError(f"latest pointer refers to a missing run: {directory}; specify RUN=...")
    return directory


def shade_phases(axes, resources, origin):
    segments = []
    for row in resources:
        timestamp = numeric(row.get("state_timestamp_s", row.get("timestamp_s")))
        if timestamp is None:
            continue
        phase = row.get("phase", "")
        if not segments or segments[-1][0] != phase:
            segments.append((phase, timestamp - origin))
    colors = {"prepare": "#f5c870", "drain": "#b6d5ee", "shutdown": "#dddddd"}
    end = max((float(row["state_timestamp_s"]) - origin for row in resources), default=0)
    for index, (phase, start) in enumerate(segments):
        finish = segments[index + 1][1] if index + 1 < len(segments) else end
        if phase in colors:
            for axis in axes:
                axis.axvspan(start, finish, color=colors[phase], alpha=0.2)


def pipeline_report(plt, directory, config, summary, figures):
    unit = memory_unit(config)
    jobs, resources = read_csv(directory / "jobs.csv"), read_csv(directory / "resources.csv")
    completed = [row for row in jobs if row["status"] == "COMPLETED"]
    arrivals = [float(row["arrival_time_s"]) for row in jobs if row.get("arrival_time_s")]
    origin = min(arrivals) if arrivals else min((float(row["timestamp_s"]) for row in resources), default=0)
    lines = ["## Pipeline results", "", f"Policy: `{config.get('scheduler', 'unknown')}`. Successful jobs supply latency percentiles; failed jobs are reported separately.", ""]
    stats = summary["jobs"]
    lines += table(["Accepted", "Completed", "Failed", "Unfinished", "Reconciled", "Throughput jobs/s"], [[
        stats["accepted_jobs"], stats["completed_jobs"], stats["failed_jobs"], len(stats["unfinished_job_ids"]),
        stats["reconciled"], stats["throughput_jobs_s"]]])
    lines += table(["Metric (ms)", "Samples", "Mean", "P50", "P95", "P99"], [
        [name, values["count"], values["mean"], values["p50"], values["p95"], values["p99"]]
        for name, values in stats["successful_job_metrics"].items()])
    overhead = stats.get("scheduler_overhead_ms")
    lines += ["Scheduler overhead measures snapshot + selection + enqueue; it differs from dispatch waiting time.", ""]
    if overhead:
        lines += table(["Attempts", "Mean ms", "P95 ms", "Total ms"], [[
            overhead["count"], overhead["mean"], overhead["p95"], overhead["total"]]])
    else:
        lines += ["Scheduler overhead was not recorded by this historical run.", ""]
    lines += table(["Device", "Assigned", "Completed", "Failed"], [
        [device, value["assigned"], value["completed"], value["failed"]] for device, value in stats["devices"].items()])
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
        lines.extend([f"![{name}](figures/{name})", ""])
    fig, axis = plt.subplots(figsize=(10, 4))
    for name in ("incoming", "cpu", "gpu"):
        points = [(float(row["state_timestamp_s"]) - origin,
                   json.loads(row["queue_lengths_json"]).get(name)) for row in resources]
        points = [(x, y) for x, y in points if y is not None]
        if points:
            axis.step(*zip(*points), where="post", label=name)
    axis.set(xlabel="Seconds relative to first accepted job", ylabel="Queued jobs", title="Sampled queue lengths")
    shade_phases([axis], resources, origin)
    if axis.lines:
        axis.legend()
    save(fig, "queues.png")
    fig, axis = plt.subplots(figsize=(10, 4))
    for device in ("cpu", "gpu"):
        points = [(float(row["finish_time_s"]) - origin, float(row["latency_s"]) * 1000)
                  for row in completed if row["device"] == device and row["latency_s"]]
        if points:
            axis.scatter(*zip(*points), s=12, label=device, alpha=0.7)
    axis.set(xlabel="Completion time relative to first accepted job (s)", ylabel="Latency (ms)", title="Successful job latency")
    if axis.collections:
        axis.legend()
    save(fig, "latency.png")
    fig, axis = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(2)
    for field, label in (("dispatch_wait_s", "Dispatch wait"), ("device_wait_s", "Device queue wait"), ("execution_s", "Execution")):
        values = []
        for device in ("cpu", "gpu"):
            observed = [float(row[field]) * 1000 for row in completed if row["device"] == device and row[field] != ""]
            values.append(float(np.mean(observed)) if observed else np.nan)
        axis.bar(["cpu", "gpu"], values, bottom=bottom, label=label)
        bottom += np.nan_to_num(values)
    axis.set(ylabel="Mean time (ms)", title="Latency components of successful jobs")
    axis.legend()
    save(fig, "waiting_execution.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    groups = (("cpu_load_pct", "gpu_utilization_pct"), ("process_rss_mb", "ram_free_mb"),
              ("gpu_memory_used_mb", "gpu_memory_free_mb"))
    x = [float(row["timestamp_s"]) - origin for row in resources]
    for axis, metrics, axis_unit in zip(axes, groups, ("Percent", unit, unit)):
        for metric in metrics:
            values = [numeric(row[metric]) for row in resources]
            if any(value is not None for value in values):
                axis.plot(x, [np.nan if value is None else value for value in values], marker=".", label=metric)
        axis.set_ylabel(axis_unit)
        if axis.lines:
            axis.legend()
    shade_phases(axes, resources, origin)
    axes[-1].set_xlabel("Seconds relative to first accepted job")
    axes[0].set_title("Sampled resources; shaded prepare / drain / shutdown phases")
    save(fig, "resources.png")
    lines += ["## Measurement limits", "", "Prepare/warm-up is excluded from job timing. Resource plots include all phases, including negative time before the first accepted job. Queue curves use state timestamps; resource curves use hardware-sample timestamps.", "",
              "Missing metrics are gaps, not zero. Resource maxima are sampled observations, not absolute peaks. Sparse sampling can miss short queues or memory spikes. Small job counts do not support stable tail-percentile conclusions.", "",
              "Both CPU/GPU backends are prepared even for single-device policies. A finite alternating batch is not a replay benchmark of four workload levels.", ""]
    lines += table(["Component", "Diagnostics"], [[name, json.dumps(summary.get(name, {}), ensure_ascii=False)]
                                                for name in ("controller", "collector", "monitor", "logger")])
    return lines


def profiling_report(plt, directory, config, summary, figures):
    unit = memory_unit(config)
    rows = summary["profiles"]
    measurements = read_csv(directory / "measurements.csv")
    # Read the exported table too, so a missing required artifact is never hidden.
    read_csv(directory / "profile_summary.csv")
    read_csv(directory / "resources.csv")
    lines = ["## Model/device profiling", "", f"Runs per pair per repeat: {config['runs']}; repeats: {config['repeats']}; warm-up: {config['warmup']}; ORT threads: {config['threads']}.", ""]
    lines += table(["Model", "Device", "Status", "Samples", "Repeats", "Mean ms", "P50 ms", "P95 ms", "P99 ms", "Repeat mean std ms"], [
        [row[key] for key in ("model_id", "device", "status", "samples", "repeats", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "repeat_mean_std_ms")] for row in rows])
    lines += table(["Model", "Device", f"RSS baseline {unit}", f"RSS ready {unit}", f"RSS delta {unit}", f"RSS observed max {unit}", f"VRAM delta {unit}", f"VRAM observed max {unit}"], [
        [row.get(key) for key in ("model_id", "device", "baseline_process_rss_mb", "ready_process_rss_mb", "delta_process_rss_mb",
                                  "observed_max_process_rss_mb", "delta_gpu_memory_used_mb", "observed_max_gpu_memory_used_mb")] for row in rows])
    fig, axis = plt.subplots(figsize=(10, 4))
    groups, labels = [], []
    successful = {(trial["model_id"], trial["device"], trial["repeat"])
                  for trial in summary["trials"] if trial["status"] == "passed"}
    for row in rows:
        values = [float(measurement["execution_ms"]) for measurement in measurements
                  if (measurement["model_id"], measurement["device"]) == (row["model_id"], row["device"])
                  and (measurement["model_id"], measurement["device"], int(measurement["repeat"])) in successful]
        if values:
            groups.append(values)
            labels.append(f"{row['model_id']}\n{row['device']}")
    if groups:
        axis.boxplot(groups, tick_labels=labels, showmeans=True)
    axis.set(ylabel="Execution time (ms)", title="Successful trials; preparation and warm-up excluded")
    fig.tight_layout()
    fig.savefig(figures / "execution.png", dpi=150)
    plt.close(fig)
    lines += ["![Execution distribution](figures/execution.png)", ""]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, key, title in zip(axes, ("delta_process_rss_mb", "delta_gpu_memory_used_mb"),
                                ("Process RSS after warm-up minus baseline", "Whole-GPU VRAM after warm-up minus baseline")):
        observed = [(f"{row['model_id']}\n{row['device']}", row.get(key)) for row in rows if row.get(key) is not None]
        if observed:
            axis.bar([label for label, _ in observed], [value for _, value in observed])
        else:
            axis.text(0.5, 0.5, "No available measurements", ha="center", transform=axis.transAxes)
        axis.axhline(0, color="gray", linewidth=0.5)
        axis.set(ylabel=f"Median signed delta ({unit})", title=title)
    fig.tight_layout()
    fig.savefig(figures / "memory.png", dpi=150)
    plt.close(fig)
    lines += ["![Memory deltas](figures/memory.png)", "", "## Measurement limits", "",
              config["timing_scope"], "", config["memory_scope"], "",
              "Each model/device/repeat uses a fresh process. Memory baseline/ready/deltas in the table are medians across successful repeats; observed maxima cover sampled inference-phase values. Short inference spikes can be missed. Whole-GPU VRAM may include other applications; a signed delta is not an exact model allocation.", "",
              "Failed trials are excluded from calibration. Missing memory remains unavailable. CPU-only profiles do not measure GPU inference. These measurements do not compare scheduler policies.", ""]
    if config.get("verify_cuda"):
        lines += ["**ORT tracing was enabled: timing includes profiling overhead; this profile is rejected by the calibration loader.**", ""]
    lines += table(["Model", "Device", "Repeat", "Status", "Errors / unavailable metrics"], [[
        trial["model_id"], trial["device"], trial["repeat"], trial["status"],
        json.dumps({"errors": trial["errors"], "resource_metrics": trial.get("resources", {}).get("metrics", {})}, ensure_ascii=False)]
        for trial in summary["trials"]])
    return lines


def generate_report(directory: Path) -> Path:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("run_type") == "benchmark":
        from .benchmark_report import generate_benchmark_report
        return generate_benchmark_report(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    run_type = config.get("run_type")
    # Compatibility with original saved pipeline runs, independent of folder name.
    if run_type is None and "scheduler" in config and "jobs" in summary:
        run_type = "pipeline"
    if run_type not in ("pipeline", "profiling"):
        raise ValueError("unsupported or missing run_type in config.json")
    unit = memory_unit(config)
    output = directory / "report"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = [f"# {run_type.title()} run report", "", f"Run: `{config.get('run_id', directory.name)}`", "",
             f"Source: `{directory}`", "", f"Status: **{summary['status']}**", "",
             f"Memory unit: **{unit}** (1 {unit} = {1_000_000 if unit == 'MB' else 1_048_576:,} bytes). Values retain the saved run's original unit.", "",
             "This report reads saved artifacts; inference is not rerun.", ""]
    if summary.get("errors"):
        lines += ["## Run errors", "", *[f"- {show(error)}" for error in summary["errors"]], ""]
    lines += (pipeline_report if run_type == "pipeline" else profiling_report)(plt, directory, config, summary, figures)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "report.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=os.environ.get("RUN") or None, help="explicit run directory; defaults to latest")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    args = parser.parse_args(argv)
    try:
        directory = select_run(args.run, args.logs_dir.resolve())
        report = generate_report(directory)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Report failed: {exc}", file=sys.stderr)
        return 1
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
