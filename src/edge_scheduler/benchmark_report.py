"""T16: rebuild comparison tables, standalone HTML and PNGs from T15 artifacts."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

import numpy as np

from .benchmark import PATTERNS, POLICIES, validate_run, sha256_file
from .inference.model_registry import PROJECT_ROOT
from .reporting import read_csv, show, table
from .runtime.run_results import summarize_rows, write_json
from .workload.manifest import WorkloadManifest

COLORS = dict(zip(POLICIES, ("#64748b", "#2563eb", "#d97706", "#059669")))
LABELS = ("CPU-only", "GPU-only", "Round Robin", "Resource-aware")
METRICS = ("throughput_jobs_s", "p50_ms", "p95_ms", "p99_ms", "mean_latency_ms",
           "dispatch_wait_ms", "device_wait_ms", "execution_ms", "overhead_ms", "jitter_p95_ms",
           "jitter_max_ms", "cpu_load_pct_mean", "gpu_utilization_pct_mean", "process_rss_mb_mean",
           "process_rss_mb_observed_max", "gpu_memory_used_mb_mean", "gpu_memory_used_mb_observed_max")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def load_measurements(directory, experiment):
    rows, raw = [], {}
    manifests = {name: WorkloadManifest.load(directory / info["path"])
                 for name, info in experiment["manifests"].items()}
    for run in experiment["runs"]:
        row = {key: run[key] for key in ("index", "repeat", "workload", "scheduler", "status")}
        row.update(run_path=run.get("run_path"), validation_errors=list(run.get("validation_errors", [])))
        if not run.get("run_path"):
            row["status"] = "failed"
            rows.append(row)
            continue
        child = directory / run["run_path"]
        try:
            manifest = manifests[run["workload"]]
            row["validation_errors"].extend(validate_run(child, manifest, run["scheduler"]))
            if manifest.sha256 != experiment["manifests"][run["workload"]]["sha256"]:
                row["validation_errors"].append("experiment manifest changed")
            config = json.loads((child / "config.json").read_text())
            frozen = experiment["config"]
            for key in ("threads", "warmup", "monitor_interval", "device_id", "gpu_index",
                        "gpu_uuid", "no_gpu_telemetry", "require_gpu_metrics"):
                if config.get(key) != frozen.get(key):
                    row["validation_errors"].append(f"configuration mismatch: {key}")
            if config.get("profile") or config.get("live"):
                row["validation_errors"].append("trace/live enabled during timed run")
            if run["scheduler"] == "resource-aware":
                expected_profile = directory / experiment["profile_path"]
                if sha256_file(expected_profile) != experiment["profile_sha256"]:
                    row["validation_errors"].append("profile hash mismatch")
                # Config has original absolute paths; relocating an entire bundle
                # remains supported, while the stored profile bytes are verified.
                original_profile = Path(frozen["output_dir"]) / frozen["run_id"] / experiment["profile_path"]
                if Path(config["service_profile"]) != original_profile:
                    row["validation_errors"].append("service profile identity mismatch")
            jobs = read_csv(child / "jobs.csv")
            converted = []
            for job in jobs:
                numeric = {key: (None if value == "" else float(value))
                           for key, value in job.items() if key.endswith("_s")}
                converted.append(dict(job, **numeric))
            summary = json.loads((child / "summary.json").read_text())
            stats = summarize_rows(converted, summary["jobs"]["accepted_jobs"], summary["jobs"]["terminal_jobs"])
            if not same_metrics(stats, summary["jobs"]):
                row["validation_errors"].append("raw job CSV metrics differ from summary")
            times = [j["arrival_time_s"] for j in converted if j["arrival_time_s"] is not None]
            finishes = [j["finish_time_s"] for j in converted if j["finish_time_s"] is not None]
            begin, end = min(times), max(finishes)
            resources = [r for r in read_csv(child / "resources.csv")
                         if r["phase"] in ("workload", "drain") and begin <= float(r["timestamp_s"]) <= end]
            replay = read_csv(child / "replay.csv")
            jitter = [float(r["jitter_ms"]) for r in replay]
            metrics = stats["successful_job_metrics"]
            latency = metrics["latency_ms"]
            row.update(manifest_sha256=manifest.sha256, accepted=stats["accepted_jobs"],
                       completed=stats["completed_jobs"], failed=stats["failed_jobs"],
                       unfinished=len(stats["unfinished_job_ids"]), failure_rate=stats["failure_rate"],
                       reconciled=stats["reconciled"], throughput_jobs_s=stats["throughput_jobs_s"],
                       p50_ms=latency["p50"], p95_ms=latency["p95"], p99_ms=latency["p99"],
                       mean_latency_ms=latency["mean"], dispatch_wait_ms=metrics["dispatch_wait_ms"]["mean"],
                       device_wait_ms=metrics["device_wait_ms"]["mean"], execution_ms=metrics["execution_ms"]["mean"],
                       overhead_ms=stats["scheduler_overhead_ms"]["mean"],
                       jitter_p95_ms=float(np.percentile(jitter, 95)) if jitter else None,
                       jitter_max_ms=max(jitter) if jitter else None, resource_samples=len(resources),
                       stale_samples=sum(r["stale"] == "True" for r in resources))
            for metric in ("cpu_load_pct", "gpu_utilization_pct", "process_rss_mb", "gpu_memory_used_mb"):
                values = [float(r[metric]) for r in resources if r[metric] and r["stale"] == "False"]
                row[metric + "_mean"] = statistics.fmean(values) if values else None
                row[metric + "_observed_max"] = max(values) if values else None
                row[metric + "_missing_samples"] = sum(not r[metric] for r in resources)
                row[metric + "_unavailable_reasons"] = dict(Counter(
                    json.loads(r["errors_json"]).get(metric, "unspecified") for r in resources if not r[metric]))
            raw[run["index"]] = dict(jobs=jobs, resources=resources, origin=begin, end=end)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["validation_errors"].append(f"{type(exc).__name__}: {exc}")
        if row["validation_errors"]:
            row["status"] = "failed"
        rows.append(row)
    return rows, raw


def same_metrics(actual, expected):
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_metrics(actual[k], expected[k]) for k in actual)
    if isinstance(actual, (float, int)) and not isinstance(actual, bool) and isinstance(expected, (float, int)):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def compare(rows, repeats):
    groups = []
    for pattern in PATTERNS:
        for policy in POLICIES:
            selected = [r for r in rows if r["workload"] == pattern and r["scheduler"] == policy]
            good = [r for r in selected if r["status"] == "passed"]
            row = dict(workload=pattern, scheduler=policy, expected_repeats=repeats,
                       attempted=len(selected), passed_repeats=len(good), failed_repeats=len(selected) - len(good),
                       missing_repeats=max(0, repeats - len(selected)), complete=len(good) == repeats,
                       completed_jobs=sum(r.get("completed", 0) for r in good))
            for metric in METRICS:
                values = [r[metric] for r in good if r.get(metric) is not None]
                row.update({metric + "_mean": statistics.fmean(values) if values else None,
                            metric + "_min": min(values) if values else None,
                            metric + "_max": max(values) if values else None,
                            metric + "_std": statistics.stdev(values) if len(values) > 1 else None,
                            metric + "_n": len(values)})
            groups.append(row)
    return groups


def paired_comparisons(rows):
    by_key = {(r["workload"], r["repeat"], r["scheduler"]): r for r in rows if r["status"] == "passed"}
    pairs = []
    for (pattern, repeat, policy), ra in by_key.items():
        if policy != "resource-aware":
            continue
        for baseline in POLICIES[:-1]:
            other = by_key.get((pattern, repeat, baseline))
            if other is None or other["manifest_sha256"] != ra["manifest_sha256"]:
                continue
            pairs.append(dict(workload=pattern, repeat=repeat, baseline=baseline,
                              p95_reduction_pct=100 * (1 - ra["p95_ms"] / other["p95_ms"]),
                              throughput_gain_pct=100 * (ra["throughput_jobs_s"] / other["throughput_jobs_s"] - 1)))
    return pairs


def representatives(rows):
    selected = {}
    for pattern in PATTERNS:
        for policy in POLICIES:
            candidates = [r for r in rows if r["workload"] == pattern and r["scheduler"] == policy
                          and r["status"] == "passed"]
            if candidates:
                median = statistics.median(r["p95_ms"] for r in candidates)
                selected[(pattern, policy)] = min(candidates, key=lambda r: (abs(r["p95_ms"] - median), r["repeat"]))
    return selected


def make_figures(output, rows, raw, selected):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    names = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
        names.append(name)

    def values(pattern, policy, metric):
        return [r[metric] for r in rows if r["workload"] == pattern and r["scheduler"] == policy
                and r["status"] == "passed" and r.get(metric) is not None]

    def bars(axis, pattern, metric, label):
        for i, policy in enumerate(POLICIES):
            points = values(pattern, policy, metric)
            if points:
                mean = statistics.fmean(points)
                axis.bar(i, mean, color=COLORS[policy], alpha=.8)
                axis.vlines(i, min(points), max(points), color="black", lw=1)
                axis.scatter(np.linspace(i - .1, i + .1, len(points)), points, color="black", s=13, zorder=3)
            else:
                axis.text(i, .05, "N/A", ha="center", transform=axis.get_xaxis_transform())
        axis.set_xticks(range(4), LABELS, rotation=22, ha="right")
        axis.set(title=pattern.upper(), ylabel=label)
        axis.grid(axis="y", alpha=.15)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for col, pattern in enumerate(PATTERNS):
        bars(axes[0, col], pattern, "throughput_jobs_s", "Throughput (jobs/s), higher is better")
        bars(axes[1, col], pattern, "p95_ms", "P95 latency (ms), lower is better")
    fig.suptitle("T15 comparison: bars = mean of runs; dots = individual runs; lines = min/max", y=1.02)
    save(fig, "overview.png")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for axis, pattern in zip(axes, PATTERNS):
        for metric, marker in (("p50_ms", "o"), ("p95_ms", "s"), ("p99_ms", "^")):
            means = [statistics.fmean(v) if (v := values(pattern, p, metric)) else np.nan for p in POLICIES]
            axis.plot(range(4), means, marker=marker, label=metric)
        axis.set_xticks(range(4), LABELS, rotation=25, ha="right")
        axis.set(title=pattern.upper(), ylabel="Mean of per-run quantiles (ms)")
        axis.legend()
    save(fig, "latency_quantiles.png")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for axis, pattern in zip(axes, PATTERNS):
        for policy in POLICIES:
            representative = selected.get((pattern, policy))
            if representative:
                jobs = raw[representative["index"]]["jobs"]
                latency = sorted(float(j["latency_s"]) * 1000 for j in jobs if j["status"] == "COMPLETED")
                axis.step(latency, np.arange(1, len(latency) + 1) / len(latency), where="post",
                          label=policy, color=COLORS[policy])
        axis.set(title=pattern.upper(), xlabel="Latency (ms)", ylabel="Fraction completed", ylim=(0, 1.02))
        if axis.lines:
            axis.legend(fontsize=7)
    save(fig, "latency_ecdf.png")

    fig, axes = plt.subplots(4, 4, figsize=(17, 12))
    for row_index, pattern in enumerate(PATTERNS):
        for col, policy in enumerate(POLICIES):
            axis = axes[row_index, col]
            chosen = selected.get((pattern, policy))
            if chosen:
                data = raw[chosen["index"]]
                x = [float(r["state_timestamp_s"]) - data["origin"] for r in data["resources"]]
                for queue, color in (("incoming", "#94a3b8"), ("cpu", "#d97706"), ("gpu", "#2563eb")):
                    y = [json.loads(r["queue_lengths_json"]).get(queue, np.nan) for r in data["resources"]]
                    axis.step(x, y, where="post", label=queue, color=color)
                axis.legend(fontsize=7)
                axis.set_xlim(left=0)
            axis.set(title=f"{pattern} / {policy}", xlabel="Elapsed workload (s)", ylabel="Sampled queued jobs")
    save(fig, "queues.png")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for axis, pattern in zip(axes, PATTERNS):
        bottom = np.zeros(4)
        for metric, label, color in (("dispatch_wait_ms", "Dispatch wait", "#94a3b8"),
                                    ("device_wait_ms", "Device queue wait", "#e9a23b"),
                                    ("execution_ms", "Execution", "#2563eb")):
            means = [statistics.fmean(v) if (v := values(pattern, p, metric)) else np.nan for p in POLICIES]
            axis.bar(range(4), means, bottom=bottom, label=label, color=color)
            bottom += np.nan_to_num(means)
        axis.set_xticks(range(4), LABELS, rotation=25, ha="right")
        axis.set(title=pattern.upper(), ylabel="Mean time per successful job (ms)")
        axis.legend(fontsize=7)
    save(fig, "waiting_execution.png")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    components = (("mobilenetv2", "cpu", "#fbbf24"), ("mobilenetv2", "gpu", "#93c5fd"),
                  ("resnet50", "cpu", "#b45309"), ("resnet50", "gpu", "#1d4ed8"))
    for axis, pattern in zip(axes, PATTERNS):
        bottom = np.zeros(4)
        for model, device, color in components:
            means = []
            for policy in POLICIES:
                matching = [r for r in rows if r["workload"] == pattern and r["scheduler"] == policy and r["status"] == "passed"]
                counts = [sum(j["model_id"] == model and j["device"] == device for j in raw[r["index"]]["jobs"]) for r in matching]
                means.append(statistics.fmean(counts) if counts else np.nan)
            axis.bar(range(4), means, bottom=bottom, label=f"{model}/{device}", color=color)
            bottom += np.nan_to_num(means)
        axis.set_xticks(range(4), LABELS, rotation=25, ha="right")
        axis.set(title=pattern.upper(), ylabel="Mean assigned jobs per run")
        axis.legend(fontsize=6)
    save(fig, "job_distribution.png")

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, pattern in enumerate(PATTERNS):
        for policy in POLICIES:
            chosen = selected.get((pattern, policy))
            if not chosen:
                continue
            data = raw[chosen["index"]]
            x = [float(r["timestamp_s"]) - data["origin"] for r in data["resources"]]
            for col, metric, style in ((0, "cpu_load_pct", "-"), (0, "gpu_utilization_pct", "--"),
                                       (1, "process_rss_mb", "-"), (2, "gpu_memory_used_mb", "-")):
                y = [float(r[metric]) if r[metric] and r["stale"] == "False" else np.nan for r in data["resources"]]
                if any(math.isfinite(v) for v in y):
                    axes[i, col].plot(x, y, color=COLORS[policy], linestyle=style,
                                      label=policy + (" GPU" if style == "--" else ""), lw=1)
        for col, label in enumerate(("CPU solid / GPU dashed (%)", "Whole-process RSS (MB)", "Whole-GPU VRAM (MB)")):
            axis = axes[i, col]
            axis.set(title=f"{pattern}: {label}", xlabel="Elapsed workload (s)")
            if axis.lines:
                axis.legend(fontsize=6)
            else:
                axis.text(.5, .5, "Unavailable", transform=axis.transAxes, ha="center")
    save(fig, "resources.png")

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for col, pattern in enumerate(PATTERNS):
        bars(axes[0, col], pattern, "overhead_ms", "Mean scheduler overhead (ms)")
        bars(axes[1, col], pattern, "jitter_p95_ms", "Replay jitter P95 (ms)")
    save(fig, "overhead_jitter.png")
    return names


def standalone_html(markdown, output):
    """Small escaped Markdown renderer; embeds plots so HTML is one shareable file."""
    body, in_table = [], False
    for line in markdown.splitlines():
        if line.startswith("| "):
            if re.match(r"^\|[\s|:-]+$", line):
                continue
            if not in_table:
                body.append('<div class="scroll"><table>')
                in_table = True
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table></div>")
            in_table = False
        match = re.fullmatch(r"!\[([^]]*)\]\((figures/[^)]+)\)", line)
        if match:
            payload = base64.b64encode((output / match[2]).read_bytes()).decode()
            body.append(f'<img alt="{html.escape(match[1])}" src="data:image/png;base64,{payload}">')
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 4)
            body.append(f"<h{level}>{html.escape(line[level:].strip())}</h{level}>")
        elif line:
            safe = html.escape(line)
            safe = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', safe)
            body.append(f"<p>{safe}</p>")
    if in_table:
        body.append("</table></div>")
    document = ('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>T15–T16 Benchmark</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#17263b;background:#f8fafc}'
                'h1,h2{color:#12335b}img{width:100%;height:auto;background:white;border:1px solid #dbe3ee;border-radius:10px;margin:12px 0}'
                '.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td{padding:9px 12px;border:1px solid #dbe3ee;white-space:nowrap}'
                'tr:first-child{font-weight:700;background:#e7eff9}a{color:#1d4ed8}</style><body>' + "\n".join(body) + '</body></html>')
    (output / "index.html").write_text(document, encoding="utf-8")


def generate_benchmark_report(directory):
    directory = Path(directory).resolve()
    experiment = json.loads((directory / "experiment.json").read_text())
    if experiment.get("run_type") != "benchmark":
        raise ValueError("not a benchmark experiment")
    output = directory / "report"
    output.mkdir(exist_ok=True)
    rows, raw = load_measurements(directory, experiment)
    groups = compare(rows, experiment["config"]["repeats"])
    pairs = paired_comparisons(rows)
    selected = representatives(rows)
    write_csv(directory / "results.csv", rows, (list(dict.fromkeys(k for r in rows for k in r)) if rows else
                                              ["index", "workload", "scheduler", "repeat", "status"]))
    write_csv(output / "comparison.csv", groups)
    write_csv(output / "paired_comparison.csv", pairs, ["workload", "repeat", "baseline", "p95_reduction_pct", "throughput_gain_pct"])
    write_json(output / "representative_runs.json", {f"{k[0]}/{k[1]}": {key: r[key] for key in ("index", "repeat", "run_path", "p95_ms")}
                                                        for k, r in selected.items()})
    complete = experiment["status"] == "passed" and all(g["complete"] for g in groups)
    report_status = "complete" if complete else "incomplete"
    write_json(output / "analysis.json", dict(status=report_status, groups=groups, paired_comparisons=pairs,
                                               validation_errors=[r for r in rows if r["status"] != "passed"]))
    figures = make_figures(output, rows, raw, selected)
    config = experiment["config"]
    passed = sum(r["status"] == "passed" for r in rows)
    lines = ["# T15–T16: so sánh điều phối CPU/GPU", "", f"Experiment: `{directory.name}`.", "",
             f"Trạng thái: {report_status.upper()}; {passed}/{len(experiment['schedule'])} lượt benchmark hợp lệ. Trạng thái runner: {experiment['status']}.", "",
             "[Bảng từng lượt CSV](../results.csv) · [Bảng tổng hợp CSV](comparison.csv) · [So sánh ghép cặp CSV](paired_comparison.csv) · [Cấu hình và lịch chạy](../experiment.json)", "",
             "## Cách đọc nhanh", "",
             "Throughput cao hơn và latency thấp hơn là tốt hơn. Cột là trung bình các lượt; chấm đen là từng lượt; đoạn đen là min–max. Không dùng ba lần lặp để khẳng định ý nghĩa thống kê.", "",
             "![](figures/overview.png)", "", "## Điều kiện và phương pháp", "",
             f"Mỗi workload có {config['jobs']} job, {config['repeats']} lần lặp, hai model trộn cân bằng bằng seed cố định. Batch=1; ORT intra-op threads={config['threads']}; warm-up={config['warmup']}; sampling={config['monitor_interval']} s; cooldown={config['cooldown']} s.", "",
             "Mỗi lượt chạy trong process mới. Cùng workload dùng cùng manifest/hash qua mọi scheduler và lần lặp. Thứ tự workload/scheduler luân chuyển; log nhiệt độ/công suất/xung GPU trước và sau lượt nằm trong experiment.json. Đây không phải kiểm soát nhiệt độ tuyệt đối.", "",
             "Profiling T11 mới trên máy này dùng process riêng từng model/device/repeat, tắt trace. CUDA trace được kiểm tra riêng trước benchmark; các lượt đo không bật trace hoặc live display.", "",
             "Latency tính từ arrival thực tế khi Controller tiếp nhận; replay jitter đo độ trễ so với lịch manifest và báo riêng. Throughput = completed / (terminal cuối − arrival đầu), gồm cả thời gian drain.", "",
             "Tài nguyên chỉ lấy mẫu phase workload/drain nằm giữa arrival đầu và terminal cuối; loại mẫu stale. RSS là toàn process; VRAM là toàn GPU và có thể gồm ứng dụng khác. Max là observed maximum theo sampling, không phải peak tuyệt đối. Missing giữ trống trong CSV; số mẫu thiếu và lý do được lưu ở results.csv.", ""]
    hardware = config["hardware"]
    lines += [f"Máy đo: {hardware['platform']}; physical cores={hardware['physical_cores']}, logical CPUs={hardware['logical_cpus']}, RAM={hardware['ram_total_mb']:.0f} MB.", "",
              f"GPU (name, UUID, driver, memory MiB, nhiệt độ °C, power W lúc bắt đầu): {hardware['gpu'].get('stdout', hardware['gpu'].get('error', 'unavailable'))}.", ""]
    if experiment.get("rates_jobs_s"):
        lines += table(["Workload", "Rate mục tiêu job/s", "Job", "Arrival cuối (s)", "Manifest SHA-256"], [
            [p, experiment["rates_jobs_s"].get(p, "burst"), info["jobs"], info["last_arrival_s"], info["sha256"]]
            for p, info in experiment["manifests"].items()])
        lines += [f"Năng lực ước lượng từ profile độc lập (job/s): {json.dumps(experiment['estimated_capacities_jobs_s'])}. Low=0,5×min capacity; medium=max(1,5×low; 0,85×max capacity); high=max(1,5×medium; 1,25×tổng capacity). Burst rate trung bình mục tiêu=0,65×tổng capacity, theo burst_size/burst_gap đã lưu.", "",
                  "Các capacity này là ước lượng từ inference độc lập; chưa khẳng định bằng năng lực chạy CPU/GPU đồng thời. Mức nghẽn thực tế phải đọc từ queue và waiting time.", ""]
    lines += ["## Bảng kết quả", "", "P50/P95/P99 dưới đây là trung bình các phân vị tính riêng từng lượt; không phải phân vị của dữ liệu gộp. Chỉ lượt hợp lệ tham gia số liệu hiệu năng; số failed/missing vẫn hiển thị. P99 của workload hữu hạn là mô tả, chưa chứng minh tail ổn định.", ""]
    lines += table(["Workload", "Scheduler", "PASS / dự kiến", "Failed / thiếu", "Throughput job/s", "P50 ms", "P95 ms", "P99 ms", "Overhead ms"], [
        [g["workload"], g["scheduler"], f"{g['passed_repeats']}/{g['expected_repeats']}",
         f"{g['failed_repeats']}/{g['missing_repeats']}", *[g[m + "_mean"] for m in
           ("throughput_jobs_s", "p50_ms", "p95_ms", "p99_ms", "overhead_ms")]] for g in groups])
    lines += ["## Nhận xét theo workload", ""]
    for pattern in PATTERNS:
        matching = [g for g in groups if g["workload"] == pattern and g["complete"]]
        if len(matching) != 4:
            lines += [f"{pattern.upper()}: chưa đủ bốn scheduler với số lần lặp yêu cầu; chưa kết luận so sánh đầy đủ.", ""]
            continue
        best_latency = min(matching, key=lambda g: g["p95_ms_mean"])
        best_throughput = max(matching, key=lambda g: g["throughput_jobs_s_mean"])
        lines += [f"{pattern.upper()}: P95 trung bình thấp nhất là {best_latency['scheduler']} ({best_latency['p95_ms_mean']:.2f} ms); throughput trung bình cao nhất là {best_throughput['scheduler']} ({best_throughput['throughput_jobs_s_mean']:.2f} job/s).", ""]
        for baseline in ("gpu-only", "round-robin"):
            paired = [p for p in pairs if p["workload"] == pattern and p["baseline"] == baseline]
            if paired:
                latency = [p["p95_reduction_pct"] for p in paired]
                throughput = [p["throughput_gain_pct"] for p in paired]
                lines += [f"Resource-aware so với {baseline}, ghép theo lần lặp: giảm P95 {statistics.fmean(latency):+.2f}% (min–max {min(latency):+.2f}% đến {max(latency):+.2f}%); tăng throughput {statistics.fmean(throughput):+.2f}%. Giá trị âm nghĩa là kém hơn baseline.", ""]
    lines += ["## Phân bố latency và nguyên nhân chờ", "", "![](figures/latency_quantiles.png)", "",
              "![](figures/latency_ecdf.png)", "", "![](figures/waiting_execution.png)", "",
              "## Queue, phân công job và tài nguyên", "",
              "ECDF và timeline dùng lượt có P95 gần median nhất trong từng cặp workload–scheduler, hòa chọn repeat nhỏ hơn; lựa chọn cụ thể ở representative_runs.json. Không gộp các lượt thành một timeline. Queue lấy mẫu có thể bỏ lỡ đỉnh ngắn.", "",
              "![](figures/queues.png)", "", "![](figures/job_distribution.png)", "", "![](figures/resources.png)", "",
              "## Chi phí điều phối và độ trễ phát job", "", "![](figures/overhead_jitter.png)", "",
              "Overhead đo snapshot + chọn thiết bị + enqueue; khác thời gian job chờ dispatch. Replay cùng lịch dự kiến không đảm bảo arrival thực tế giống hệt, nên phải đọc jitter cùng latency.", "",
              "## Giới hạn và phạm vi kết luận", "",
              "Resource-aware hiện tối thiểu hóa ETA từ profile, queued work và running work khi snapshot đủ mới. Chưa có admission theo RAM/VRAM hoặc điều chỉnh trực tiếp theo CPU load. Snapshot cũ làm scheduler dùng queue/profile, nên remaining-time có thể thiếu. Benchmark này không nghiệm thu đầy đủ chính sách bộ nhớ T13.", "",
              "Mọi policy đều prepare cả CPU và GPU, cùng hai model. CPU-only mô tả thiết bị chạy job; VRAM của nó vẫn gồm session GPU đã nạp. Số liệu bộ nhớ không đại diện cho triển khai chỉ nạp backend đang dùng.", "",
              "Một máy, một ảnh mẫu, tỷ lệ model 50/50, một manifest/seed cho mỗi workload, ba lần lặp mặc định; kết quả chưa đại diện mọi Edge AI, accuracy dataset hoặc mọi cách trộn model. Seed ở đây kiểm soát tái lập, chưa là nghiên cứu độ nhạy nhiều seed. Low-load throughput thường bị giới hạn bởi arrival rate.", "",
              "## Đối soát và lỗi", ""]
    lines += table(["Lượt", "Workload", "Scheduler", "Repeat", "Trạng thái", "Completed / accepted", "Lỗi validation"], [
        [r["index"], r["workload"], r["scheduler"], r["repeat"], r["status"],
         f"{r.get('completed', '—')}/{r.get('accepted', '—')}", "; ".join(r["validation_errors"])] for r in rows])
    lines += [*experiment["errors"], "", "Báo cáo được tái tạo từ artifact đã lưu; tạo lại report không chạy inference.", ""]
    markdown = "\n".join(lines)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    standalone_html(markdown, output)
    return output / "index.html"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(os.environ["RUN"]) if os.environ.get("RUN") else None)
    args = parser.parse_args(argv)
    try:
        directory = args.run
        if directory is None:
            directory = Path(json.loads((PROJECT_ROOT / "logs" / "benchmark" / "latest.json").read_text())["path"])
        print(f"Report: {generate_benchmark_report(directory)}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Benchmark report failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
