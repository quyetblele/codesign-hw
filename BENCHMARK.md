# T15 và T16 — benchmark và đánh giá trực quan

Chạy từ project root với `.venv` theo `requirements.txt`, CUDA/NVML truy cập được.
Không chạy đồng thời inference nặng khác trong lúc benchmark. Công cụ ghi phần cứng,
package versions, commit, source/model/input SHA-256 và nhiệt độ/công suất/xung GPU
trước/sau từng lượt; không tự thay driver hay cấu hình máy.

```bash
make benchmark ARGS="--require-gpu-metrics --verify-tests"
make benchmark-report
make benchmark-report RUN=logs/benchmark/<experiment_id>
make report RUN=logs/benchmark/<experiment_id>
```

Mặc định: 1.000 job/workload, 3 lần lặp, hai model cân bằng, seed 42, warm-up 5,
ORT intra-op threads=2, resource sampling 100 ms, cooldown 2 giây. Thay đổi bằng
`JOBS`, `REPEATS`, hoặc `ARGS`; ít nhất 3 lần lặp. Ví dụ một đợt ngắn hơn:

```bash
make benchmark JOBS=500 REPEATS=3 ARGS="--require-gpu-metrics --verify-tests"
```

## Quy trình T15

1. Nếu có `--verify-tests`: chạy toàn suite với `T06_TEST_GPU=1 T08_TEST_GPU=1`;
   phải exit 0 và không skip. Lưu `tests.log`, dừng trước benchmark nếu thất bại.
2. Profiling mới tại máy hiện tại: 2 model × CPU/GPU × 3 lượt, 100 inference/cặp/lượt.
   Mỗi trial là process mới, tắt ORT tracing. Lưu toàn bộ calibration.
3. Tính năng lực ước lượng theo tỷ lệ model 50/50: `capacity = 1 / mean(service_time)`.
   Low = 0,5 × min capacity; medium = max(1,5 × low; 0,85 × max capacity);
   high = max(1,5 × medium; 1,25 × tổng capacity).
   Burst mặc định 100 job/đợt, khoảng cách để rate trung bình = 0,65 × tổng capacity.
   Capacity độc lập chỉ để chọn tải; queue thực tế mới phản ánh nghẽn dưới chạy đồng thời.
4. Sinh và đóng băng bốn manifest, lưu seed/hash. Xáo trộn thứ tự model bằng RNG riêng,
   không thay lịch arrival. Tránh model luân phiên trùng nhịp Round Robin.
5. Preflight riêng: GPU-only bật trace xác nhận compute CUDA cho cả hai model;
   sau đó chạy nhỏ bốn policy không trace trên cùng pilot manifest.
6. Chạy 48 lượt chính: 4 policy × 4 workload × 3 lần. Process mới mỗi lượt;
   đổi thứ tự policy/workload, cùng manifest qua mọi lần. Không bật trace/live display.
7. Kiểm tra exit status, manifest/hash, nội dung và ID job, replay order/arrival,
   completed/accepted/terminal reconciliation, decision log T13. Giữ lượt lỗi,
   không tự thay bằng lượt mới. Hard timeout dừng process group, lưu lỗi và log.
8. Tự tạo T16. Tạo lại report sau đó chỉ đọc artifact, không chạy inference.

## Artifact

```text
logs/benchmark/<experiment_id>/
├── config.json, experiment.json, summary.json
├── tests.log                    # nếu --verify-tests
├── calibration.log, calibration/ # profile mới và raw trials
├── manifests/                   # low/medium/high/burst và pilot
├── preflight/                   # CUDA trace + bốn policy smoke
├── runs/<workload>/repeat-N/<policy>/<run_id>/
│   └── config.json, workload.json, replay.csv, events.jsonl,
│       jobs.csv, resources.csv, summary.json, scheduler_decisions.jsonl (T13)
├── results.csv                  # mỗi dòng một lượt, gồm validation/missing metric
└── report/
    ├── index.html               # HTML tự chứa tám biểu đồ, chia sẻ riêng file được
    ├── report.md
    ├── analysis.json, comparison.csv, paired_comparison.csv
    ├── representative_runs.json
    └── figures/                 # PNG độc lập để chèn báo cáo/slide
```

## Cách đọc T16

- `overview.png`: throughput và P95; cột = trung bình các lượt, chấm = từng lượt,
  đoạn = min–max. CSV có sample standard deviation.
- `latency_quantiles.png`: trung bình P50/P95/P99 từng lượt, không phải phân vị gộp.
- `latency_ecdf.png`: tỷ lệ job hoàn thành trước một ngưỡng latency.
- `queues.png`: incoming/CPU/GPU theo thời gian, thấy nghẽn và quá trình drain.
- `waiting_execution.png`: tách dispatch wait, device queue wait, execution.
- `job_distribution.png`: model nào được gán CPU/GPU, trung bình qua các lượt hợp lệ.
- `resources.png`: CPU/GPU utilization, RSS và VRAM trong workload/drain.
- `overhead_jitter.png`: scheduler overhead và P95 replay jitter.

ECDF/timeline chọn lượt có P95 gần median nhất từng nhóm, hòa chọn repeat nhỏ hơn;
lựa chọn nằm trong `representative_runs.json`. So sánh Resource-aware với baseline
ghép cùng workload/repeat; phần trăm âm là kém hơn. Báo cáo không khẳng định ý nghĩa
thống kê chỉ từ ba lần lặp. Lượt failed/incomplete không tham gia hiệu năng thành công,
nhưng luôn hiển thị số failed/missing. Báo cáo thiếu ma trận mang nhãn INCOMPLETE.

## Giới hạn được giữ nguyên và công khai

- T13 hiện là ETA theo profile/queue/running work, chưa admission theo RAM/VRAM,
  chưa điều chỉnh trực tiếp theo CPU load; snapshot cũ bỏ remaining-time trong ETA.
- Mọi policy prepare cả CPU/GPU và cả hai model, kể cả CPU-only. VRAM không phải
  bằng chứng rằng CPU-only thực thi job trên GPU. Đây là so sánh scheduling với
  cùng tập session thường trú, chưa là so sánh deployment chỉ nạp một backend.
- Latency bắt đầu tại arrival thực tế của Controller; jitter so với manifest báo riêng.
- Throughput gồm thời gian drain. Resource summary chỉ trong arrival đầu → terminal cuối,
  phase workload/drain, loại stale. Missing là unavailable, không biến thành 0.
- RSS toàn process, VRAM toàn GPU; observed maximum theo sampling có thể bỏ lỡ peak.
- Một ảnh mẫu, một máy, một seed/manifest mỗi workload; nhiều seed, input dataset,
  thời gian dài hơn và nhiều lần lặp là các bước kiểm tra độ khái quát tiếp theo.
- Không sử dụng kết quả T15 để tuyên bố T13 đã đáp ứng đầy đủ memory-aware scheduling.

Exit code: 0 nếu runner và report thành công; 1 nếu calibration/preflight/benchmark/report lỗi;
130 nếu gián đoạn. Giữ nguyên artifact failed/interrupted để kiểm tra.
