# Báo cáo triển khai Tuần 3 — T09 và T12

**Ngày rà soát:** 25/10/2026
**Nhánh làm việc:** `feature/week3-workload-round-robin`
**Baseline:** `c4e73b5` (tích hợp module từ nhánh T08)
**Phạm vi:** Thành viên A: T09 Workload Generator/Replay, T12 Round Robin acceptance.

## 1. Các tệp đã triển khai
- `src/edge_scheduler/workload/manifest.py`: dữ liệu workload bất biến, schema_version=1, xác thực mẫu, ID duy nhất, thời gian tăng không giảm; JSON canonical và SHA-256.
- `src/edge_scheduler/workload/generator.py`: batch, low, medium, high, burst; RNG cục bộ theo seed, thời gian đến độc lập với thời gian hoàn thành Job.
- `src/edge_scheduler/workload/replay.py`: đồng hồ monotonic, deadline tuyệt đối, pump Controller trong lúc chờ; ghi replay.csv và độ trễ khi phát.
- `src/edge_scheduler/main.py`: CLI tham số workload, manifest, seed, rate và burst; lưu manifest và replay vào thư mục của mỗi lần chạy; duy trì CLI cũ.
- `tests/test_t09_workload.py`: 7 test.
- `tests/test_t12_round_robin.py`: 3 test cho policy, Controller + threads, Job FAILED.

## 2. Kết quả test
`PYTHONPATH=src:/tmp/codesign_review_dependencies python3 -m unittest discover -s tests -q`
**109 test; OK (skipped=3).** Bao gồm 10 test mới (T09=7, T12=3).
Các test GPU thật được skip mặc định theo cơ chế của dự án; không được mô tả 109/109 PASS.

## 3. Smoke test ONNX thật và replay cùng workload
Manifest gốc: `/tmp/edge_ai_week3_real/20261025T050222Z_003bc195/workload.json`.
SHA-256 (prefix): `1a96fea5b64483`.

| Chính sách | Accepted | Completed | Failed | Terminal reconciled | Max replay jitter |
|---|---:|---:|---:|---|---:|
| Round Robin | 6 | 6 | 0 | Có | 0,330 ms |
| CPU-only | 6 | 6 | 0 | Có | 0,495 ms |
| GPU-only | 6 | 6 | 0 | Có | 0,213 ms |

Cả ba run có cùng `manifest_sha256`, cùng thứ tự Job, input, model và thời điểm phát dự kiến. Tất cả có `workload.json`, `replay.csv`, `jobs.csv`, `resources.csv`, `summary.json`.

**Giới hạn:** 6 Job trong một đợt ngắn, chỉ xác nhận tính đúng và khả năng tích hợp, không đại diện cho performance benchmark. Chế độ CPU-only hiện vẫn prepare cả CPU/GPU Backend do kiến trúc của main.py. Các preset 2/8/20 jobs/s chỉ để dạy và chưa hiệu chỉnh theo năng lực máy.

## 4. Lệnh tái chạy
```bash
cd /home/quyet/t04_github_upload/week3_worktree
PYTHONPATH=src python3 -m edge_scheduler.main --scheduler round-robin \
  --jobs 6 --workload-pattern burst --seed 42 --burst-size 3 \
  --burst-gap 0.05 --no-gpu-telemetry --output-dir /tmp/week3_demo

# Sau đó dùng chính workload.json của lượt trước với cả CPU-only và GPU-only.
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

## 5. Lộ trình tiếp theo
1. **Review nhóm:** đồng ý API manifest `schema_version=1`, định nghĩa replay jitter và cấu hình test môi trường.
2. **Phía B — T10:** collector chính thức từ JobEvent, không nhầm với các file summary của harness.
3. **Phía B — T11:** profile service time (model, device) bằng cùng preprocessing, tách warm-up và đo.
4. **Phía A — T12:** tăng workload 10/11/50 Job, test FIFO và failure khi monitor không khả dụng; lưu chứng cứ theo manifest.
5. **T13:** Resource-aware lấy snapshot và profile thật; xử lý missing, stale, unavailable, NaN/Inf và giới hạn bộ nhớ.
6. **T15:** benchmark 4 scheduler x 4 workload x >=3 lượt, cùng manifest, cấu hình và phần cứng; đo P50/P95/P99 latency, throughput, scheduler overhead, replay jitter.
7. **Trước benchmark:** tách chế độ CPU-only không prepare GPU, thống nhất TF32/FP32 và unit test đối chiếu độ chính xác theo cả RTX 5060/GTX1650.

## 6. Lưu ý giới hạn môi trường
Các lệnh smoke test trong vòng này sử dụng PYTHONPATH trỏ tới dependencies đã cài tạm trong `/tmp/codesign_review_dependencies`. Khi bàn giao sang máy khác, hãy dùng venv được khóa theo `requirements.txt`; không giả định /tmp là cấu hình sản phẩm.

**Không push/merge GitHub trong lượt triển khai này.**
