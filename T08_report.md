# T08 — Resource Monitor

Ngày kiểm chứng: 2026-10-18. **T08 đã triển khai và kiểm chứng độc lập, đồng thời
theo dõi được hai backend T06 chạy song song.** Queue, Controller, WorkerLoop
và MetricsCollector toàn hệ thống vẫn thuộc T05/T07/T10.

## Thành phần và thư viện

| File | Trách nhiệm |
|---|---|
| `src/edge_scheduler/monitoring_models.py` | ProbeReading, ResourceSample bất biến và WorkerRegistry |
| `src/edge_scheduler/system_probe.py` | psutil: CPU/RAM/process, prime CPU counter trong sampler thread |
| `src/edge_scheduler/gpu_probe.py` | NVML: VRAM/utilization, UUID/name và lỗi từng nhóm metric |
| `src/edge_scheduler/resource_monitor.py` | Lấy mẫu nền, cache, stale, ghép callback trạng thái, T04 snapshot |
| `src/edge_scheduler/resource_log.py` | ResourceSampleSink ghi CSV bằng thread riêng và queue có giới hạn |
| `scripts/T08_monitor_test.py` | Theo dõi idle/T06, log theo run, đo overhead theo cặp |
| `tests/test_t08_*.py` | Probe, lifecycle, concurrency, logging, compatibility và phần cứng thật |

Python 3.10; bổ sung `psutil==7.2.2` và `nvidia-ml-py==13.615.71` vào
`requirements.txt` và cài vào `.venv`. NVML binding import qua `pynvml`.
Giữ nguyên backend T06 và API `ResourceMonitor.snapshot() -> ResourceSnapshot`.

## Thông số và phạm vi đo

Mỗi mẫu gồm 10 metric:

| Metric | Ý nghĩa |
|---|---|
| `cpu_load_pct` | Tổng CPU utilization trong OS/WSL |
| `ram_total_mb` | RAM total trong OS/WSL |
| `ram_free_mb` | RAM **available**, theo tên trường T04 hiện có |
| `ram_used_pct` | Usage % do psutil trả về |
| `process_rss_mb` | RSS của toàn process ứng dụng |
| `process_cpu_pct` | CPU % của process; có thể vượt 100 khi dùng nhiều core |
| `gpu_memory_total_mb` | VRAM total của GPU được chọn |
| `gpu_memory_used_mb` | VRAM used của toàn GPU, có thể gồm ứng dụng khác |
| `gpu_memory_free_mb` | VRAM free của GPU được chọn |
| `gpu_utilization_pct` | Utilization do NVML trả về nếu được hỗ trợ |

Trường `_mb` dùng **MiB = bytes / 1024²**. CPU/RAM trên máy kiểm chứng có phạm
vi Linux/WSL guest; không coi đó là toàn bộ Windows host. CPU/GPU worker là
thread cùng process nên không quy RSS process thành bộ nhớ riêng của từng worker.

Log còn có sequence, timestamp bắt đầu mẫu, thời gian đọc, thời điểm đọc trạng
thái worker/queue, tuổi mẫu, stale, phase và lỗi. Giá trị chưa đo được là None
(ô CSV rỗng); nguyên nhân nằm trong `errors_json`. Không giả missing thành 0.
VRAM và utilization được truy vấn độc lập, nên utilization unsupported không
làm mất thông tin VRAM. Không suy ra GPU backend hỏng từ lỗi NVML telemetry.

Metadata ghi platform, CPU logic, PID, GPU name/UUID/index và package versions.
NVML index khác khái niệm CUDA ordinal; script cho phép chỉ rõ GPU UUID và yêu
cầu mapping này khi dùng CUDA_VISIBLE_DEVICES với workload GPU có telemetry.

## Sampling, concurrency và lỗi

- Mặc định 200 ms/mẫu, stale sau 1 giây; cấu hình được cả hai giá trị.
- Prime CPU counters ở sampler thread, đợi một interval rồi mới đọc mẫu đầu.
- Snapshot không gọi psutil/NVML hay ghi file. Phần cứng lấy từ cache; các
  callback trạng thái phải nhanh, trả bản sao thread-safe và không làm I/O.
- Chỉ giữ lock khi đọc/thay cache, không giữ khi gọi native probe hoặc callback.
- Timestamp dùng `time.monotonic()` và lấy ở **đầu quá trình đọc mẫu**. Nhờ đó
  query chậm không khiến một metric đã cũ bị gắn timestamp mới ở cuối query.
- Sample duration được ghi rõ; khi query chậm hơn chu kỳ, bỏ lịch tick đã lỡ
  để tránh vòng lặp catch-up chiếm CPU. Sampling không bảo đảm hard realtime.
- Trước mẫu đầu: sequence=0, timestamp=0, metric=None và stale=True. Reader
  không tạo timestamp mới cho dữ liệu phần cứng cũ.
- Mapping trong sample/snapshot do monitor trả về là bản sao bất biến.
- Lỗi probe được biểu diễn theo metric và không làm mất dữ liệu probe khác.
  Lỗi bất ngờ làm sampler dừng được công khai qua `diagnostics.sampler_error`.
- Writer CSV có queue bounded mặc định 1024 mẫu. Không chặn sampler khi queue
  đầy; tăng dropped_samples. CLI nghiệm thu coi mất mẫu là lỗi.
- Writer hỏng được báo qua logger.stats/error và monitor.diagnostics.sink_error.
- `start()`/`close()` idempotent. Có thể start lại sau khi thread cũ đã dừng;
  sink phải còn mở hoặc được thay mới. Close join có timeout và báo lỗi nếu
  native call/callback còn bị chặn. Không ép đóng NVML từ thread khác khi read
  chưa kết thúc. Native API không có bảo đảm thời gian thực cứng.
- Caller đóng monitor trước, rồi close logger để drain/flush. NVML cleanup
  thuộc sampler thread; lỗi cleanup được ghi trong diagnostics.

## Tích hợp với các task khác

T04 không đổi required fields của ResourceSnapshot hoặc chữ ký snapshot.
WorkerState thêm hai trường ở cuối với default, giữ tương thích cách gọi cũ:

```python
running_started_s: float | None = None
estimated_remaining_known: bool = False
```

Compatibility có test bằng constructor positional T04 cũ. WorkerRegistry là
adapter tùy chọn; T07 có thể dùng registry riêng qua callback cùng kiểu dữ liệu.

```python
monitor = ResourceMonitor(
    worker_state_provider=registry.snapshot,
    queue_lengths_provider=queue_state_snapshot,  # tùy chọn; tên queue -> int
    sink=resource_logger,
)
```

T07 cập nhật ready/running/unavailable bằng lifecycle thực tế. Busy vẫn
available nếu worker ready/healthy và có queue; không suy từ CPU/GPU utilization.
Idle có remaining=0 known; running mặc định remaining chưa biết. T11 có thể
cung cấp WorkerState có estimate known; T08 không tự ước lượng bằng utilization.
`memory_free_mb` trong WorkerState mà monitor trả về là bộ nhớ khả dụng của
thiết bị, không phải cấp phát riêng cho worker.

Monitor đọc state callback ngay khi read_sample/snapshot để không chờ tick
phần cứng kế tiếp; log lưu state_timestamp riêng với timestamp phần cứng.
Thiếu worker/queue source ghi not_configured. Harness T06 có registry nhưng
không giả số lượng queue bằng 0 khi T05 chưa có.

T13 nhận `monitor.snapshot()` cùng queued_jobs do T05 cung cấp. Controller cần
kiểm tra độ mới và worker capacity trước dispatch; thiếu telemetry không phải
mất năng lực inference. Không tính job đang chạy vào queued_jobs lần nữa.

T10 có thể triển khai `ResourceSampleSink.record_resource(ResourceSample)`;
`MetricsCollector.record(JobEvent)` không thay đổi. Không biến resource sample
thành JobEvent giả, không sửa Job.status trong T08.

## Kết quả thực tế

Lượt nghiệm thu bản cuối:
[summary.json](logs/t08/20261018T084314Z_0c742da6/summary.json),
[resources.csv](logs/t08/20261018T084314Z_0c742da6/resources.csv),
[monitor_summary.json](logs/t08/20261018T084314Z_0c742da6/monitor_summary.json).

- WSL2, 12 CPU logic, RAM guest khoảng 7604.64 MiB; NVIDIA GTX 1650 4096 MiB.
- Hai backend chạy đồng thời; mỗi device chạy 100 job/model, tổng **400 job**.
- Thu **39 mẫu**, tất cả 10 metric có giá trị hợp lệ trong 39/39 mẫu.
- Máy/driver này hỗ trợ cả GPU utilization qua NVML. Các máy WSL khác vẫn có
  thể không hỗ trợ; code và tests xử lý unavailable độc lập theo field.
- Writer ghi 39/39 mẫu, dropped=0; sampler/writer không lỗi, đều dừng sau run.
- Ghi nhận phase idle, prepare_cpu, prepare_gpu, inference và cooldown;
  30 mẫu quan sát được worker đang chạy model.
- Không có stale sample trong lượt này. Thời gian đọc mẫu lớn nhất 0.2424 s,
  có thể dài hơn interval 0.2 s; độ mới được đánh giá theo timestamp thực.
- RSS cao nhất quan sát được 1073.25 MiB; GPU used cao nhất 1022.95 MiB;
  GPU utilization cao nhất quan sát 47%. Đây không phải peak tuyệt đối hoặc
  tài nguyên quy riêng cho một job/model.

Lượt CPU-only telemetry trước đó PASS với GPU disabled. Lượt yêu cầu VRAM
trong sandbox bị chặn, giữ summary failed tại
`logs/t08/20261018T081923Z_809dd24f/`; chạy ngoài sandbox trên cùng phần cứng
đã PASS. Lỗi sandbox không bị chuyển thành số đo VRAM=0.

## Đo overhead

Lượt có 3 cặp đo:
[summary.json](logs/t08/20261018T082334Z_ad189e2f/summary.json).
Giữ nguyên model/input, sessions đã prepare, thread settings, 400 job/pass.
Đổi thứ tự baseline/monitored giữa các cặp. Thời gian prepare/prime và cooldown
không tính trong elapsed của workload; baseline không có sampler/writer hoạt động.
Các pass monitored có CSV riêng trong `overhead/<iteration>/`.

| Cặp | Baseline (s) | Có monitor (s) | Thay đổi elapsed |
|---|---:|---:|---:|
| 0 | 6.8916 | 6.4646 | -6.20% |
| 1 | 6.3551 | 6.2055 | -2.35% |
| 2 | 6.4086 | 6.6506 | +3.78% |

Median thay đổi -2.35%, khoảng dao động -6.20% đến +3.78%. Kết quả cho thấy
biến động hệ thống đáng kể; **không kết luận monitor làm inference nhanh hơn**
hoặc có overhead bằng 0. Đây là kiểm tra chi phí T08 ban đầu, chưa phải benchmark
T15; muốn kết luận chặt hơn cần nhiều lần lặp và kiểm soát tải nền/nhiệt độ.

Lượt đo overhead này thu 44 mẫu ở pass đầu; một lần đọc trong giai đoạn khởi
tạo mất khoảng 1.096 s. Bản cuối dùng timestamp đầu mẫu và có regression test
cho truy vấn chậm để không che giấu độ cũ của các metric được đọc trước đó.

## Kiểm thử và chạy lại

```bash
.venv/bin/python -m pip install -r requirements.txt

# Mặc định: CPU thật, fake NVML, regression; hai ca GPU được skip có lý do.
.venv/bin/python -m unittest discover -s tests -v

# Full acceptance: cả T06 CUDA inference và T08 VRAM phải chạy thật.
T06_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -v

# Monitor độc lập; --no-gpu dùng cho môi trường chỉ CPU.
.venv/bin/python scripts/T08_monitor_test.py --duration 10 --interval 0.2

# Nghiệm thu cùng T06.
.venv/bin/python scripts/T08_monitor_test.py --workload both --runs 100 --require-gpu-metrics

# Đo overhead theo cặp.
.venv/bin/python scripts/T08_monitor_test.py --workload both --runs 100 \
  --require-gpu-metrics --overhead-repeats 3
```

Đã chạy **62 tests PASS, không skip**, gồm 33 tests T04/T06 và 29 tests T08.
T08 bao gồm: 12 probe tests, 12 monitor/state/compatibility tests, 3 CSV tests,
2 real integration tests. Có tests cho unsupported query, quyền, driver/library
thiếu, invalid sentinel, prime counter, stale, sensor bị chặn, nhiều reader,
snapshot bất biến, sampler chết bất ngờ, writer lỗi/queue đầy và cleanup.
Concurrency tests dùng Event để điều khiển thay vì dựa vào sleep ngắn.
`pip check` không phát hiện dependencies hỏng.

Lệnh GPU inference yêu cầu backend thật, không skip/fallback. Telemetry mặc
định chấp nhận unavailable có nguyên nhân; `--require-gpu-metrics` riêng yêu
cầu VRAM hợp lệ, không ép utilization phải được hỗ trợ.

## Giới hạn còn lại

Chưa có queue/controller để nghiệm thu queue state và worker health của toàn
pipeline. Harness chỉ phát lifecycle facts quanh T06. Không cung cấp ước lượng
service time T11, không chọn device T13 hoặc tính job latency/throughput T10.
Sampling không ghi đủ biến động của từng inference ngắn. Không có GUI,
Prometheus hay power/temperature metrics trong scope này.

Tham chiếu API:
[psutil](https://psutil.readthedocs.io/stable/index.html),
[NVIDIA Python NVML binding](https://pypi.org/project/nvidia-ml-py/),
[NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html),
[WSL limitations](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#features-not-yet-supported).
