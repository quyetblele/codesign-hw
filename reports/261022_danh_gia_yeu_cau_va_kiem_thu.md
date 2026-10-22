# Đánh giá yêu cầu và khả năng kiểm thử của mã nguồn hiện tại

**Ngày kiểm tra:** 22/10/2026.  
**Commit:** `3d67759`.  
**Căn cứ:** mã trong `src/`, `scripts/`, `tests/`, yêu cầu trong `PLAN.md`, README và các báo cáo bàn giao. Không thay đổi mã nguồn ứng dụng hoặc bộ test trong lượt đánh giá này.

## 1. Kết quả kiểm tra mới nhất

| Lượt thực hiện | Kết quả | Phạm vi có thể xác nhận |
|---|---|---|
| Suite mặc định | 83 ca, 81 PASS, 2 SKIP; 3,084 giây | Unit tests, CPU inference thật, CPU/RAM monitor thật và controller dùng backend giả |
| Suite yêu cầu GPU trong sandbox | 81 PASS, 1 ERROR, 1 FAIL | CUDA báo failure 35, NVML báo `driver_unavailable`; chưa thể xác nhận GPU trong sandbox |
| Cùng suite yêu cầu GPU chạy ngoài sandbox | **83/83 PASS, không skip; 5,618 giây** | Cả CUDA inference/output comparison/operator trace và VRAM telemetry thật đều đạt |
| `scripts/demo_week2.py` | 4 accepted = 4 terminal; 3 completed, 1 failed cố ý | Demo điều phối bằng backend giả |
| Test stress nằm trong suite | 300 accepted = 282 completed + 18 failed cố ý | Đối soát job và dừng worker với backend giả |

Hai lỗi GPU trong sandbox không tái hiện khi chạy cùng bộ test ngoài sandbox; vì vậy không dùng chúng làm bằng chứng rằng driver máy thật không tương thích. Kết quả 83/83 ở đây là lượt mới ngày 22/10, không sao chép từ báo cáo lịch sử.

**83/83 PASS không có nghĩa đã hoàn tất dự án.** Các test hiện tại xác nhận từng module và controller giả lập; chưa có test ghép Controller với backend ONNX thật, monitor, profile đo thật và collector hoàn chỉnh.

## 2. Hiện tại có thể kiểm thử những gì?

### 2.1. Những test tự động đã có và đã chạy

| Nhóm | Số ca | Hành vi đang được kiểm tra | Loại bằng chứng |
|---|---:|---|---|
| `test_t04_contract.py` | 9 | Job hợp lệ/sai, lifecycle cơ bản, API scheduler/backend, nội dung worker event và lỗi | Contract, fake backend |
| `test_t06_preprocess.py` | 6 | Chuẩn hóa ImageNet, crop, grayscale/RGBA, ảnh hỏng, đường dẫn và symlink thoát root | Ảnh/file tạm thật |
| `test_t06_backend_unit.py` | 15 | Tái sử dụng session, prepare/close, metadata/output sai, provider thiếu, không fallback khi GPU lỗi, cleanup, xác thực trace | Mock runtime/session |
| `test_t06_backend_integration.py` | 3 | Hai model CPU thật, job lặp, input lỗi rồi chạy tiếp; CPU/GPU đồng thời, output tương đồng và CUDA compute trong trace | ONNX CPU/CUDA thật |
| `test_t08_probes.py` | 12 | Đơn vị, CPU đa core, prime counter, quyền, NVML unsupported/driver thiếu, sentinel sai, cleanup | Mock API phần cứng |
| `test_t08_resource_monitor.py` | 12 | Start/close/restart, cache bất biến, tuổi mẫu/stale, probe chậm/lỗi, nhiều reader, lỗi sink và worker registry | Fake sensor/clock, thread thật |
| `test_t08_resource_log.py` | 3 | CSV, null, flush/close, không ghi đè, queue đầy, dropped samples, writer lỗi | File/thread thật, lỗi mô phỏng |
| `test_t08_monitor_integration.py` | 2 | CPU/RAM/RSS thật; VRAM thật, GPU utilization có giá trị hoặc có lý do unavailable | Phần cứng thật |
| `test_week2_controller.py` | 6 | Round Robin, job thất bại, ID trùng, message sai thiết bị, dispatch lỗi, submit song song | Controller thật, backend/collector giả |
| `test_week2_hardening.py` | 15 | Shutdown với backlog/in-flight, timeout/retry, terminal trùng, collector/cleanup lỗi, nhiều producer, worker unavailable, profile giả, stress 300 job | Controller/queue/thread thật, backend giả |
| **Tổng** | **83** | Các hành vi hữu hạn được assert trong bộ test | Không phải benchmark hiệu năng |

GPU-only có lớp triển khai nhưng chưa có testcase chạy controller chuyên biệt sử dụng `GpuOnlyScheduler`. Việc import lớp này trong file test không đồng nghĩa đã kiểm tra policy đó. CPU-only có được dùng trong một số testcase; Round Robin được kiểm tra phân phối 30 job và nhiều kịch bản controller. Resource-aware hiện chỉ có một testcase controller chọn GPU theo profile giả của một model; chưa kiểm tra đầy đủ các nhánh quyết định.

### 2.2. Các harness hiện có có thể chạy thêm

| Script | Có thể kiểm tra/đo | Giới hạn |
|---|---|---|
| `T02_smoke_test.py` | Smoke test môi trường CPU/CUDA | Không qua queue/controller |
| `T03_model_test.py` | Input chuẩn và suy luận hai model | Không đo accuracy trên dataset có nhãn |
| `T06_worker_test.py` | Hai backend, output CPU/GPU, trace CUDA, execution mean/P95, log từng lượt | Không có queue latency hoặc so sánh scheduler |
| `T08_monitor_test.py` | Monitor độc lập hoặc theo dõi hai backend đồng thời; resource CSV và overhead bật/tắt monitor | Không chạy qua Controller; worker facts do harness cập nhật |
| `demo_week2.py` | Luồng queue → scheduler → worker → terminal | Backend giả, không có inference thật |

Các harness T02/T03/T06/T08 trong bảng là khả năng có sẵn của repository. Lượt đánh giá này chạy suite và demo; không chạy lại toàn bộ harness hoặc đo lại benchmark T06/T08.

## 3. Đã hoàn thành những yêu cầu nào?

Quy ước: **Đạt thành phần** nghĩa là có implementation và kiểm thử trong phạm vi module. **Một phần** nghĩa là đã có nền tảng nhưng chưa đáp ứng đầy đủ deliverable/nghiệm thu của task. **Chưa có** nghĩa là chưa tìm thấy implementation tương ứng trong checkout.

| Task theo PLAN | Đánh giá | Bằng chứng và phần còn thiếu |
|---|---|---|
| T01 — Chốt phạm vi và máy benchmark | Có tài liệu | Có scope/acceptance criteria và môi trường thử nghiệm. Không suy ra việc nhóm đã duyệt mọi quyết định từ mã nguồn |
| T02 — Môi trường CPU/CUDA | **Đạt thành phần** | Runtime/dependencies có sẵn; integration CPU/CUDA thật đạt trong lượt mới; có tài liệu và log môi trường trước đó |
| T03 — Hai model và input chuẩn | **Đạt thành phần** | MobileNetV2, ResNet-50, ảnh mẫu, preprocessing dùng chung, tests input/output |
| T04 — Interface giữa module | **Đạt phần code contract** | Job, JobEvent, WorkerMessage, Backend/Monitor/Collector/Controller protocols và 9 test. Việc phê duyệt hợp đồng giữa thành viên không được xác nhận bởi unittest |
| T05 — Queue và lifecycle | **Đạt trong phạm vi in-process đã test** | Incoming/CPU/GPU queues, chặn ID/dispatch trùng, terminal reconciliation, controller sở hữu chuyển trạng thái |
| T06 — CPU/GPU inference | **Đạt phần backend** | Prepare/warm-up/tái sử dụng session/infer/close; CPU/GPU thật, so output, trace compute. WorkerLoop có mã ở controller nhưng nghiệm thu ghép backend thật thuộc T07 |
| T07 — Pipeline CPU-only/GPU-only | **Một phần** | Có controller và worker threads; test controller dùng fake backend. Chưa có test/demo pipeline ONNX thật trên từng policy |
| T08 — Resource Monitor | **Đạt thành phần** | CPU/RAM/process/VRAM, cache, timestamp/stale, logger và tests thật/giả. Worker/queue callbacks chưa nối hoàn chỉnh vào controller |
| T09 — Generator và replay | **Chưa có** | Chưa thấy manifest generator/replayer cho low/medium/high/burst. Trường `arrival_offset_s` mới được lưu/validate, chưa được replay |
| T10 — Metrics Collector | **Chưa có implementation đầy đủ** | Có protocol, controller phát JobEvent và collector giả trong tests. CSV của harness T06/T08 chưa thay thế collector của pipeline |
| T11 — Profiling | **Một phần** | T06 có execution mean/P95 và ORT trace; chưa có bộ profile service-time/bộ nhớ model–device đo thật được nạp vào scheduler, chưa cung cấp remaining time |
| T12 — Round Robin | **Đạt logic trong phạm vi giả lập** | Luân phiên CPU/GPU được assert. Chưa có số liệu chạy cùng manifest và backend thật |
| T13 — Resource-aware | **Một phần** | Có estimated finish delay từ profile, queue và remaining time. Chưa hoàn thiện freshness, known flag, ràng buộc bộ nhớ, profile thật và decision log |
| T14 — Lỗi/đồng bộ/shutdown | **Một phần ở mức toàn hệ thống** | Nhiều test in-process đạt; backend thật có kiểm tra lỗi riêng. Chưa nghiệm thu drain + monitor + collector + ONNX thật trong một pipeline |
| T15 — Benchmark bốn scheduler | **Chưa có** | Thiếu replay, collector/profile và ma trận benchmark cùng điều kiện, ít nhất ba lần/mỗi cấu hình |
| T16 — Phân tích và biểu đồ | **Chưa đạt deliverable** | Có số liệu kiểm tra backend/monitor, chưa có bảng/biểu đồ so sánh scheduler đầy đủ |
| T17 — Báo cáo và demo cuối | **Một phần** | Có báo cáo module, báo cáo phân tích và demo giả lập. Chưa có báo cáo kết quả benchmark/demo pipeline thật hoàn chỉnh |

README/PLAN và một số báo cáo cũ còn ghi queue/controller/scheduler chưa có. Mã hiện tại đã chứa các phần đó sau merge, nên không thể dùng riêng những dòng mô tả tiến độ cũ để xác định hiện trạng. Ngược lại, việc đã merge hai nhóm module không đủ chứng minh chúng đã được ghép và kiểm chứng end-to-end.

## 4. Các điểm còn thiếu được xác nhận khi đọc và thử code

### 4.1. CPU-only vẫn phụ thuộc khởi tạo GPU nếu dùng hai backend thật

[controller.py](../src/edge_scheduler/controller.py) yêu cầu map backend có cả CPU và GPU; `start()` gọi `prepare()` cho cả hai rồi mới tạo worker. Vì vậy, chọn `CpuOnlyScheduler` chưa tạo ra chế độ chạy được trên máy chỉ có CPU với cấu hình toàn backend thật. Nếu GPU prepare lỗi, toàn controller start thất bại.

Điều này cũng ảnh hưởng so sánh RAM/VRAM: CPU-only hiện chỉ mô tả nơi job thực thi, không có nghĩa GPU không nạp model. Đây là giới hạn tích hợp cần công khai hoặc xử lý trước benchmark.

### 4.2. Monitor và controller chưa tự đồng bộ worker state

[monitoring_models.py](../src/edge_scheduler/monitoring_models.py) có WorkerRegistry với `set_ready`, `set_running`, `set_unavailable`. [controller.py](../src/edge_scheduler/controller.py) chỉ gọi `monitor.snapshot()`, chưa cập nhật registry hoặc tự start/close monitor.

Nếu gắn registry mới mà không có adapter cập nhật ready, registry vẫn báo worker unavailable và job có thể bị từ chối ở dispatch. Nếu không cấu hình worker callback, snapshot thiếu worker states; controller không tự dựng lại trạng thái sức khỏe thiết bị từ các worker đang chạy.

Queue length callback cũng chưa được nối trong một entrypoint pipeline. Cần thêm code tích hợp và test ready → running → ready/unavailable, queue state và vòng đời monitor/logger.

### 4.3. Resource-aware chưa kiểm soát đủ điều kiện tài nguyên

[schedulers.py](../src/edge_scheduler/schedulers.py) tính:

```text
finish_delay = remaining_time + tổng service time trong queue + service time job mới
```

Đã chạy thêm các probe trực tiếp ngoài unittest, không chỉnh sửa mã ứng dụng. Kết quả:

| Input thử | Kết quả thực tế | Ý nghĩa |
|---|---|---|
| Snapshot timestamp=0, VRAM free=0, GPU profile nhanh hơn | Chọn GPU | Chưa kiểm tra tuổi snapshot hoặc ràng buộc bộ nhớ |
| GPU có running model, remaining=0 và `estimated_remaining_known=False` | Vẫn chọn GPU theo profile nhanh hơn | Chưa phân biệt remaining chưa biết với remaining bằng 0 đã xác nhận |
| Hai worker unavailable, remaining đều 0 | Cả hai cost=Infinity; `choose_device()` trả CPU theo tie-break | Chưa có kết quả no-capacity riêng ở scheduler; controller sẽ đánh FAILED nếu nhận đúng snapshot unavailable này |
| Service-time profile chứa NaN hoặc Infinity | `ServiceTimeProfile.estimate()` vẫn trả giá trị đó | Validation hiện chỉ chặn `value <= 0`, chưa chặn số không hữu hạn |

Các probe trên chỉ xác nhận hành vi ở input cụ thể. Chưa có hệ thống tự đo lượng bộ nhớ cần cho mỗi job/model, nên bổ sung kiểm tra VRAM cần đi cùng profile bộ nhớ hoặc chính sách rõ ràng, không chỉ đặt ngưỡng tùy ý.

### 4.4. Chưa có thống kê hiệu năng của pipeline

Job có timestamps và controller có events, nhưng chưa có collector chung tính dispatch wait, device wait, end-to-end latency, P50/P95/P99, throughput, failure rate và scheduler overhead theo quy ước PLAN.

Mean/P95 của T06 chỉ đo execution backend. Summary T08 thống kê resource samples và elapsed của harness, không thay thế benchmark scheduler.

### 4.5. Chưa nghiệm thu vận hành lâu dài

Job queues chưa bounded; `Controller.jobs`, `events` và các tập ID của queue giữ lịch sử trong RAM. Chưa có test soak kéo dài hoặc chính sách giải phóng lịch sử. Vì vậy, test stress 300 job đạt không đủ khẳng định bộ nhớ ổn định dưới tải liên tục nhiều giờ.

Tương tự, chưa kiểm tra khả năng phục hồi sau kill process/mất điện; đây không phải tính năng bắt buộc của MVP đã chốt, không nên tính là lỗi khiến MVP chưa hoàn thành.

## 5. Những test cần bổ sung để nghiệm thu phần còn lại

Theo thứ tự phụ thuộc:

1. **Controller + backend thật:** chạy CPU-only/GPU-only/Round Robin với hai model và ảnh thật; xác nhận output, thiết bị, CUDA trace, timestamps, terminal đúng một lần và workers dừng. GPU-only cần testcase chuyên biệt.
2. **Monitor tích hợp:** cập nhật registry theo lifecycle worker, queue callback, start/close monitor/logger; thử thiếu telemetry, stale snapshot, probe chậm và worker unavailable.
3. **Resource-aware:** queue rỗng/không rỗng, hai model, remaining đã biết/chưa biết, tie-break, thiếu profile, NaN/Infinity, bộ nhớ không đủ, cả hai thiết bị không khả dụng; kiểm tra lựa chọn với profile đo thật.
4. **Generator/replay:** cùng seed sinh cùng manifest; replay arrival độc lập với thời gian hoàn thành; đo trễ replay; không mất/trùng job ở low/medium/high/burst.
5. **Collector:** dữ liệu thời gian mẫu có kết quả tính tay để đối chiếu; job failed trước STARTED; không tính mẫu thiếu như 0; CSV/summary khớp tổng job, flush và lỗi ghi file được báo.
6. **Shutdown toàn pipeline:** còn job chờ/đang infer thật, shutdown đồng thời submit, backend/collector/logger lỗi; đối soát và xác nhận thread/session/log đều kết thúc đúng.
7. **Benchmark lặp:** bốn scheduler × bốn mức tải × ít nhất ba lần, cùng manifest/cấu hình; lưu config/profile/log và phân tích độ biến động. Không mặc định Resource-aware phải thắng.

Các mục 1–3 có thể bắt đầu ngay bằng việc nối các module hiện có. Các mục 4–5 cần implementation mới trước khi có thể nghiệm thu đầy đủ. Benchmark cuối phụ thuộc các bước trước.

## 6. Lệnh kiểm thử có sẵn

Chạy tại project root với `.venv` đã cài dependencies:

```bash
# Mặc định: CPU thật + unit/fake tests; bỏ qua hai ca GPU thật.
.venv/bin/python -m unittest discover -s tests -v

# Suite yêu cầu cả CUDA inference lẫn telemetry GPU thật.
T06_TEST_GPU=1 T08_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -v

# Chỉ queue/controller/shutdown dùng backend giả.
.venv/bin/python -m unittest discover -s tests -p 'test_week2_*.py' -v

# Chỉ backend/preprocessing, yêu cầu GPU thật cho integration CUDA.
T06_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -p 'test_t06_*.py' -v

# Chỉ resource monitor/logging, yêu cầu telemetry GPU thật.
T08_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -p 'test_t08_*.py' -v

# Demo giả lập.
PYTHONPATH=src .venv/bin/python scripts/demo_week2.py

# Harness backend và monitor thật; tạo artifact riêng trong logs/.
.venv/bin/python scripts/T06_worker_test.py --device both --require-gpu --profile
.venv/bin/python scripts/T08_monitor_test.py --workload both --runs 100 --require-gpu-metrics
```

Suite GPU cần truy cập được driver CUDA/NVML. Một lượt bị sandbox chặn không được biến thành PASS hoặc âm thầm chuyển GPU sang CPU. Chưa có CLI `run.py --scheduler ...` để chạy benchmark end-to-end.

## 7. Kết luận về mức sẵn sàng

**Sẵn sàng kiểm thử và demo ở cấp thành phần:** backend CPU/CUDA, preprocessing, resource monitor, resource CSV, contract; queue/controller/worker/synchronization/shutdown đã có kiểm thử giả lập tương đối rộng.

**Chưa đủ nghiệm thu mục tiêu toàn đề tài:** thiếu pipeline tích hợp thật có collector/replay, Resource-aware hoàn thiện và benchmark so sánh có thể lặp lại. Ưu tiên tiếp theo là T07 + nối T08, rồi T09/T10/T11 và hoàn thiện T13/T14 trước T15.

Không đưa ra phần trăm hoàn thành vì các task có khối lượng khác nhau và chưa có trọng số nghiệm thu thống nhất.

## 8. Nguồn đối chiếu

- [PLAN.md](../PLAN.md): scope, T01–T17, benchmark và acceptance criteria.
- [README.md](../README.md), [WEEK2_IMPLEMENTATION_REPORT.md](../WEEK2_IMPLEMENTATION_REPORT.md): mô tả và phạm vi bàn giao; tiến độ trong README có phần đã cũ.
- [controller.py](../src/edge_scheduler/controller.py), [queues.py](../src/edge_scheduler/queues.py), [schedulers.py](../src/edge_scheduler/schedulers.py), [contracts.py](../src/edge_scheduler/contracts.py): implementation điều phối và contract.
- [resource_monitor.py](../src/edge_scheduler/resource_monitor.py), [monitoring_models.py](../src/edge_scheduler/monitoring_models.py): snapshot, callbacks và worker registry.
- [tests/](../tests/), [scripts/](../scripts/): các testcase/harness đã kiểm tra.
- [T06_report.md](../T06_report.md), [T08_report.md](../T08_report.md): log nghiệm thu lịch sử, khác với lượt suite mới ngày 22/10.
