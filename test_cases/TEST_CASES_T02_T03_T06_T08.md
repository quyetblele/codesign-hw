# Tổng hợp test case T02, T03, T06 và T08

Ngày tổng hợp: **20/10/2026**.

Tài liệu tổng hợp các kiểm thử đã thực hiện, đối chiếu với test code, báo cáo
và artifact trong repository. **Không chạy lại inference/benchmark khi lập
tài liệu ngày 20/10/2026**; trạng thái bên dưới là kết quả lịch sử. Lượt full
acceptance gần nhất được ghi nhận ngày **18/10/2026**.

## 1. Phạm vi, trạng thái và bằng chứng

| Yêu cầu | Mục tiêu kiểm thử | Nguồn kết quả |
|---|---|---|
| T02 | Môi trường WSL/Python/ONNX Runtime, chạy MobileNetV2 CPU/CUDA | [T02_report.md](../T02_report.md), [environment log](../logs/t02_environment.txt), [script T02](../scripts/T02_smoke_test.py) |
| T03 | Hai model, ảnh mẫu, preprocessing và output | [T03_report.md](../T03_report.md), [script T03](../scripts/T03_model_test.py), tests preprocessing dùng chung |
| T06 | Backend lifecycle, session reuse, input/output, xử lý lỗi, CUDA thật | [T06_report.md](../T06_report.md), tests T06, log nghiệm thu T06 |
| T08 | Probe, cache, worker state, stale, concurrency, CSV, resource thật, overhead | [T08_report.md](../T08_report.md), tests T08, log nghiệm thu T08 |

Quy ước:

- **PASS**: ca đã đạt trong lượt kiểm thử được báo cáo; không có nghĩa vừa chạy lại ngày tổng hợp.
- **MOCK/FAKE**: giả lập session, sensor hoặc lỗi để kiểm tra hành vi; không phải bằng chứng inference hay telemetry GPU thật.
- **REAL**: dùng model/phần cứng/hệ điều hành thật.
- **BLOCKED — môi trường**: lượt chạy không hoàn tất vì giới hạn môi trường. Giữ nguyên trạng thái `failed` trong artifact; không đổi thành PASS.
- **SKIP**: test GPU không được chạy do chưa bật biến môi trường; không tính là PASS GPU.
- Ca âm tính PASS khi chương trình từ chối/xử lý đúng dữ liệu hoặc lỗi đã đưa vào.

### 1.1. Số lượng test tự động

| Nhóm | Số phương thức test duy nhất | Ghi chú |
|---|---:|---|
| Preprocessing/path dùng chung T03/T06 | 6 | Nằm trong `test_t06_preprocess.py`, liệt kê một lần ở mục T03 |
| T06 backend unit | 15 | ONNX session/provider được mock |
| T06 backend integration | 3 | Hai CPU tests và một GPU test |
| T08 probes | 12 | psutil/NVML được mock |
| T08 monitor/state/compatibility | 12 | Fake probe, clock, Event điều khiển concurrency |
| T08 CSV logger | 3 | File tạm thật; có mô phỏng writer chậm/lỗi |
| T08 integration | 2 | CPU/RAM/process thật và GPU telemetry thật |
| **Tổng trực tiếp T02/T03/T06/T08** | **53** | T02 được kiểm tra bằng smoke script/báo cáo, không có unittest riêng |
| T04 regression ngoài phạm vi chi tiết | 9 | `test_t04_contract.py` |
| **Bộ full acceptance đã chạy** | **62** | **62 PASS, không SKIP** theo T08 report ngày 18/10/2026 |

Các kịch bản smoke/CLI ở dưới là kiểm tra bổ sung, không cộng vào số phương
thức unittest. Một phương thức có nhiều `subTest` vẫn chỉ được tính một lần.

### 1.2. Môi trường và dữ liệu đã dùng

| Hạng mục | Cấu hình đã ghi nhận |
|---|---|
| OS | Ubuntu 22.04.5 / WSL2 |
| Python / ONNX Runtime | Python 3.10.12 / `onnxruntime-gpu` 1.23.2 |
| GPU | NVIDIA GeForce GTX 1650, 4096 MiB VRAM |
| CPU/RAM trong log T08 | 12 CPU logic; RAM guest khoảng 7604.64 MiB |
| Thư viện | NumPy 2.2.6, Pillow 12.3.0; T08: psutil 7.2.2, nvidia-ml-py 13.615.71 |
| Model IDs | `mobilenetv2`, `resnet50` |
| Model files | `models/mobilenetv2-12.onnx`, `models/resnet50-v1-12.onnx` |
| Ảnh thật | `inputs/test.jpg` |
| Tensor đầu vào | NCHW `(1, 3, 224, 224)`, float32, batch 1 |
| Output | `(1, 1000)`; T06 kiểm tra float32 hữu hạn |
| Cấu hình CLI T06/T08 | 5 warm-up/model/device; intra-op 2, inter-op 1; ORT sequential; tắt intra-op spinning |
| Integration tests T06 | 1 warm-up/model/device |
| So CPU/GPU | `assert_allclose(rtol=1e-4, atol=1e-4)` |
| Sampling T08 CLI | 0.2 s/mẫu; stale sau 1 s |

CPU/RAM là tài nguyên OS/WSL nhìn thấy. RSS thuộc toàn process. VRAM là toàn
GPU được chọn và có thể gồm ứng dụng khác. Các trường `_mb` của T08 dùng MiB.

## 2. T02 — Setup môi trường và smoke test CPU/CUDA

### 2.1. Điều kiện và cách thực hiện

Môi trường `.venv` đã có ONNX Runtime, model MobileNetV2 và thư viện CUDA/cuDNN
phù hợp. Chạy các lệnh từ root project; GPU cần quyền truy cập driver thực tế.

```bash
.venv/bin/python --version
nvidia-smi
.venv/bin/python -c 'import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())'
.venv/bin/python scripts/T02_smoke_test.py
```

Script T02 dùng tensor ngẫu nhiên có seed 42, 5 warm-up và 20 lượt đo/provider.
Script in kết quả để đối chiếu; chưa có assertion tolerance hoặc kiểm tra
trace CUDA chặt như T06. Khi nghiệm thu lại, cần kiểm tra output/log và dùng
T06 để bổ sung bằng chứng compute CUDA.

### 2.2. Các ca đã kiểm tra

| ID | Mục tiêu / loại | Dữ liệu và bước thực hiện | Kết quả mong đợi | Kết quả đã ghi nhận / bằng chứng |
|---|---|---|---|---|
| T02-SM01 | Môi trường và provider — REAL | Kiểm tra OS, Python, `nvidia-smi`, version/providers của ORT | Nhận GPU; có CPU/CUDA provider; ghi lại cấu hình | PASS theo T02 report/environment log: Python 3.10.12, ORT 1.23.2, GTX 1650; danh sách có CPU/CUDA |
| T02-SM02 | Model ONNX hợp lệ — kiểm tra được báo cáo ở T02 | Nạp và kiểm tra `mobilenetv2-12.onnx` bằng ONNX checker | Model hợp lệ | PASS theo T02 report: `ONNX model OK`, IR version 7. Bước checker không nằm trong smoke script hiện tại |
| T02-SM03 | CPU inference — REAL | Tạo CPU session; chạy tensor `(1,3,224,224)` qua warm-up và 20 lượt đo | Session CPU; output `(1,1000)`; không lỗi | PASS theo T02 report; mean 4.150 ms, min 3.038 ms, max 7.299 ms |
| T02-SM04 | CUDA inference — REAL | Tạo CUDA session; chạy cùng tensor và số lượt như CPU | CUDA khởi tạo/chạy được; output `(1,1000)` | PASS theo T02 report; mean 3.469 ms, min 2.690 ms, max 5.247 ms. Bằng chứng trace được bổ sung ở T06 |
| T02-SM05 | Đối chiếu CPU/CUDA — REAL | Lấy output của hai provider, tính max absolute difference | Output gần tương đương; ghi sai khác | PASS theo đối chiếu T02 report: max diff `4.7683716e-06`; smoke script chỉ in sai khác, không tự assert tolerance |

Không dùng riêng `get_available_providers()` để kết luận GPU đã thực thi.
`CUDA UMD Version` trong environment log không được coi là phiên bản mọi
CUDA runtime package đã cài trong `.venv`.

## 3. T03 — Hai model và preprocessing dùng chung

### 3.1. Smoke test model thật

```bash
.venv/bin/python scripts/T03_model_test.py
```

Pipeline: ảnh → RGB → resize cạnh ngắn 256, giữ tỉ lệ → center crop 224 →
float32/255 → ImageNet normalization → CHW → batch → contiguous.

| ID | Mục tiêu / loại | Dữ liệu và bước thực hiện | Kết quả mong đợi | Kết quả đã ghi nhận |
|---|---|---|---|---|
| T03-SM01 | Chuẩn hóa ảnh thật — REAL | Preprocess `inputs/test.jpg` | Tensor `(1,3,224,224)`, float32 | PASS theo T03 report và lượt chạy script khi triển khai T06 |
| T03-SM02 | MobileNetV2 — REAL CPU | Nạp model, đọc tên input từ metadata, chạy tensor chung | Feed đúng input; output `(1,1000)` | PASS; input name `input`, Top-1 quan sát được 3 |
| T03-SM03 | ResNet-50 — REAL CPU | Nạp model, đọc tên input từ metadata, chạy cùng tensor | Feed đúng input; output `(1,1000)` | PASS; input name `data`, Top-1 quan sát được 3 |

Top-1=3 là kết quả quan sát trên ảnh mẫu, không phải nhãn ground truth để
nghiệm thu độ chính xác phân loại. Script T03 in thông số; các assertion nằm
trong unit/integration tests dưới đây và ở T06.

### 3.2. Unit tests preprocessing/path — dùng chung T03/T06

Code: [test_t06_preprocess.py](../tests/test_t06_preprocess.py), class
`PreprocessTest`. Ảnh được tạo trong thư mục tạm; không cần GPU.

```bash
.venv/bin/python -m unittest discover -s tests -p test_t06_preprocess.py -v
```

Tất cả 6 ca dưới đây **PASS trong lượt full acceptance ngày 18/10/2026**.

| ID | Phương thức test | Input / thao tác | Kết quả mong đợi |
|---|---|---|---|
| T03-UT01 | `test_constant_color_matches_imagenet_normalization` | Ảnh RGB 400×280, màu `(255,128,0)`; preprocess | Shape đúng, float32, contiguous, hữu hạn; từng kênh đúng ImageNet normalization với `atol=1e-6` |
| T03-UT02 | `test_center_crop_preserves_center_for_both_orientations` | Ảnh 512×256 và 256×512, vùng trắng 224×224 ở giữa | Crop đúng vùng giữa ở cả hai hướng; giá trị hai góc khớp màu trắng sau normalization |
| T03-UT03 | `test_grayscale_and_rgba` | Ảnh grayscale và RGBA 180×300 | Chuyển RGB đúng; pixel kiểm tra khớp normalization |
| T03-UT04 | `test_corrupt_image` | File `.jpg` chứa bytes không phải ảnh | Raise `UnidentifiedImageError` |
| T03-UT05 | `test_repository_path_validation` | File hợp lệ; sau đó path rỗng, khoảng trắng, absolute, `../`, `None`, file thiếu, thư mục | Resolve file hợp lệ; input sai raise `ValueError`; file thiếu/thư mục raise `FileNotFoundError` |
| T03-UT06 | `test_symlink_cannot_escape_root` | Symlink trong root trỏ tới file ngoài root | Raise `ValueError`, không cho truy cập file ngoài root |

Implementation dùng chung hiện ở `src/edge_scheduler/preprocess.py`;
`scripts/preprocess.py` giữ compatibility import.

## 4. T06 — Backend CPU/GPU

### 4.1. Unit tests backend — MOCK/FAKE

Code: [test_t06_backend_unit.py](../tests/test_t06_backend_unit.py), class
`BackendUnitTest`. Mỗi ca dùng root tạm, hai file model giả và ảnh PNG;
`InferenceSession`, available providers và CUDA preload đều được mock.

```bash
.venv/bin/python -m unittest discover -s tests -p test_t06_backend_unit.py -v
```

Tất cả 15 ca dưới đây **PASS trong lượt full acceptance ngày 18/10/2026**.

| ID | Phương thức test | Input / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|
| T06-UT01 | `test_protocol_and_fixed_devices` | Tạo CPUBackend và GPUBackend | Đúng `InferenceBackend`; device cố định CPU/GPU |
| T06-UT02 | `test_prepare_once_and_reuse_sessions_with_correct_input_names` | Prepare hai lần; chạy xen kẽ hai model, 3 job/model | Chỉ tạo 2 session; đúng tên `input`/`data`; 1 warm-up + 3 run/session; fallback bị tắt; CPU không preload CUDA |
| T06-UT03 | `test_lifecycle_and_reprepare_after_close` | Infer trước prepare; prepare; close hai lần; infer; prepare lại | Chặn infer khi chưa ready; close idempotent; giải phóng session; prepare lại tạo session mới |
| T06-UT04 | `test_gpu_provider_absent` | Runtime giả chỉ có CPU provider; prepare GPU | Raise lỗi CUDA unavailable; không tạo session; backend không ready |
| T06-UT05 | `test_gpu_session_fallback_is_rejected` | Yêu cầu GPU nhưng session trả provider CPU | Raise lỗi fallback rejected; backend không ready, không giữ session |
| T06-UT06 | `test_partial_prepare_failure_cleans_up_and_can_retry` | MobileNet tạo được; ResNet ném lỗi; sau đó bỏ lỗi và retry | Cleanup session/profile đã tạo; không ready sau lỗi; lần prepare kế tiếp thành công |
| T06-UT07 | `test_missing_model_does_not_create_sessions` | Xóa file ResNet giả trước prepare | Raise lỗi file not found; không tạo session nào |
| T06-UT08 | `test_bad_metadata_is_rejected` | Metadata input NHWC hoặc tensor double | Raise lỗi invalid; backend không ready |
| T06-UT09 | `test_bad_requests_do_not_poison_next_job` | Model ID sai/None/list; file thiếu; traversal; ảnh hỏng; cuối cùng input hợp lệ | Từng request sai raise đúng lỗi; job hợp lệ sau đó vẫn chạy |
| T06-UT10 | `test_input_root_is_independent_of_cwd` | Prepare rồi đổi cwd sang thư mục models trước infer | Resolve ảnh theo root đã cấu hình và infer thành công |
| T06-UT11 | `test_gpu_run_failure_is_not_retried_on_cpu` | GPU device_id=2; prepare; session.run ném `CUDA device lost` | Preload đúng; device_id truyền đúng; lỗi có context GPU/model; không retry, không tạo session mới/chuyển provider |
| T06-UT12 | `test_invalid_output_is_rejected` | Output sai shape, float64, NaN hoặc chuỗi | Infer raise lỗi, không trả output không hợp lệ |
| T06-UT13 | `test_profile_finalization_failure_still_releases_sessions` | Finalize profile session đầu bị disk error; close | Báo lỗi; vẫn finalize session còn lại; bỏ mọi session; close lại an toàn |
| T06-UT14 | `test_invalid_configuration` | Warm-up 0/True, threads 0, device_id -1 | Constructor raise `ValueError` |
| T06-UT15 | `test_trace_requires_cuda_compute_not_copy_or_provider_list` | Trace chỉ có CUDA session/Memcpy và CPU Conv; sau đó thêm CUDA Conv | Trace đầu không verified; chỉ verified khi có CUDA compute thật trong dữ liệu trace |

### 4.2. Integration tests model thật

Code: [test_t06_backend_integration.py](../tests/test_t06_backend_integration.py).
Hai model và ảnh mẫu phải tồn tại. Tất cả 3 ca đã **PASS trên CPU/GPU thật**
trong lượt full acceptance ngày 18/10/2026.

| ID | Class / phương thức test | Bước thực hiện | Kết quả mong đợi |
|---|---|---|---|
| T06-IT01 | `CPUIntegrationTest.test_both_models_and_repeated_mixed_jobs` | CPU warm-up 1; xen kẽ hai model trong 3 vòng | Output `(1,1000)`, float32, hữu hạn; các lượt lặp khớp với `rtol=1e-5`, `atol=1e-5` |
| T06-IT02 | `CPUIntegrationTest.test_bad_input_then_success` | Infer ResNet với file thiếu, rồi ảnh mẫu | Request lỗi raise `FileNotFoundError`; request kế tiếp thành công |
| T06-IT03 | `GPUIntegrationTest.test_both_models_match_cpu_and_execute_cuda_compute` | Prepare CPU/GPU; dùng 2 thread chạy hai backend đồng thời, 2 vòng/model; đóng và đọc trace GPU | Output CPU/GPU khớp `rtol=atol=1e-4`; có trace cho cả hai model; mỗi model có CUDA compute verified |

```bash
# CPU integration; GPU case SKIP nếu chưa bật T06_TEST_GPU.
.venv/bin/python -m unittest discover -s tests -p test_t06_backend_integration.py -v

# Yêu cầu CUDA thật, không skip nếu GPU lỗi.
T06_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -p test_t06_backend_integration.py -v
```

### 4.3. Nghiệm thu CLI và kết quả đã lưu

```bash
.venv/bin/python scripts/T06_worker_test.py --device both --require-gpu --profile
```

Nguồn bằng chứng **E06**:
[summary.json](../logs/t06/20261018T033611Z_865ab910/summary.json),
[jobs.csv](../logs/t06/20261018T033611Z_865ab910/jobs.csv),
[trace directory](../logs/t06/20261018T033611Z_865ab910/profiles/).

| ID | Kịch bản / loại | Kết quả mong đợi | Kết quả thực tế |
|---|---|---|---|
| T06-AC01 | 4 tổ hợp hai model × CPU/GPU — REAL | Mỗi tổ hợp chạy 10 job sau 5 warm-up; output đúng | PASS, E06 `status=passed`; tổng 40 job |
| T06-AC02 | So output CPU/GPU — REAL | `assert_allclose(rtol=atol=1e-4)` đạt cho cả hai model | PASS; MobileNet max diff `8.106231689453125e-06`; ResNet max diff `1.33514404296875e-05` |
| T06-AC03 | Xác nhận compute CUDA — REAL | Trace mỗi model có Conv/Gemm/MatMul gán CUDA, không chỉ Memcpy/provider list | PASS; MobileNet 780 CUDA Conv + 15 Gemm; ResNet 795 Conv + 15 Gemm; bao gồm warm-up |
| T06-ENV01 | Chạy cùng lệnh trong sandbox | Nếu không truy cập CUDA được: báo lỗi, không chuyển inference GPU sang CPU | Lượt thực thi BLOCKED — môi trường; artifact `failed` ở [summary lỗi](../logs/t06/20261018T033454Z_a6f97d46/summary.json). Cùng lệnh ngoài sandbox PASS ở E06 |

Thời gian quan sát từ E06: MobileNet CPU/GPU mean 12.654/14.484 ms;
ResNet CPU/GPU mean 51.274/16.578 ms. Có preprocessing và profiling overhead;
không dùng các số này làm ngưỡng PASS hoặc kết luận GPU luôn nhanh hơn CPU.

## 5. T08 — Resource Monitor

### 5.1. Probe unit tests — MOCK

Code: [test_t08_probes.py](../tests/test_t08_probes.py). Không yêu cầu GPU thật.
Tất cả 12 ca **PASS trong lượt full acceptance ngày 18/10/2026**.

```bash
.venv/bin/python -m unittest discover -s tests -p test_t08_probes.py -v
```

| ID | Class / phương thức test | Input / thao tác | Kết quả mong đợi |
|---|---|---|---|
| T08-P01 | `SystemProbeTest.test_units_available_ram_and_multicore_process_cpu` | RAM available 3000 MiB, free 100 MiB; RSS 120 MiB; process CPU 240% | Dùng available=3000, đổi đơn vị đúng; giữ process CPU 240%; CPU API luôn nonblocking `interval=None` |
| T08-P02 | `SystemProbeTest.test_failed_prime_does_not_publish_first_zero` | Prime lỗi; lần sau CPU API trả 0 rồi 20 | Mẫu đầu sau lỗi là None/warming_up; mẫu sau là 20; không coi zero đầu là số đo hợp lệ |
| T08-P03 | `SystemProbeTest.test_memory_failure_preserves_cpu` | virtual_memory ném PermissionError | RAM None/permission_denied; CPU vẫn có giá trị |
| T08-P04 | `SystemProbeTest.test_process_permission_and_exit_are_field_errors` | RSS bị AccessDenied; process CPU bị NoSuchProcess | Có reason riêng cho từng field; RAM hệ thống vẫn hợp lệ |
| T08-P05 | `SystemProbeTest.test_invalid_values_are_not_fabricated_zero` | CPU NaN; process CPU âm | Field None/invalid_value, không gán 0 giả |
| T08-P06 | `GPUProbeTest.test_read_units_metadata_and_cleanup` | NVML trả total/used/free 4096/1024/3072 MiB, util 50%, tên dạng bytes; gọi start/close lặp | Chuyển đơn vị/metadata đúng; init/shutdown chỉ một lần |
| T08-P07 | `GPUProbeTest.test_unsupported_utilization_keeps_memory` | Utilization ném NotSupported | Utilization None/not_supported; VRAM free vẫn 3072 MiB |
| T08-P08 | `GPUProbeTest.test_memory_failure_keeps_utilization` | Memory query ném NoPermission | VRAM None/permission_denied; utilization vẫn 50% |
| T08-P09 | `GPUProbeTest.test_driver_or_library_missing_does_not_raise` | Init NVML ném DriverNotLoaded hoặc LibraryNotFound | GPU metrics None với reason tương ứng; không crash, không shutdown nếu chưa init |
| T08-P10 | `GPUProbeTest.test_uuid_selection_and_handle_failure_cleanup` | Chọn UUID; lấy handle ném NotFound sau init | Đã gọi lookup đúng UUID; ghi device_not_found; vẫn shutdown NVML |
| T08-P11 | `GPUProbeTest.test_disabled_does_not_initialize` | GPUProbe enabled=False | Các field disabled; không gọi NVML init |
| T08-P12 | `GPUProbeTest.test_invalid_memory_sentinel_is_unavailable` | VRAM free=`2**64-1` lớn hơn total | Free None, có error; không ghi sentinel thành số đo |

### 5.2. Monitor, trạng thái, concurrency và compatibility — FAKE

Code: [test_t08_resource_monitor.py](../tests/test_t08_resource_monitor.py),
class `MonitorTest`. Fake sensor, clock, callback hoặc sink; dùng Event điều
khiển concurrency. Tất cả 12 ca **PASS ngày 18/10/2026**.

```bash
.venv/bin/python -m unittest discover -s tests -p test_t08_resource_monitor.py -v
```

| ID | Phương thức test | Input / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|
| T08-M01 | `test_protocol_initial_sample_and_legacy_worker_constructor` | Đọc monitor chưa start; tạo WorkerState bằng 5 positional args cũ | Đúng protocol; sequence=0, stale=True, CPU None; các field mới có default tương thích |
| T08-M02 | `test_idempotent_lifecycle_restart_and_same_sampling_thread` | Start hai lần; close hai lần; start lại | Không tạo sampler trùng; prime/read cùng thread nền; close đúng một lần/vòng đời |
| T08-M03 | `test_timestamp_age_and_stale_do_not_relabel_old_data` | Lấy mẫu tại clock=100; dừng; đổi clock=102 | Timestamp mẫu không đổi; age=2 s và stale=True |
| T08-M04 | `test_live_worker_state_and_immutable_copies` | Worker ready→running, queue source CPU=2; đọc; sau đó unavailable; thử sửa mapping | Busy vẫn available; remaining chưa known; nguồn state mới phản ánh ngay; bản cũ không đổi; mappings bất biến |
| T08-M05 | `test_probe_failure_preserves_other_probe_and_reports_null` | System probe ném lỗi; GPU probe hợp lệ | CPU None + reason; GPU memory vẫn có; sampler không mất cả hai nhóm |
| T08-M06 | `test_slow_probe_does_not_refresh_older_metrics_timestamp` | Fake clock tăng 100→102 trong GPU read | Timestamp giữ 100; duration=2 s; sample stale, không giả dữ liệu mới |
| T08-M07 | `test_unexpected_sampler_failure_is_visible_and_probes_close` | Clock ném lỗi ngoài logic probe | Không có sample hợp lệ; diagnostics chứa lỗi; cả hai probe được close |
| T08-M08 | `test_failed_sources_do_not_invent_worker_or_queue_state` | Worker source lỗi; queue source trả -1 | Không tạo worker/queue state giả; có source_error; CPU metric vẫn hợp lệ |
| T08-M09 | `test_snapshot_readers_do_not_wait_for_a_blocked_probe` | Event giữ sensor đang đọc; 4 reader threads gửi 20 snapshot calls | Các reader trả snapshot ban đầu trong timeout dù sensor chưa nhả; nhả sensor thì monitor tiếp tục có mẫu |
| T08-M10 | `test_sink_failure_is_visible_without_stopping_sampling` | Sink ném disk full | Lỗi xuất hiện ở resource_log; sampler_error vẫn None |
| T08-M11 | `test_registry_requires_ready_and_idle_estimate_is_known` | Gọi running trước ready; sau đó set_ready | Chặn running trước ready; idle remaining được đánh dấu known |
| T08-M12 | `test_bad_configuration` | interval=0 hoặc max_age=NaN | Constructor raise ValueError |

### 5.3. CSV logger — file tạm thật và lỗi được mô phỏng

Code: [test_t08_resource_log.py](../tests/test_t08_resource_log.py), class
`LogTest`. Tất cả 3 ca **PASS ngày 18/10/2026**.

```bash
.venv/bin/python -m unittest discover -s tests -p test_t08_resource_log.py -v
```

| ID | Phương thức test | Input / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|
| T08-L01 | `test_schema_nulls_flush_close_and_no_overwrite` | Ghi 1 sample có field None và reason; close lặp; đọc CSV; ghi sau close; mở lại cùng path | Schema/JSON đúng; None thành ô rỗng; written=1; chặn ghi sau close và chặn overwrite; đúng ResourceSampleSink |
| T08-L02 | `test_bounded_queue_reports_drops_and_drains_on_close` | Capacity=1; dùng Event giữ writer; gửi thêm mẫu vượt capacity rồi nhả | dropped=1; close drain; written=2; không chặn producer khi đầy |
| T08-L03 | `test_writer_failure_surfaces` | Hàm tạo row ném OSError disk full | close raise lỗi có nguyên nhân; logger.stats.error có giá trị |

### 5.4. Integration tests tài nguyên thật

Code: [test_t08_monitor_integration.py](../tests/test_t08_monitor_integration.py),
class `MonitorIntegrationTest`. Hai ca **PASS trên môi trường thật ngày
18/10/2026**; GPU utilization là metric tùy khả năng hỗ trợ.

| ID | Phương thức test | Bước thực hiện | Kết quả mong đợi |
|---|---|---|---|
| T08-IT01 | `test_real_cpu_ram_and_process_with_gpu_disabled` | Sampler interval 0.1 s; GPU disabled; đợi mẫu rồi close | RAM available/RSS >0; CPU trong [0,100]; GPU reason disabled; thread dừng, không sampler error |
| T08-IT02 | `test_real_gpu_memory_utilization_optional_on_wsl` | Sampler interval 0.1 s, NVML thật; đọc mẫu rồi close | VRAM total >0; utilization trong [0,100] nếu có, hoặc None kèm error nếu không hỗ trợ |

```bash
T08_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -p test_t08_monitor_integration.py -v
```

Nếu không đặt `T08_TEST_GPU`, test GPU T08 dùng giá trị `T06_TEST_GPU` làm
mặc định. Đặt `T08_TEST_GPU=0` sẽ vô hiệu riêng GPU test T08 ngay cả khi
`T06_TEST_GPU=1`; cần kiểm tra biến môi trường trước khi nghiệm thu đầy đủ.

### 5.5. Harness T08 cùng T06, logging và overhead

Các lệnh đã dùng cho các lượt có artifact:

```bash
# T08-AC01: standalone, không yêu cầu GPU.
.venv/bin/python scripts/T08_monitor_test.py --duration 1 --no-gpu

# T08-AC02: monitor CPU/GPU thật, 100 job/model/device.
.venv/bin/python scripts/T08_monitor_test.py --workload both --runs 100 --require-gpu-metrics

# T08-AC03: 3 cặp baseline/monitored, đổi thứ tự giữa các cặp.
.venv/bin/python scripts/T08_monitor_test.py --workload both --runs 100 \
  --require-gpu-metrics --overhead-repeats 3

# T08-ENV01: lượt yêu cầu telemetry đã bị sandbox chặn.
.venv/bin/python scripts/T08_monitor_test.py --duration 1 --require-gpu-metrics
```

Nguồn bằng chứng:

- **E08-IDLE:** [summary](../logs/t08/20261018T081705Z_e33b9e55/summary.json).
- **E08-RUN:** [summary](../logs/t08/20261018T084314Z_0c742da6/summary.json), [monitor summary](../logs/t08/20261018T084314Z_0c742da6/monitor_summary.json), [CSV](../logs/t08/20261018T084314Z_0c742da6/resources.csv).
- **E08-OH:** [summary overhead](../logs/t08/20261018T082334Z_ad189e2f/summary.json), [CSV từng cặp](../logs/t08/20261018T082334Z_ad189e2f/overhead/).
- **E08-BLOCKED:** [summary lỗi sandbox](../logs/t08/20261018T081923Z_809dd24f/summary.json).

| ID | Kịch bản / mong đợi | Kết quả đã ghi nhận |
|---|---|---|
| T08-AC01 | Standalone với GPU disabled; có CPU/RAM/process, log và shutdown sạch | PASS, E08-IDLE: 7 mẫu, written=7, dropped=0, không sampler/sink error; không inference job |
| T08-AC02 | Theo dõi hai backend đồng thời; 400 job; yêu cầu VRAM; log đầy đủ, không mất mẫu | PASS, E08-RUN: 400 job, 39 mẫu; 10/10 metric hợp lệ ở 39/39 mẫu; written=39, dropped=0; sampler/writer đã dừng |
| T08-AC03 | Thu đủ 3 cặp overhead; cùng session/input/thread settings; lưu elapsed thô, không ép overhead phải âm hoặc bằng 0 | PASS về thực hiện phép đo, E08-OH: mỗi pass 400 job; thay đổi elapsed -6.20%, -2.35%, +3.78%; median -2.35%. Không chứng minh monitor tăng tốc hoặc overhead=0 |
| T08-ENV01 | Nếu yêu cầu VRAM mà môi trường không đọc được thì báo lỗi, không tạo số 0 giả | Lượt thực thi BLOCKED — môi trường, E08-BLOCKED `status=failed`, lỗi GPU memory telemetry unavailable. Ngoài sandbox E08-RUN PASS |

E08-RUN còn ghi nhận phase idle, prepare_cpu, prepare_gpu, inference, cooldown;
30 mẫu thấy worker đang chạy; không mẫu stale trong lượt này. Max sample
duration 0.2424 s có thể dài hơn interval 0.2 s. RSS max quan sát 1073.25 MiB,
GPU memory used max 1022.95 MiB, GPU utilization max 47%.

Các giá trị max là đỉnh quan sát qua sampling, không phải peak tuyệt đối.
Lượt overhead trước bản chỉnh timestamp cuối có một lần đọc khoảng 1.096 s;
T08-M06 là regression test đảm bảo query chậm không che giấu độ cũ dữ liệu.

## 6. Chạy lại và đọc kết quả

### 6.1. Chuẩn bị

Chạy tại root `/home/yoo/HDH`, dùng `.venv` có dependencies trong
[requirements.txt](../requirements.txt). CPU unit/integration cần các thư
viện runtime đã cài; các ca GPU thật cần NVIDIA driver và quyền truy cập GPU.
Không cần GPU thật để chạy các ca MOCK/FAKE.

Không dùng CPU/GPU utilization làm bằng chứng thay thế cho compute CUDA.
Không chạy benchmark overhead đồng thời với workload GPU khác nếu muốn
đối chiếu thời gian có ý nghĩa.

### 6.2. Full suite

```bash
# Mặc định: các ca GPU thật được SKIP nếu chưa đặt biến môi trường.
.venv/bin/python -m unittest discover -s tests -v

# Chỉ định tường minh cả hai nhóm GPU khi nghiệm thu lại.
T06_TEST_GPU=1 T08_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -v

# Chỉ nhóm T06 hoặc T08.
T06_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -p 'test_t06_*.py' -v
T08_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -p 'test_t08_*.py' -v
```

Với bộ code hiện tại, full suite có 62 phương thức: 53 trực tiếp trong tài
liệu và 9 T04 regression. Lượt lịch sử đã đạt 62/62, không skip. Nếu không bật
GPU, có 2 ca GPU dự kiến skip; phải đọc dòng tổng kết thực tế của lượt chạy
mới, không sao chép kết quả cũ thành kết quả hiện tại.

Ví dụ chạy riêng một ca, chỉ cần đổi file/class/method theo các bảng:

```bash
PYTHONPATH=src:tests .venv/bin/python -m unittest \
  test_t06_backend_unit.BackendUnitTest.test_gpu_run_failure_is_not_retried_on_cpu -v
```

### 6.3. Cách xác nhận kết quả

1. Unittest: `OK` cùng số tests; `OK (skipped=...)` chưa đủ để xác nhận GPU thật.
2. CLI T06/T08: xem exit code và `summary.json`; `status=failed` vẫn là lượt không hoàn tất dù có một số bước CPU đã thành công.
3. T06: xem shape, tolerance và trace compute cho **cả hai model**; không chỉ xem providers.
4. T08: xem valid/unavailable theo metric, reason, written/dropped, sampler/sink error và running=False khi kết thúc.
5. Overhead: giữ từng cặp baseline/monitored và thứ tự chạy; không biến số âm do dao động thành kết luận tăng tốc.

Không có ngưỡng latency tuyệt đối trong các ca này. Timing T02, T06 và T08
có phạm vi/cấu hình khác nhau, không dùng để so sánh trực tiếp hiệu năng.

## 7. Phạm vi chưa được nghiệm thu bởi bộ testcase này

- Queue/controller/WorkerLoop và vòng đời job end-to-end của T05/T07.
- Worker health và queue state của pipeline thực; T08 hiện kiểm tra qua callback/fake source và harness T06.
- Scheduler CPU-only/GPU-only/Round Robin/Resource-aware của pipeline.
- Profiling service time T11, Collector T10, replay workload và benchmark T15.
- Drain queue, đối soát accepted/completed/failed và graceful shutdown toàn hệ thống khi còn job.
- Độ chính xác classification trên dataset có nhãn, memory leak dài hạn hoặc mọi race condition có thể xảy ra.

Kết quả PASS của các ca hữu hạn ở trên chỉ xác nhận hành vi được assert hoặc
được đối chiếu bằng artifact; không thay thế nghiệm thu các phần chưa triển khai.
