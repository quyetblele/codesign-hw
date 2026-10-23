# T06 — CPU/GPU inference backends

Ngày xác nhận: 2026-10-18. **T06 hoàn thành trong phạm vi backend đã chốt ở T04**:
load/warm-up model một lần mỗi vòng đời, chạy inference, trả output hoặc lỗi.
WorkerLoop/Queue/Controller thuộc T05/T07; pipeline và shutdown toàn hệ thống
chưa được nghiệm thu bởi báo cáo này.

## Implementation

| File | Trách nhiệm |
|---|---|
| `src/edge_scheduler/inference/model_registry.py` | Hai model ID, input/output shape, resolve file trong root |
| `src/edge_scheduler/inference/preprocess.py` | Tiền xử lý T03 dùng chung |
| `src/edge_scheduler/inference/onnx_backend.py` | Session lifecycle, warm-up, validation, chạy ONNX |
| `src/edge_scheduler/inference/cpu_backend.py` | `CPUBackend`, cố định `Device.CPU` |
| `src/edge_scheduler/inference/gpu_backend.py` | `GPUBackend`, cố định `Device.GPU` |
| `src/edge_scheduler/inference/verification.py` | Đọc operator trace, xác nhận compute CUDA |
| `scripts/T06_worker_test.py` | CLI chạy thật, so CPU/GPU, lưu artifact mỗi lượt |
| `tests/test_t06_preprocess.py` | Chuẩn hóa ảnh, center crop, RGB conversion, path containment |
| `tests/test_t06_backend_unit.py` | Session reuse, lifecycle, lỗi input/model/provider, cleanup, fallback |
| `tests/test_t06_backend_integration.py` | Hai model thật, job lặp, CPU/GPU đồng thời và trace CUDA |

Giữ nguyên `InferenceBackend` trong `contracts.py`:

```python
device: Device
prepare() -> None
infer(model_id: str, input_ref: str) -> numpy.ndarray
close() -> None
```

`prepare()` tạo hai session cho hai model, warm-up và chỉ đặt `ready=True` khi
cả hai thành công. Gọi lại khi ready không reload. Chuẩn bị thất bại dọn các
session đã tạo; có thể thử `prepare()` lại. `infer()` không tạo session.
`close()` kết thúc profiling, bỏ tham chiếu session, gọi lặp an toàn; sau close
có thể prepare lại. Backend không giữ/sửa Job hoặc tự chọn queue.

Một backend được một worker gọi tuần tự; không hỗ trợ chạy close/prepare song
song với infer trên cùng instance. Hai backend riêng được kiểm tra inference
đồng thời bằng hai thread. Khi ghép T07, Controller quản lý trạng thái Job và
WorkerLoop chuyển exception thành WorkerMessage FAILED.

## Dữ liệu và xử lý lỗi

- Model ID đúng: `mobilenetv2`, `resnet50`.
- Input: đường dẫn ảnh tương đối với root project, ví dụ `inputs/test.jpg`.
- Root mặc định từ vị trí package, không phụ thuộc current working directory;
  có thể truyền `root=...`. Từ chối absolute path và đường dẫn/symlink thoát root.
- RGB → resize cạnh ngắn 256 → center crop 224 → float32 / 255 → ImageNet
  mean/std → NCHW `(1, 3, 224, 224)`, contiguous.
- Tên input đọc từ metadata: MobileNetV2 `input`, ResNet-50 `data`.
- Output: mảng điểm số float32 hữu hạn `(1, 1000)`, giữ đủ 1000 lớp.
- Input thiếu/hỏng/model ID sai phát exception; session error có context
  device/model. Backend không retry tự động.
- Provider phải tồn tại trong runtime và đứng đầu danh sách của session.
  Tắt fallback cả constructor (ORT 1.23.2 `enable_fallback=False`) lẫn
  `session.disable_fallback()`; warm-up phải thành công. GPU không chuyển toàn
  bộ job về CPU khi khởi tạo hoặc inference lỗi.
- Node phụ có thể được ORT gán cho CPU. Đây là graph partitioning, khác với
  chuyển toàn bộ job sang CPU. Nghiệm thu yêu cầu Conv/Gemm/MatMul chạy trên CUDA.

`scripts/preprocess.py` giữ compatibility import; script T03 đã chuyển sang
import implementation trong package. Không nhân đôi logic preprocessing.

## Môi trường và cấu hình kiểm tra

Python 3.10.12, ONNX Runtime GPU 1.23.2, NumPy 2.2.6, Pillow 12.3.0 trên
Ubuntu/WSL2, NVIDIA GTX 1650 4 GB. Dependencies trực tiếp và NVIDIA packages đã
khóa trong `requirements.txt`; toàn bộ package versions được lưu trong summary.

- Batch size: 1.
- Warm-up: 5 lượt/model/device trong CLI; 1 lượt trong integration tests.
- CLI nghiệm thu: 10 job/model/device, xen kẽ hai model, tổng 40 job thành công.
- Mỗi session: intra-op 2, inter-op 1, ORT_SEQUENTIAL, intra-op spinning tắt.
- Đồng hồ: `time.monotonic()`.
- Execution gồm đọc ảnh/preprocess, transfer khi cần, ONNX run đồng bộ trả
  NumPy và validation. Model loading/CUDA initialization/warm-up ghi riêng.
- Profiling bật trong lượt nghiệm thu để lấy bằng chứng; có overhead.
- So sánh output: `np.testing.assert_allclose(rtol=1e-4, atol=1e-4)`.

## Kết quả thực tế

Lượt nghiệm thu thành công:
[summary.json](logs/t06/20261018T033611Z_865ab910/summary.json),
[jobs.csv](logs/t06/20261018T033611Z_865ab910/jobs.csv).
Trace gốc trong thư mục `profiles/` cùng lượt; output đầu tiên của mỗi
model/device nằm trong `outputs.npz`.

| Model | CPU mean (ms) | GPU mean (ms) | Max absolute difference |
|---|---:|---:|---:|
| MobileNetV2 | 12.654 | 14.484 | 8.1062e-06 |
| ResNet-50 | 51.274 | 16.578 | 1.3351e-05 |

Cả hai phép so output đều PASS. Tất cả output có shape `(1, 1000)`.

Trace xác nhận:

| Model GPU | CUDA Conv events | CUDA Gemm events | Compute verified |
|---|---:|---:|---|
| MobileNetV2 | 780 | 15 | Có |
| ResNet-50 | 795 | 15 | Có |

Số event bao gồm 5 warm-up và 10 job. MobileNetV2 có một số node phụ trên CPU;
trace vẫn xác nhận Conv/Gemm thực thi CUDA. Không sử dụng GPU utilization hoặc
chỉ danh sách available providers làm bằng chứng thay thế.

Đây là số liệu kiểm tra backend trên một ảnh và một lượt chạy có profiling,
chưa là benchmark T11/T15. Không suy ra GPU luôn nhanh hơn CPU hoặc hiệu quả
của bất kỳ scheduler nào từ bảng này.

Lượt đầu trong sandbox thất bại ở CUDA driver và trả exit code 1, không chuyển
GPU về CPU. Summary lỗi được giữ tại
`logs/t06/20261018T033454Z_a6f97d46/summary.json`. Cùng lệnh chạy ngoài sandbox
đã PASS, nên lỗi này là giới hạn truy cập GPU của môi trường kiểm thử.

## Kiểm thử và chạy lại

Từ root project, dùng `.venv` đã có hoặc cài `requirements.txt` vào môi trường
Python 3.10 riêng. Không cài đồng thời `onnxruntime` và `onnxruntime-gpu`.

```bash
# Unit tests, real CPU integration, T04 regression. GPU case được skip có lý do.
.venv/bin/python -m unittest discover -s tests -v

# Full acceptance: GPU bắt buộc, không skip nếu CUDA lỗi.
T06_TEST_GPU=1 .venv/bin/python -m unittest discover -s tests -v

# Backend CPU thật.
.venv/bin/python scripts/T06_worker_test.py --device cpu

# Cả bốn model/device, so output và yêu cầu trace compute CUDA.
.venv/bin/python scripts/T06_worker_test.py --device both --require-gpu --profile
```

Đã chạy bộ full acceptance: **33 tests PASS, không skip**, gồm 9 test T04 cũ,
15 unit tests backend/evidence, 6 test preprocessing/path và 3 integration tests.
Integration GPU chạy CPU/GPU đồng thời, đối chiếu cả hai model và đọc trace.
Lượt mặc định không bật GPU: 32 PASS, 1 SKIP.

CLI tạo thư mục UTC timestamp + UUID riêng cho mỗi lượt, không ghi đè lượt cũ.
Lỗi làm CLI trả 1 và ghi `status=failed` trong `summary.json`; thông số CLI sai
trả 2. `--require-gpu` không dùng với `--device cpu`. GPU được yêu cầu luôn
bắt buộc chạy thật, kể cả khi không truyền `--require-gpu`.

## Bàn giao cho T07

Khởi tạo một CPUBackend và một GPUBackend; prepare trước khi nhận workload đo.
Mỗi WorkerLoop gọi infer tuần tự và phát STARTED ngay trước preprocess, sau đó
COMPLETED khi output đã sẵn sàng hoặc FAILED khi exception. Controller gắn
kết quả với job ID, cập nhật Job và đối soát terminal outcomes. Drain queue
trước khi gọi close; flush collector thuộc phần tích hợp T07/T14.

Chưa triển khai hoặc nghiệm thu queue, controller, scheduler, monitor,
collector, end-to-end latency và graceful shutdown khi queue còn job.

Tài liệu API tham chiếu:
[ORT Python API](https://onnxruntime.ai/docs/api/python/api_summary.html),
[CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[Thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
