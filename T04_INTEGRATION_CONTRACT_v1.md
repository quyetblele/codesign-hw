# T04 — HỢP ĐỒNG GIAO TIẾP GIỮA CÁC MÔ-ĐUN (DỰ THẢO V1)

Mục đích: hai người có thể code độc lập và ghép các phần không phải đoán kiểu dữ liệu, chữ ký hàm hoặc nơi chịu trách nhiệm.
Trạng thái: ĐỀ XUẤT CẦN HAI NGƯỜI DUYỆT. Tệp này KHÔNG thay đổi mã nguồn hiện tại.

Cập nhật checkout 2026-10-18: hợp đồng Python đã có trong `contracts.py` và
backend T06 đã triển khai. Tài liệu này vẫn giữ vai trò thiết kế; các phần
Queue/Scheduler/Controller mô tả bên dưới chưa có implementation trong checkout.

## 1. Nguyên tắc kiến trúc

- Job là cấu trúc dữ liệu duy nhất của một yêu cầu suy luận.
- Queue chỉ nhận/chứa/chuyển Job, không suy luận và không chọn thiết bị.
- Scheduler chỉ quyết định Device.CPU hoặc Device.GPU; Controller mới gọi dispatch.
- WorkerLoop do phía điều phối quản lý, đọc Device Queue và gọi CPUBackend hoặc GPUBackend.
- CPUBackend/GPUBackend chỉ chuẩn hóa đầu vào, chạy ONNX và trả tensor/lỗi; không tự chọn thiết bị.
- ResourceMonitor chỉ cung cấp ResourceSnapshot, không tự sửa Job.
- MetricsCollector nhận sự kiện, ghi dữ liệu và tính metrics; không tự chạy Job.
- Một phía duy nhất được phép cập nhật trạng thái Job; tránh hai thread cùng ghi.

## 2. Sơ đồ giao tiếp

~~~mermaid
flowchart TD
  W["T09: Workload Generator"] -->|"ManifestEntry / Job"| IQ["T05: Incoming Queue"]
  IQ -->|"Job"| C["T07: Controller"]
  R["T08: ResourceMonitor"] -.->|"ResourceSnapshot"| C
  C -->|"Job + queued_jobs + snapshot"| S["T04/T12/T13: Scheduler"]
  S -->|"CPU hoặc GPU"| C
  C -->|"dispatch(Job, Device)"| Q["T05: Device Queue"]
  Q -->|"Job"| L["T07: WorkerLoop"]
  L -->|"infer(model_id, input_ref)"| B["T06: CPUBackend / GPUBackend"]
  B -->|"Tensor hoặc lỗi"| L
  L -->|"Hoàn thành / thất bại"| C
  C -->|"JobEvent"| M["T10: MetricsCollector"]
~~~

## 3. Mã đã tồn tại (đã kiểm tra)

- src/edge_scheduler/models.py: Job, JobStatus, Device và mark_waiting, mark_scheduled, mark_running, mark_completed, mark_failed.
- src/edge_scheduler/interfaces.py: WorkerState, ResourceSnapshot, Scheduler Protocol.
- src/edge_scheduler/contracts.py: InferenceBackend, JobEvent, WorkerMessage và các protocol tích hợp.
- src/edge_scheduler/cpu_backend.py, gpu_backend.py: backend T06; xem T06_report.md.

`queues.py`, `schedulers.py`, `workload.py`, `shutdown.py`, Controller,
WorkerLoop và implementation MetricsCollector/ResourceMonitor chưa có trong checkout này.

## 4. Hợp đồng T04 cần đóng băng trước khi triển khai

### 4.1. Job và định danh

- job_id: str, duy nhất trong một lần chạy.
- model_id: str, nhận đúng 2 ID chuẩn là "mobilenetv2" và "resnet50" (không truyền tên file .onnx).
- input_ref: MVP thống nhất dùng đường dẫn ảnh str, tính tương đối từ thư mục gốc dự án.
- priority: int, hiện chỉ lưu, chưa tác động lên scheduler.
- arrival_offset_s: float, lịch đến tương đối trong manifest, đơn vị giây.
- arrival_time/dispatch_time/start_time/finish_time: float, đồng hồ monotonic, đơn vị giây.
- assigned_device: Device.CPU hoặc Device.GPU.
- status: CREATED, WAITING, SCHEDULED, RUNNING, COMPLETED hoặc FAILED.
- result/error: kết quả hoặc lý do lỗi.

### 4.2. Scheduler (GIỮ API HIỆN CÓ)

~~~python
choose_device(
    job: Job,
    queued_jobs: Mapping[Device, Sequence[Job]],
    snapshot: ResourceSnapshot | None = None,
) -> Device
~~~

queued_jobs chỉ tính Job còn chờ trong device queue, không tự tính lặp Job đang chạy. Scheduler không đổi trạng thái hay đưa Job vào queue. Trường hợp không có thiết bị khả dụng phải xử lý lỗi/backpressure trước khi dispatch.

### 4.3. JobQueues (GIỮ API HIỆN CÓ)

~~~python
queues.accept(job)             # WAITING, incoming queue, chặn job_id trùng
queues.dispatch(job, device)   # SCHEDULED, CPU/GPU queue
queues.record_terminal(job)    # Ghi nhận kết thúc, chặn trùng
accepted, terminal = queues.reconcile()
~~~

T05 bổ sung phương thức lấy queue và snapshot Job đang chờ an toàn giữa các thread. Không yêu cầu người B truy cập cấu trúc nội bộ queue.

### 4.4. InferenceBackend (API ĐÃ TRIỂN KHAI TRONG T06 — NGƯỜI B)

~~~python
from typing import Protocol
import numpy as np

class InferenceBackend(Protocol):
    device: Device

    def prepare(self) -> None: ...
    def infer(self, model_id: str, input_ref: str) -> np.ndarray: ...
    def close(self) -> None: ...
~~~

Cần hai triển khai CPUBackend(device=CPU) và GPUBackend(device=GPU). Cả hai cùng:
- Nhận model_id thuộc {mobilenetv2, resnet50}, input_ref là đường dẫn ảnh.
- Áp dụng cùng tiền xử lý T03: RGB -> resize cạnh ngắn 256 -> center crop 224 -> float32 [0,1] -> ImageNet normalization -> CHW -> batch.
- Tạo đầu vào (1,3,224,224), float32; trả numpy output (1,1000) với hai mô hình này.
- Nạp model và warm-up ở prepare(), không khởi tạo ONNX session cho từng job.
- Nếu lỗi input, model, thiếu provider hoặc suy luận: raise exception có thông báo rõ; KHÔNG âm thầm chuyển GPU về CPU.
- Không nhận job_id, priority hay JobQueues; không cập nhật Job.status.
- Chỉ Controller/WorkerLoop được gắn output với job_id và tính thời gian.

### 4.5. JobEvent (API ĐỀ XUẤT CHƯA CODE — HAI NGƯỜI CÙNG DUYỆT)

~~~python
from dataclasses import dataclass
from typing import Literal

EventType = Literal["ACCEPTED", "DISPATCHED", "STARTED", "COMPLETED", "FAILED"]

@dataclass(frozen=True, slots=True)
class JobEvent:
    kind: EventType
    job_id: str
    model_id: str
    timestamp_s: float
    device: Device | None = None
    top1_class: int | None = None
    error: str | None = None
~~~

- timestamp_s theo time.monotonic() trên cùng một máy.
- FAILED phải có lý do; COMPLETED có thể kèm top-1 và shape (nếu thống nhất thêm).
- Không ghi nguyên cả mảng 1000 điểm số vào CSV mỗi Job; kết quả đầy đủ có thể được Job giữ riêng.

### 4.6. ResourceMonitor và MetricsCollector (API ĐỀ XUẤT — NGƯỜI B)

~~~python
class ResourceMonitor(Protocol):
    def snapshot(self) -> ResourceSnapshot: ...

class MetricsCollector(Protocol):
    def record(self, event: JobEvent) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
~~~

Monitor trả snapshot theo WorkerState/ResourceSnapshot hiện có. Trường không đo được = None, không gán 0 giả.
Collector liên kết sự kiện bằng job_id, tính:
- Waiting = start_time - arrival_time.
- Execution = finish_time - start_time.
- Latency = finish_time - arrival_time.
Quy ước đo đề xuất: START ngay trước preprocess; FINISH sau khi ONNX output thực sự sẵn sàng. Thời gian inference riêng có thể lưu thêm.

### 4.7. Controller và WorkerLoop (ĐỀ XUẤT — NGƯỜI A)

- Controller nhận Job, gọi accept, lấy queued_jobs và snapshot, gọi Scheduler, dispatch đúng một lần.
- WorkerLoop A đọc Job từ CPU/GPU queue, phát sự kiện STARTED với timestamp thực tế, gọi Backend B và phát sự kiện COMPLETED/FAILED qua một event queue an toàn giữa các thread.
- Controller là nơi duy nhất áp dụng sự kiện vào Job (mark_running/mark_completed/mark_failed), ghi vào JobQueues.record_terminal và chuyển JobEvent cho MetricsCollector. Các bước accept/dispatch hiện có chỉ được Controller gọi. Backend và WorkerLoop không trực tiếp sửa trạng thái Job.
- B có thể hoàn thành backend độc lập và kiểm thử bằng một hàm infer().
- A có thể hoàn thành Controller độc lập nhờ FakeBackend trả mảng numpy giả.

## 5. Một Job đi qua hệ thống như thế nào?

1. T09 tạo manifest; replay chờ đến thời điểm arrival_offset_s rồi mới tiếp nhận Job.
2. Khi thật sự accept, ghi arrival_time bằng time.monotonic(). Chú ý mã hiện tại ghi lúc tạo Job; cần điều chỉnh khi replay nếu Job được tạo trước thời điểm đến.
3. JobQueues.accept(job): trạng thái WAITING.
4. Controller lấy Job, lấy Queue snapshot + ResourceMonitor.snapshot().
5. Scheduler.choose_device(): trả CPU hoặc GPU.
6. JobQueues.dispatch(job, device): trạng thái SCHEDULED.
7. WorkerLoop đúng device lấy Job, ghi start_time, trạng thái RUNNING.
8. Backend.infer(model_id, input_ref): trả output hoặc raise lỗi.
9. WorkerLoop/controller ghi finish_time, COMPLETED/FAILED; phát JobEvent; record_terminal(job).
10. Collector ghi log; sau lượt chạy kiểm tra accepted == terminal.
11. Dừng bình thường: ngừng nhận Job -> chạy hết hàng đợi -> đóng worker -> flush/close collector.

## 6. Phân chia trách nhiệm đề xuất

| Hạng mục | Người A | Người B |
|---|---|---|
| Hợp đồng Job, Scheduler, JobEvent | Sở hữu mã, hai bên cùng duyệt | Cùng duyệt, không sửa trực tiếp |
| models.py / interfaces.py | Chủ trì | Dùng chung |
| queues.py / controller.py / worker_loop.py | Triển khai | Không chạm nội bộ |
| schedulers.py / workload.py / shutdown.py | Triển khai | Cung cấp profile/thông tin cần dùng |
| cpu_backend.py / gpu_backend.py | Gọi API | Triển khai ONNX CPU/GPU |
| preprocess.py / model_registry.py | Dùng API | Triển khai/kiểm tra |
| resource_monitor.py / metrics_collector.py / profiling.py | Gọi API | Triển khai |
| tests/test_contract.py / tests/test_integration.py | Chủ trì | Cùng nghiệm thu |

`cpu_backend.py`, `gpu_backend.py`, `model_registry.py`, preprocessing và
`JobEvent` đã có. `controller.py`, `worker_loop.py`, `resource_monitor.py`,
`metrics_collector.py`, `profiling.py` vẫn là kiến trúc dự kiến.

## 7. Checklist nghiệm thu trước khi code song song

- [ ] Hai người xác nhận tên module, class, kiểu tham số, giá trị trả về và lỗi.
- [ ] Dùng chung đúng model_id mobilenetv2/resnet50, device CPU/GPU và chuẩn input_ref.
- [ ] Backend trả cùng kiểu output; không âm thầm fallback CUDA -> CPU.
- [ ] Scheduler không đụng đến Queue hay Backend.
- [ ] Chỉ một nơi chuyển trạng thái Job; không có hai terminal result cho một job_id.
- [ ] Đồng hồ đo thời gian, phạm vi tính execution/latency đã thống nhất.
- [ ] T08 trả ResourceSnapshot, T10 nhận JobEvent, T11 trả số đo thực cho mỗi model/device.
- [ ] FakeBackend chạy được luồng A mà không cần GPU; hai Backend B được test độc lập.
- [ ] T07 ghép tối thiểu 10 Job và đối soát số tiếp nhận = số kết thúc.
- [ ] Test input lỗi, Worker lỗi, Job ID trùng, tắt hệ thống khi còn job.
- [ ] Bốn kết hợp MobileNetV2/ResNet-50 trên CPU/GPU đều được xác nhận thực thi.
- [ ] Mọi thay đổi API sau khi khóa hợp đồng phải có hai người đồng ý.

## 8. Các vấn đề trong code hiện tại cần sửa khi triển khai tiếp

1. Worker/Monitor/Collector/Event contracts hiện nằm trong contracts.py; implementation tích hợp còn thiếu.
2. JobQueues chưa cung cấp snapshot hàng đợi an toàn cho Scheduler.
3. Job.mark_* chưa kiểm soát hoàn toàn các bước chuyển trạng thái sai.
4. Job.arrival_time lấy ở lúc tạo thay vì lúc phát lại Job thực sự đến.
5. ResourceAwareScheduler cần xử lý trường hợp cả CPU/GPU không khả dụng thay vì tie-breaker.
6. priority chưa được sử dụng trong điều phối; tránh giới thiệu như đã hoạt động.
7. T06 đã xác nhận cả MobileNetV2 và ResNet-50 trên CUDA bằng operator trace; xem T06_report.md.

## 9. Quy trình tích hợp để giảm lỗi

Giai đoạn 1 — A+B khóa hợp đồng v1 và tạo test hợp đồng mẫu.
Giai đoạn 2 — A làm Job/Queue/Controller/Scheduler với FakeBackend; B làm CPUBackend/GPUBackend với input thật.
Giai đoạn 3 — Chạy bộ kiểm thử hợp đồng cho từng phần trước khi merge.
Giai đoạn 4 — Ghép T07 qua API duy nhất, kiểm tra CPU-only / GPU-only.
Giai đoạn 5 — Thêm monitor, profiling, metrics; chạy Round Robin / Resource-aware.
Giai đoạn 6 — Chạy shutdown/error/stress tests, rồi mới benchmark.

**CHƯA CHẠM VÀO MÃ NGUỒN**: đây chỉ là tài liệu hợp đồng đề xuất; phải được cả hai người duyệt trước khi thay đổi interface.
