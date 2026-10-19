# Kế hoạch triển khai và đánh giá project CPU/GPU Edge AI

Tài liệu này xác định cách xây dựng và đánh giá project. Giới thiệu tổng quan nằm trong [README.md](README.md). Kế hoạch chưa gắn ngày bắt đầu/hạn nộp. Cập nhật 2026-10-18: đã có backend T06 và bằng chứng chạy hai model trên CPU/CUDA trong [T06_report.md](T06_report.md), cùng Resource Monitor/log T08 trong [T08_report.md](T08_report.md); các tiêu chí pipeline/benchmark còn lại chưa được nghiệm thu.

## 1. Scope

### Required scope

Terminal interface; ít nhất hai pretrained ONNX models; CPU/GPU inference; incoming/CPU/GPU queues; tối thiểu một worker mỗi thiết bị; bốn scheduler; Resource Monitor; Metrics Collector; workload generator/replay; synchronization, xử lý lỗi và graceful shutdown; benchmark, report/demo.

### Optional scope

Priority/deadline-aware scheduling, nhiều CPU workers, CPU affinity, dynamic batching, TensorRT, memory/power-aware scheduling, Jetson và mở rộng CPU/GPU/NPU. Chỉ thực hiện sau khi MVP ổn định.

### Out of scope

Train model mới, tự viết CUDA kernel/GPU driver/deep learning framework, thiết kế phần cứng và GUI trong phiên bản đầu. Không triển khai automatic retry ở MVP để đơn giản hóa kiểm soát xử lý trùng.

## 2. Functional Requirements

- Mỗi job có ID duy nhất, model, input, timestamps, thiết bị được gán, trạng thái và kết quả/lỗi.
- Job Generator tạo/replay mixed workload với low, medium, high và burst load.
- Có Incoming Queue, CPU Queue, GPU Queue và các worker độc lập.
- Scheduler hỗ trợ CPU-only, GPU-only, Round Robin và Resource-aware.
- Resource-aware xét thời gian xử lý theo model–device, công việc đang chạy/chờ, tài nguyên và trạng thái worker.
- Monitor cung cấp snapshot tài nguyên; collector ghi sự kiện và tổng hợp metrics.
- Sử dụng queue phù hợp với thread/process và bảo vệ dữ liệu chia sẻ khi cần; kiểm tra race condition/deadlock trong các kịch bản tích hợp.
- Xử lý lỗi có nguyên nhân; kiểm soát không mất/trùng job trong các kịch bản kiểm tra.
- Graceful shutdown ngừng tạo job, xử lý hết job đã tiếp nhận, đóng worker và flush log. Khi lỗi nghiêm trọng, ghi rõ job chưa hoàn thành.

## 3. System Requirements

| Hạng mục | Yêu cầu / công việc xác nhận |
|---|---|
| Hardware | CPU x86-64, NVIDIA GPU hỗ trợ CUDA, SSD; cấu hình mục tiêu khoảng 4 CPU cores, RAM 16 GB, VRAM 4 GB; điều chỉnh theo model thực tế |
| Environment | Windows 11 + WSL2 + Ubuntu dự kiến; có thể dùng Linux trực tiếp nếu nhóm thống nhất |
| Runtime | Python và ONNX Runtime; chưa khóa version |
| CUDA | Chọn CUDA/cuDNN dependencies tương thích với ONNX Runtime theo tài liệu chính thức khi setup |
| CPU inference | Smoke-test thành công bằng `CPUExecutionProvider`; ghi CPU thread settings |
| GPU inference | Smoke-test thành công bằng `CUDAExecutionProvider`; xác nhận thực thi CUDA thực tế, không chỉ provider khả dụng |
| Benchmark models | T03 chọn ít nhất hai pretrained ONNX models có chi phí khác nhau; mỗi model chạy được trên cả CPU/GPU |
| Monitoring | Thu CPU load, RAM, queue state, worker state và GPU memory khi khả dụng; ghi rõ giới hạn/missing metrics |

**GPU utilization % trong WSL không phải điều kiện bắt buộc để hoàn thành project.** Nếu metric không được hỗ trợ, ghi là unavailable thay vì thay bằng 0 hoặc số liệu suy đoán. Xác nhận backend GPU vẫn phải dựa trên inference thực tế và bằng chứng thực thi phù hợp.

Camera không bắt buộc; có thể dùng ảnh/video sẵn có. Nạp model và warm-up trước khi worker nhận workload đo lường. Không coi việc cài đồng thời hai gói ONNX Runtime CPU/GPU trong cùng môi trường là yêu cầu.

## 4. Architecture Responsibilities

Các interface dưới đây là hợp đồng thiết kế dự kiến, không phải API đã tồn tại.

| Thành phần | Input / output | Trách nhiệm |
|---|---|---|
| Job | `job_id`, `model_id`, `input_ref`, `priority`; timestamps, device, status, result/error | Lưu dữ liệu và vòng đời; `deadline` chỉ khi mở rộng |
| Scheduler | Nhận job + queue/resource/worker snapshot; trả device và sự kiện dispatch | Quyết định điều phối; chuyển job đúng một queue; đo overhead |
| Worker | Nhận job từ device queue; trả sự kiện start/finish và result/error | Sở hữu inference session, load/warm-up model một lần; xử lý độc lập |
| Queue | Nhận/lấy job và tín hiệu shutdown | Giao tiếp producer–consumer an toàn theo thread/process được chọn |
| Monitor | Đọc tài nguyên và tình trạng worker; xuất snapshot/log | Cung cấp thông tin cho scheduler, ghi thời điểm lấy mẫu và metric unavailable |
| Metrics Collector | Nhận sự kiện theo job ID và resource samples; xuất log/summary | Đối soát job, ghi dữ liệu và tính metrics nhất quán |

Job Generator sinh manifest và replay arrival schedule vào Incoming Queue. Trạng thái bình thường: `CREATED → WAITING → SCHEDULED → RUNNING → COMPLETED`; lỗi chuyển sang `FAILED` kèm nguyên nhân. T04 chốt thành phần chịu trách nhiệm cập nhật từng trạng thái và quyền truy cập dữ liệu chia sẻ.

### Resource-aware decision

Với một worker mỗi thiết bị, ước lượng thời gian hoàn thành tương đối:

```text
estimated_finish_delay(device, job)
    = estimated_remaining_time_of_running_job
    + sum(estimated_service_time_of_each_queued_job)
    + estimated_service_time(device, job)
```

Profiling xây dựng `estimated_service_time` riêng cho từng model–device. Xác định rõ chi phí này đã bao gồm transfer/preprocessing hay chưa để tránh cộng hai lần. Có thể cập nhật bằng trung bình trượt. Kiểm tra worker hoạt động và đủ bộ nhớ; dùng quy tắc cố định khi hai lựa chọn tương đương. CPU load và GPU memory là thông tin bổ sung.

## 5. Milestones

Các mốc dưới đây là mục tiêu, chưa xác nhận đạt.

| Mốc | Mục tiêu | Bằng chứng cần có | Thời điểm đề xuất |
|---|---|---|---|
| M1 | CPU/GPU backend hoạt động | Smoke-test CPU/CUDA, xác nhận GPU execution và cấu hình môi trường | Cuối tuần 1 |
| M2 | End-to-end job pipeline hoạt động | Workload nhỏ đi từ generator đến log kết quả; job được đối soát | Cuối tuần 2 |
| M3 | Các scheduler cơ bản hoạt động | CPU-only, GPU-only, Round Robin chạy cùng manifest, có metrics | Cuối tuần 3 |
| M4 | Resource-aware hoạt động | Quyết định dựa trên profiling/snapshot; có kiểm tra lỗi và shutdown | Cuối tuần 4 |
| M5 | Benchmark hoàn tất | Đủ chiến lược/mức tải, tối thiểu ba lần mỗi cấu hình, đủ config/log | Cuối tuần 5 |
| M6 | Report/demo hoàn tất | Báo cáo, biểu đồ, hướng dẫn chạy lại và demo | Cuối tuần 6 |

## 6. 6-Week Plan

A và B là placeholder cho đến khi xác định tên thành viên. Đây là phân công đề xuất, không phải tiến độ đã thực hiện.

| Week | Member A responsibilities | Member B responsibilities | Deliverable |
|---|---|---|---|
| 1 | Chốt phạm vi; thiết kế Job và interface | Setup, smoke-test; chọn model/input | M1; interface thống nhất và model/input ban đầu |
| 2 | Queue, vòng đời job; tích hợp CPU-only/GPU-only | Worker và log cơ bản | M2: workload nhỏ chạy trọn pipeline |
| 3 | Generator/replay, Round Robin | Monitor, metrics, profiling | M3: ba scheduler cơ bản có số liệu |
| 4 | Resource-aware, xử lý lỗi và shutdown | Hỗ trợ tích hợp; kiểm tra backend/số liệu | M4: bốn scheduler hoạt động ổn định |
| 5 | Kiểm tra điều phối; viết phần thiết kế | Benchmark và tổng hợp kết quả | M5: dữ liệu so sánh đủ mức tải |
| 6 | Hoàn thiện báo cáo, hướng dẫn và demo | Biểu đồ, phân tích và kiểm tra chạy lại | M6: report/demo hoàn chỉnh |

Cuối mỗi tuần chạy demo theo deliverable. Nếu chưa đạt mốc, cập nhật kế hoạch theo nguyên nhân trước khi bổ sung tính năng mở rộng.

## 7. Work Breakdown

> Detailed task status, owner, priority, branch, commit/PR and work logs are maintained in the shared Google Sheets project tracker.

PLAN giữ nội dung công việc, dependency và tiêu chí bàn giao; không dùng làm task tracker hằng ngày. Link tracker chưa được cung cấp trong tài liệu nguồn.

| Task | Công việc | Phụ thuộc | Deliverable / tiêu chí hoàn thành |
|---|---|---|---|
| T01 | Chốt phạm vi, phần cứng và tiêu chí đánh giá | — | Thống nhất MVP và máy benchmark |
| T02 | Setup WSL/CUDA/ONNX Runtime; smoke-test CPU/CUDA Execution Provider | T01 | Smoke-test model chạy trên hai backend; xác nhận CUDA execution; lưu môi trường |
| T03 | Chọn hai benchmark models và chuẩn hóa input | T02 | Có model, input mẫu và preprocessing dùng chung |
| T04 | Thiết kế interface giữa các module | T01 | Thống nhất Job, sự kiện kết quả, worker/scheduler interface |
| T05 | Queue và vòng đời job | T04 | Tiếp nhận/chuyển queue, theo dõi bằng job ID |
| T06 | CPU/GPU workers | T03, T04 | Load model một lần, chạy job, trả kết quả/lỗi |
| T07 | Tích hợp CPU-only và GPU-only | T05, T06 | Chạy trọn pipeline trên từng thiết bị |
| T08 | Resource Monitor | T02, T04 | Snapshot và log tài nguyên; ghi giới hạn monitoring |
| T09 | Workload generator và replay | T04 | Low/medium/high/burst; replay cùng manifest |
| T10 | Metrics Collector | T04, T07 | CSV mỗi job và metrics tổng hợp |
| T11 | Profiling | T06 | Thời gian xử lý và bộ nhớ theo model–device |
| T12 | Round Robin | T07 | Luân phiên gán thiết bị đúng quy tắc |
| T13 | Resource-aware | T08, T11, T12 | Chọn thiết bị bằng chi phí dự kiến; log quyết định |
| T14 | Xử lý lỗi, synchronization và shutdown | T07, T10 | Kiểm tra không mất/trùng job, đóng worker và flush log |
| T15 | Benchmark bốn scheduler | T09, T10, T13, T14 | Cùng manifest/cấu hình; đủ lần lặp |
| T16 | Phân tích và biểu đồ | T15 | Bảng so sánh, độ biến động và giải thích kết quả |
| T17 | Báo cáo và demo | T16 | Report, hướng dẫn chạy lại và demo |

T02 dùng smoke-test model để xác nhận môi trường, không thay cho bước chọn hai benchmark models ở T03.

## 8. Benchmark Plan

### Workload và kiểm soát điều kiện

- Mixed workload có ít nhất hai model; ghi tỉ lệ trộn trong config.
- Sinh trước manifest gồm job ID, model, input và arrival time tương đối; replay cùng manifest cho mọi scheduler.
- Arrival schedule độc lập với tốc độ hoàn thành để quan sát tích tụ queue.
- Chọn tốc độ low/medium/high dựa trên profiling; burst có số job mỗi đợt và khoảng cách rõ ràng.
- Giữ cùng hardware, software, model, input, preprocessing, CPU thread settings và batch size. Nếu chọn batch size 1, áp dụng cho tất cả scheduler.

### Quy trình

1. Lưu hardware/software configuration, model, input shape, workload và seed.
2. Load model, warm-up; chỉ bắt đầu đo khi worker sẵn sàng.
3. Chạy bốn scheduler với cùng manifest.
4. Chạy tối thiểu ba lần mỗi cấu hình; thay đổi thứ tự scheduler giữa các lần.
5. Lưu riêng config, job/resource logs và summary của từng lần, không ghi đè.
6. Đối soát job và báo cáo độ biến động giữa các lần chạy.

Không tính model loading/CUDA initialization/warm-up vào latency thông thường. GPU finish time phải phản ánh kết quả sẵn sàng, không chỉ thời gian gửi lệnh bất đồng bộ. Dùng đồng hồ monotonic cho khoảng thời gian.

### Định nghĩa metrics

| Metric | Quy ước |
|---|---|
| Dispatch waiting time | `dispatch_time − arrival_time` |
| Device queue waiting time | `start_time − dispatch_time` |
| Total queue waiting time | `start_time − arrival_time` |
| Execution time | `finish_time − start_time`; ghi rõ phạm vi preprocessing/transfer/postprocessing |
| End-to-end latency | `finish_time − arrival_time` |
| P50/P95/P99 latency | Phân vị latency của job thành công; báo cáo failure rate riêng |
| Throughput | Số job thành công / thời gian từ arrival đầu tiên đến khi toàn bộ job đạt trạng thái kết thúc |
| CPU/GPU job distribution | Số job gán và số job thành công theo thiết bị |
| Resource usage | CPU load, RAM/VRAM và metrics khả dụng; thống kê trung bình/đỉnh, ghi missing metrics |
| Scheduler overhead | Thời gian lấy thông tin, tính quyết định và chuyển job sang device queue |
| Failure rate | Số job thất bại / số job được tiếp nhận |

Resource-aware không bắt buộc tốt nhất. Phân tích ảnh hưởng của chi phí transfer, tốc độ CPU/GPU, mức tải và sai số ước lượng. Báo cáo cần trả lời khi nào GPU queue nghẽn, CPU được tận dụng ra sao, chiến lược nào cải thiện latency/throughput và kết quả có nhất quán giữa các lần chạy không.

Nếu mở rộng deadline, định nghĩa hạn hoàn thành cho từng job; báo cáo job muộn và job thất bại/chưa hoàn thành riêng.

## 9. Acceptance Criteria

- [x] Ít nhất hai model chạy thực tế trên cả CPU và GPU — bằng chứng T06_report.md, 2026-10-18.
- [ ] Có CPU-only, GPU-only, Round Robin và Resource-aware.
- [ ] Có low, medium, high và burst workload.
- [ ] Replay cùng manifest cho các scheduler.
- [ ] Job có ID duy nhất và được theo dõi đến trạng thái kết thúc.
- [ ] Khi chạy bình thường, `accepted_jobs = completed_jobs + failed_jobs`.
- [ ] Mỗi accepted job có đúng một kết quả cuối; không xử lý trùng trong các kịch bản kiểm tra.
- [ ] Có kiểm tra input lỗi, worker lỗi và dừng khi queue còn job.
- [ ] Dừng bình thường xử lý hết job, đóng worker và ghi đầy đủ log.
- [ ] Có latency, throughput, waiting time, resource metrics khả dụng và scheduler overhead; ghi giới hạn monitoring, không bắt buộc GPU utilization % trong WSL.
- [ ] Tối thiểu ba lần chạy mỗi cấu hình benchmark, có config/seed/log để chạy lại.
- [ ] Report nêu điều kiện hiệu quả, trường hợp kém hiệu quả và giới hạn thử nghiệm.

Chỉ đánh dấu hoàn thành khi có bằng chứng tương ứng. Kết quả ban đầu trên laptop/PC CPU + NVIDIA GPU chưa đại diện cho toàn bộ nền tảng Edge AI.

## 10. Risks and Mitigation

| Rủi ro | Biện pháp xử lý |
|---|---|
| CUDA/ONNX compatibility | Kiểm tra compatibility chính thức khi setup; smoke-test trước tích hợp; ghi phiên bản thực tế |
| WSL GPU monitoring limitation | Đánh dấu metric unavailable, ghi giới hạn; không lấy GPU utilization % làm điều kiện nghiệm thu |
| GPU fallback sang CPU | Xác nhận provider và bằng chứng CUDA execution; không coi chỉ có provider trong danh sách là đủ |
| Model quá nhẹ | Profiling và chọn hai model có chi phí khác nhau, đo transfer/scheduler overhead |
| Workload quá thấp, không tạo queue | Chọn tốc độ đến theo năng lực đo được; thêm high/burst workload |
| Oversubscription CPU threads | Giới hạn CPU inference threads; ghi và cố định cấu hình trong benchmark |
| Benchmark không reproducible | Replay manifest, giữ môi trường/input, lưu seed/config và lặp tối thiểu ba lần |

## 11. Future Work

- Deadline-aware scheduling.
- Priority scheduling.
- Dynamic batching.
- TensorRT backend.
- Memory-aware scheduling.
- Power-aware scheduling.
- Jetson deployment.
- CPU/GPU/NPU extension.
- Nhiều CPU workers và CPU affinity nếu còn thời gian.

Các mục này không mở rộng tiêu chí nghiệm thu MVP và chỉ triển khai sau khi phạm vi bắt buộc hoàn tất.
