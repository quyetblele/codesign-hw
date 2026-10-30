# Hướng dẫn sử dụng T10, T11 và tạo báo cáo

Tài liệu này hướng dẫn các lệnh **đã triển khai** trong dự án CPU/GPU Edge AI Scheduling. Mọi câu lệnh đều viết trên một dòng để có thể sao chép vào terminal.

## 1. Bắt đầu và chọn lệnh

Mở terminal Linux/WSL2, chuyển vào thư mục dự án:

```bash
cd /home/yoo/HDH
```

| Mục đích | Lệnh |
|---|---|
| Xem các lệnh ngắn | `make help` |
| Chạy pipeline và quan sát T10 | `make run` |
| Đo đặc tính model–thiết bị bằng T11 | `make profile` |
| Tạo báo cáo cho lượt gần nhất | `make report` |
| Chạy bộ kiểm thử mặc định | `make test` |

Makefile tự chọn `.venv/bin/python` và đặt `PYTHONPATH`. Không cần kích hoạt virtual environment bằng `source` trước mỗi lần chạy.

Các lệnh `make` phải được chạy ở thư mục chứa Makefile. Nếu đang ở thư mục khác, có thể chỉ rõ thư mục dự án:

```bash
make -C /home/yoo/HDH help
```

### Chuẩn bị môi trường lần đầu

Nếu chưa có `.venv`, tạo môi trường và cài dependencies theo thứ tự:

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Kiểm tra GPU và dependencies:

```bash
nvidia-smi
```

```bash
.venv/bin/python -m pip check
```

Dự án được kiểm tra với Python 3.10 trên Linux/WSL2, NVIDIA GPU và ONNX Runtime GPU. Cần có công cụ `make` trong hệ điều hành. Nếu môi trường đã chạy được dự án thì dùng lại, không cần tạo mới.

**Yêu cầu GPU hiện tại:** `make run` khởi tạo cả CPU và GPU, kể cả khi chọn `POLICY=cpu-only`. Máy không có GPU có thể dùng `make profile DEVICE=cpu` để đo CPU độc lập; chế độ này không thay thế pipeline T10.

## 2. T10 — chạy pipeline và quan sát trực tiếp

### 2.1. Chạy với cấu hình mặc định

```bash
make run
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `POLICY` | `round-robin` | Chính sách gán thiết bị |
| `JOBS` | `1000` | Tổng số job trong batch |
| `WINDOW` | `5` | Cửa sổ Recent, đơn vị giây |
| `UPDATE` | `1` | Chu kỳ cập nhật màn hình, đơn vị giây |
| Models | `mobilenetv2 resnet50` | Chọn model luân phiên cho các job |
| Input | `inputs/test.jpg` | Ảnh đầu vào dùng chung |
| Warm-up | `5` | Số lần warm-up mỗi model/device |
| ORT threads | `2` | Intra-op threads mỗi session |
| Monitor interval | `0.2` | Chu kỳ lấy mẫu tài nguyên, đơn vị giây |

Đây là batch hữu hạn. Job được tiếp nhận rồi điều phối; chưa phải workload low/medium/high/burst hoặc replay manifest theo lịch đến.

### 2.2. Chọn chính sách và số lượng job

Round Robin, luân phiên gán CPU và GPU:

```bash
make run POLICY=round-robin JOBS=1000
```

Mọi job chạy CPU:

```bash
make run POLICY=cpu-only JOBS=100
```

Mọi job chạy GPU:

```bash
make run POLICY=gpu-only JOBS=100
```

Chạy batch nhỏ để kiểm tra cấu hình:

```bash
make run JOBS=20
```

CLI hiện hỗ trợ đúng ba policy trên. Resource-aware chưa được bật thành lựa chọn của `make run`.

### 2.3. Điều chỉnh Recent và tốc độ cập nhật

Giữ thống kê 5 giây gần nhất, cập nhật mỗi 0,5 giây:

```bash
make run WINDOW=5 UPDATE=0.5
```

Quan sát biến động ngắn trong 1 giây gần nhất:

```bash
make run WINDOW=1 UPDATE=0.5
```

Giảm dao động bằng cửa sổ 10 giây:

```bash
make run WINDOW=10 UPDATE=1
```

`WINDOW` và `UPDATE` độc lập:

- Recent throughput đếm job **thành công kết thúc trong cửa sổ**, chia cho độ dài cửa sổ. Khi chưa chạy đủ cửa sổ, mẫu số là thời gian thực tế từ arrival đầu tiên.
- Recent P50/P95 lấy latency của chính các job thành công kết thúc trong cửa sổ. Job vẫn đang chờ/chạy chưa nằm trong nhóm này; xem thêm queue và unfinished.
- `UPDATE` chỉ thay tần suất vẽ màn hình, không thay chu kỳ lấy mẫu tài nguyên hay công thức thống kê cuối lượt.
- Cửa sổ 1 giây phản ứng nhanh nhưng ít mẫu và dao động mạnh hơn cửa sổ 5–10 giây. Luôn đọc kèm `n`, tức số mẫu latency.
- Cập nhật nhanh hơn chu kỳ cảm biến có thể hiển thị lại cùng một mẫu và tăng chi phí tính/vẽ. Mặc định `WINDOW=5 UPDATE=1` phù hợp để theo dõi thông thường.
- Không có mẫu hiển thị `—`, không thay dữ liệu thiếu bằng 0.

`JOBS` phải là số nguyên dương; `WINDOW` và `UPDATE` phải là số hữu hạn lớn hơn 0.

### 2.4. Đọc bảng terminal

Ví dụ minh họa, số thực tế phụ thuộc lượt chạy:

```text
T10 | round-robin | workload | elapsed 0.7s
Jobs accepted=100 completed=80 failed=0 unfinished=20
Queues incoming=0 cpu=10 gpu=8
CPU: running mobilenetv2 | GPU: running resnet50
Recent 1s (n=80): 109.40 jobs/s | P50=372.07 ms P95=690.15 ms
CPU=26.40% RSS=1077.88 MB RAM available=5340.33 MB
VRAM used/free=1052.20/3242.76 MB | GPU=22.00%
Log dropped resources=0 events=0 | sample stale=False age=0.01s metric_errors=0 | error=none
```

| Trường | Cách hiểu |
|---|---|
| `prepare` | Đang nạp model và warm-up; chưa nhận workload đo |
| `workload` | Đang xử lý batch |
| `drain` | Ngừng tiếp nhận, xử lý nốt các job đã nhận |
| `shutdown` | Đóng backend, monitor và writer |
| `elapsed` | Thời gian từ arrival đầu tiên; trước khi nhận job có thể bằng 0 |
| `accepted` | Số job đã được tiếp nhận |
| `completed` | Số job kết thúc thành công |
| `failed` | Số job kết thúc với lỗi |
| `unfinished` | Accepted trừ completed và failed |
| `Queues` | Số job còn nằm trong từng queue; không gồm job đã được worker lấy ra |
| `idle/running/unavailable` | Trạng thái worker; unavailable trong prepare/shutdown có thể là bình thường |
| `P50/P95` | Phân vị latency Recent, đơn vị ms |
| `CPU` | CPU utilization của hệ điều hành nhìn thấy từ Linux/WSL |
| `RSS` | RAM resident của toàn process ứng dụng |
| `VRAM used/free` | Bộ nhớ đã dùng/còn trống của toàn GPU được chọn |
| `stale`, `age` | Mẫu phần cứng có cũ không, tuổi mẫu tính bằng giây |
| `metric_errors` | Số mục thiếu/lỗi trong mẫu; đọc nguyên nhân chi tiết ở log |
| `dropped` | Số mẫu/sự kiện bị mất; phải kiểm tra nếu khác 0 |

Queue và worker state là quan sát đồng thời, không phải snapshot nguyên tử của toàn hệ thống. Vì vậy không yêu cầu `unfinished` luôn bằng tổng ba queue: một số job đang chạy hoặc thông điệp kết quả đang chờ xử lý.

Trên terminal thật, bảng được cập nhật tại chỗ. Khi redirect, chương trình xuất các dòng log thường:

```bash
make run JOBS=100 > terminal.log 2>&1
```

Lệnh trên ghi đè `terminal.log` nếu file đã tồn tại. CSV/JSON của lượt chạy vẫn được lưu riêng.

### 2.5. Tham số nâng cao qua ARGS

Đặt nhiều cờ CLI trong cùng một cặp dấu nháy; các tên biến Make như `JOBS`, `WINDOW` không có dấu `--`.

| Cờ của pipeline | Mặc định | Ý nghĩa |
|---|---|---|
| `--models` | Hai model | Danh sách model dùng để tạo job |
| `--input` | `inputs/test.jpg` | Đường dẫn ảnh tương đối với project root |
| `--warmup` | `5` | Warm-up mỗi model/device |
| `--threads` | `2` | ORT intra-op threads mỗi session |
| `--monitor-interval` | `0.2` | Chu kỳ lấy mẫu tài nguyên, giây |
| `--max-sample-age` | `1` | Ngưỡng stale, giây |
| `--timeout` | `60` | Thời gian chờ workload trước khi chuyển sang drain và báo lỗi |
| `--shutdown-timeout` | `60` | Timeout cho mỗi lần thử drain/đóng |
| `--output-dir` | `logs/pipeline` | Thư mục cha chứa các lượt mới |
| `--device-id` | `0` | CUDA ordinal |
| `--gpu-index` | `0` | Chỉ số GPU của NVML |
| `--gpu-uuid` | Không đặt | Chọn GPU telemetry theo UUID, ưu tiên hơn NVML index |
| `--require-gpu-metrics` | Tắt | Bắt buộc đọc được VRAM; utilization vẫn có thể unavailable |
| `--no-gpu-telemetry` | Tắt | Tắt NVML; không tắt GPU inference |
| `--profile` | Tắt | Lưu ORT trace và yêu cầu compute CUDA cho hai model GPU |

Ví dụ chạy một loại job và tăng thời gian chờ:

```bash
make run JOBS=100 ARGS="--models resnet50 --timeout 120"
```

Thay đổi CPU threads và chu kỳ monitor:

```bash
make run JOBS=100 ARGS="--threads 2 --monitor-interval 0.1"
```

Xác minh CUDA bằng trace và yêu cầu VRAM thật:

```bash
make run JOBS=20 ARGS="--profile --require-gpu-metrics"
```

Tắt telemetry khi chỉ cần kiểm tra inference và các metrics khác:

```bash
make run JOBS=100 ARGS="--no-gpu-telemetry"
```

Đổi thư mục lưu:

```bash
make run JOBS=100 ARGS="--output-dir logs/demo-pipeline"
```

Không dùng đồng thời `--no-gpu-telemetry` và `--require-gpu-metrics`. Nếu đặt `CUDA_VISIBLE_DEVICES`, hãy ánh xạ NVML bằng `--gpu-uuid` hoặc tắt GPU telemetry; CUDA ordinal và NVML index không mặc nhiên giống nhau.

Chọn model bằng `--models` chỉ đổi workload của pipeline; pipeline vẫn prepare hai model trên cả hai backend theo thiết kế hiện tại. T11 ở phần tiếp theo mới đo với một model được nạp trong từng process.

### 2.6. Dừng và timeout

Nhấn **Ctrl+C** để ngừng tiếp nhận job. Chương trình drain các job đã nhận rồi đóng tài nguyên; có thể mất thêm thời gian để inference đang chạy hoàn tất.

`--timeout` không hủy ngay native inference. Nếu một lần drain vượt `--shutdown-timeout`, chương trình ghi checkpoint `summary.json` với các job chưa xong rồi tiếp tục thử đóng. Đây không phải hard timeout của toàn process.

### 2.7. Đầu ra T10

```text
logs/pipeline/<run_id>/
├── config.json
├── events.jsonl
├── jobs.csv
├── resources.csv
├── summary.json
└── profiles/              # Chỉ khi bật --profile
```

| File | Nội dung và thời điểm ghi |
|---|---|
| `config.json` | Cấu hình, môi trường, loại lượt `pipeline`, phạm vi thời gian/tài nguyên |
| `events.jsonl` | Mỗi dòng một JobEvent, ghi liên tục qua writer thread |
| `jobs.csv` | Một dòng khi job terminal; timestamps, thiết bị, trạng thái, lỗi, top-1 và metrics |
| `resources.csv` | Mẫu CPU/RAM/VRAM, queue, worker state, phase và lỗi từng metric |
| `summary.json` | Thống kê cuối lượt hoặc checkpoint; đối soát controller/collector, diagnostics |
| `profiles/` | Trace ONNX Runtime khi bật `--profile` |

Job chưa terminal ở checkpoint chưa có dòng kết thúc trong `jobs.csv`; xem `unfinished_job_ids` trong summary. Khi kết thúc bình thường, cần đối soát `accepted = completed + failed` và không còn unfinished.

Metrics thời gian:

| Chỉ số | Công thức |
|---|---|
| Dispatch wait | `dispatch − arrival` |
| Device queue wait | `start − dispatch` |
| Total wait | `start − arrival` |
| Execution | `finish − start` |
| Latency | `finish − arrival` |
| Throughput toàn lượt | Completed / thời gian từ arrival đầu đến terminal cuối, kể cả terminal failed |
| Scheduler overhead | Đo quanh lấy snapshot + chọn thiết bị + enqueue; khác dispatch wait |

Timestamp và duration trong `jobs.csv` dùng giây monotonic; thống kê thời gian trong summary dùng ms. Các trường tài nguyên `_mb` của lượt mới dùng **MB = bytes / 1.000.000**. **1000 MB = 1 GB = 1.000.000.000 bytes**. Đây là thay đổi phép tính sang hệ thập phân, không chỉ đổi nhãn. Phân vị timing được tổng hợp trên job thành công; failure rate báo riêng. Thời gian prepare/warm-up không nằm trong job latency.

Lượt mới có `memory_unit="MB"` và schema version 2. Log cũ dùng MiB vẫn giữ nguyên: **1 MiB = 1.048.576 bytes = 1,048576 MB**, **1 GiB = 1,073741824 GB**. Report đọc đơn vị của từng lượt từ config; config lịch sử thiếu trường này được hiểu là MiB. Không sửa thủ công nhãn của log cũ thành MB vì sẽ làm sai ý nghĩa số liệu.

## 3. T11 — profiling model trên CPU/GPU

### 3.1. Chạy mặc định

```bash
make profile
```

Lệnh đo bốn cặp: MobileNetV2/CPU, MobileNetV2/GPU, ResNet-50/CPU và ResNet-50/GPU.

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `DEVICE` | `both` | `cpu`, `gpu` hoặc `both` |
| `RUNS` | `100` | Số inference đo cho mỗi cặp trong mỗi lượt lặp |
| `REPEATS` | `3` | Số lượt lặp mỗi cặp |
| Models | Hai model | Có thể chọn qua `ARGS="--models ..."` |
| Warm-up | `5` | Mỗi cặp/mỗi lượt lặp |
| Threads | `2` | ORT intra-op threads mỗi session |
| Sampling interval | `0.2` | Chu kỳ lấy mẫu tài nguyên, giây |
| Trial timeout | `300` | Thời gian tối đa chờ một process trial, giây |

Với cấu hình mặc định, tổng số inference được đo là `2 model × 2 thiết bị × 100 RUNS × 3 REPEATS = 1200`, không tính warm-up. Có 12 trial process độc lập, mỗi process chỉ nạp một model trên một thiết bị. Bảng tổng hợp được in sau khi toàn lượt profiling kết thúc.

### 3.2. Các cách chạy thường dùng

Chỉ đo CPU, không yêu cầu GPU inference:

```bash
make profile DEVICE=cpu
```

Chỉ đo GPU:

```bash
make profile DEVICE=gpu
```

Chạy nhanh để kiểm tra lệnh và đầu ra:

```bash
make profile RUNS=5 REPEATS=1 ARGS="--warmup 1"
```

Đo một model trên CPU:

```bash
make profile DEVICE=cpu RUNS=100 REPEATS=3 ARGS="--models mobilenetv2"
```

Đo cả hai thiết bị và yêu cầu VRAM thật:

```bash
make profile RUNS=100 REPEATS=3 ARGS="--require-gpu-metrics"
```

Đổi threads và chu kỳ sampling:

```bash
make profile ARGS="--threads 2 --interval 0.1"
```

Tăng thời gian tối đa cho mỗi trial:

```bash
make profile ARGS="--trial-timeout 600"
```

Đổi thư mục lưu:

```bash
make profile RUNS=20 REPEATS=1 ARGS="--output-dir logs/demo-profiling"
```

`--input`, `--root`, `--device-id`, `--gpu-index`, `--gpu-uuid` và `--no-gpu-telemetry` cũng được hỗ trợ. Ảnh phải nằm trong project root và truyền bằng đường dẫn tương đối.

`--require-gpu-metrics` chỉ dùng khi chọn GPU/both và telemetry đang bật. Thiếu GPU inference được yêu cầu sẽ làm lượt failed, không chuyển job sang CPU.

### 3.3. Phân biệt đo thời gian và xác minh CUDA

Đo thời gian thông thường dùng `make profile` không bật trace. Khi cần kiểm chứng compute CUDA:

```bash
make profile DEVICE=gpu RUNS=5 REPEATS=1 ARGS="--verify-cuda --require-gpu-metrics"
```

Cờ của T11 là **`--verify-cuda`**, còn cờ của pipeline T10 là **`--profile`**. Không dùng nhầm hai cờ này.

Trace có overhead. `profile.json` được tạo khi bật `--verify-cuda` vẫn lưu kết quả và trạng thái, nhưng bị profile loader từ chối làm calibration cho scheduler. Muốn có số liệu calibration, chạy lại không bật trace.

### 3.4. Đầu ra T11

```text
logs/profiling/<run_id>/
├── config.json
├── measurements.csv
├── resources.csv
├── profile_summary.csv
├── profile.json
├── summary.json
└── trials/
    └── <model>_<device>_<repeat>/
        ├── config.json
        ├── measurements.csv
        ├── resources.csv
        ├── trial.json
        └── traces/        # Chỉ khi bật --verify-cuda
```

| File | Cách sử dụng |
|---|---|
| `measurements.csv` | Mỗi inference một dòng: model, device, repeat, iteration, timestamps, execution ms và top-1 |
| `resources.csv` | Log gộp tài nguyên, có model/device/repeat để truy ngược trial |
| `profile_summary.csv` | Bảng tổng hợp mở bằng Excel/LibreOffice: số mẫu, mean/P50/P95/P99, độ biến động giữa các lượt, RSS/VRAM |
| `profile.json` | Profile có cấu trúc để `ServiceTimeProfile.from_file(...)` nạp |
| `summary.json` | Trạng thái toàn lượt, lỗi, bảng profile và kết quả từng trial |
| `trials/.../trial.json` | PID, providers, baseline/ready/delta bộ nhớ, diagnostics và trạng thái của process tương ứng |

Thời gian đo gồm đọc ảnh, preprocess, transfer khi cần, inference đồng bộ và kiểm tra output. Loại prepare/warm-up, đợi sampler và ghi CSV khỏi duration inference.

Đọc bảng bộ nhớ:

- **Baseline:** trước load model.
- **Ready:** sau load và warm-up.
- **Delta:** ready trừ baseline, giữ nguyên dấu; trong bảng tổng hợp là median delta qua các trial thành công.
- **Observed max:** số lớn nhất lấy mẫu được trong phase inference của các trial thành công.
- RSS là toàn process, có cả thư viện/runtime; VRAM là toàn GPU và có thể gồm ứng dụng khác. Không coi delta là lượng cấp phát chính xác riêng của model.
- Sampling có thể bỏ lỡ đỉnh ngắn. Dữ liệu thiếu để trống, nguyên nhân lưu trong log và trial summary.

Profile failed/incomplete và service time không hữu hạn hoặc không dương bị loader từ chối. Có profile không đồng nghĩa Resource-aware đã được nối hoàn chỉnh vào pipeline.

## 4. Tạo báo cáo bằng make report

### 4.1. Báo cáo lượt gần nhất

```bash
make report
```

Mặc định chọn lượt **được tạo gần nhất** trong hai nhóm pipeline/profiling qua `logs/latest.json`. Đây không nhất thiết là lượt thành công gần nhất. Nếu lượt mới nhất failed, report vẫn ghi trạng thái failed và các lỗi khi có đủ artifact.

Nếu lượt mới nhất còn chạy, có thể chưa có summary hoặc đủ file để tạo report. Đợi lượt kết thúc hoặc chọn một lượt đã hoàn tất bằng `RUN`.

### 4.2. Báo cáo một lượt cụ thể

Các ví dụ sau dùng hai lượt mẫu đã được tạo trong workspace:

```bash
make report RUN=logs/pipeline/20261029T034929Z_6660920a
```

```bash
make report RUN=logs/profiling/20261029T035002Z_5305ba8b
```

Với lượt khác, thay phần sau `RUN=` bằng đường dẫn thư mục được in ở dòng `Pipeline run:` hoặc `Profiling run:`. Không nhập nguyên ký hiệu `<run_id>` vào shell; đó chỉ là placeholder trong sơ đồ thư mục.

Có thể dùng đường dẫn tuyệt đối:

```bash
make report RUN=/home/yoo/HDH/logs/profiling/20261029T035002Z_5305ba8b
```

Nếu đường dẫn có khoảng trắng, bọc toàn bộ giá trị bằng dấu nháy. Truyền thư mục của **một lượt chạy**, không truyền thư mục cha `logs/profiling`, file `summary.json` hoặc thư mục `report`.

Report nhận diện loại dữ liệu qua `config.json.run_type`, không phụ thuộc tên `pipeline` hay `profiling`. Nếu di chuyển thư mục, dùng `RUN` trỏ tới vị trí mới; pointer cũ có thể vẫn chứa đường dẫn cũ.

Đường dẫn sai, artifact bắt buộc bị thiếu hoặc loại dữ liệu không hỗ trợ sẽ báo lỗi. Lệnh không tự chọn lượt khác để thay thế và không chạy lại inference.

### 4.3. Kết quả report

```text
<run_id>/report/
├── report.md
└── figures/
```

| Loại lượt | Biểu đồ PNG |
|---|---|
| Pipeline | `queues.png`, `latency.png`, `waiting_execution.png`, `resources.png` |
| Profiling | `execution.png`, `memory.png` |

Mở `report.md` bằng Markdown Preview của trình soạn thảo để xem cả bảng và hình. Các PNG độc lập có thể chèn vào báo cáo môn học. Tạo lại report cho cùng lượt sẽ cập nhật các file report/biểu đồ tương ứng, không thay số đo gốc.

Trục thời gian pipeline tính tương đối với arrival đầu tiên. Phần chuẩn bị có thể mang giá trị thời gian âm; metrics job vẫn loại prepare/warm-up. Resource curves và queue curves là dữ liệu lấy mẫu, không ghi lại mọi biến động của inference ngắn.

### 4.4. Output tùy chỉnh và pointer gần nhất

| Pointer | Ý nghĩa |
|---|---|
| `logs/latest.json` | Lượt được tạo gần nhất giữa hai thư mục mặc định |
| `logs/pipeline/latest.json` | Lượt pipeline gần nhất |
| `logs/profiling/latest.json` | Lượt profiling gần nhất |

Các pointer lưu đường dẫn, ID, loại lượt, trạng thái và thời điểm bắt đầu. Nếu hai lượt chạy đồng thời, lượt cũ kết thúc sau không thay pointer của lượt được tạo mới hơn.

Khi dùng `--output-dir` tùy chỉnh, pointer được tạo trong thư mục tùy chỉnh đó và không cập nhật `logs/latest.json` mặc định. Chọn report bằng `RUN` hoặc đổi thư mục tra pointer:

```bash
make report ARGS="--logs-dir logs/demo-profiling"
```

## 5. Quy tắc run ID và cách đọc file

Ví dụ ID:

```text
20261029T035002Z_5305ba8b
```

| Thành phần | Ý nghĩa |
|---|---|
| `20261029` | Ngày 29/10/2026 |
| `T` | Phân cách ngày và giờ |
| `035002` | 03:50:02 |
| `Z` | UTC, tương ứng 10:50:02 tại UTC+7 |
| `5305ba8b` | 8 ký tự đầu của UUID ngẫu nhiên, không phải commit ID |

ID được tạo trước prepare/warm-up. Mỗi lượt dùng thư mục mới; tên trùng không được ghi đè. ID có độ chính xác thời gian tới giây, còn `config.json.started_at` lưu thời điểm chi tiết hơn.

Đọc JSON dạng dễ nhìn ngay trong terminal:

```bash
.venv/bin/python -m json.tool logs/pipeline/20261029T034929Z_6660920a/summary.json
```

```bash
.venv/bin/python -m json.tool logs/profiling/20261029T035002Z_5305ba8b/profile.json
```

Đọc pointer:

```bash
.venv/bin/python -m json.tool logs/latest.json
```

CSV mở bằng Excel/LibreOffice hoặc trình soạn thảo. Các trường JSON lồng bên trong CSV như `workers_json`, `queue_lengths_json`, `errors_json` phải được đọc bằng CSV parser, không tách đơn giản bằng dấu phẩy vì nội dung JSON cũng chứa dấu phẩy.

## 6. Kiểm thử và trợ giúp chi tiết

Bộ mặc định có CPU inference thật và tests giả lập; các integration GPU thật được skip nếu chưa bật biến môi trường:

```bash
make test
```

Yêu cầu cả CUDA inference và NVML thật:

```bash
T06_TEST_GPU=1 T08_TEST_GPU=1 make test
```

Xem toàn bộ lựa chọn CLI của từng module:

```bash
make run ARGS="--help"
```

```bash
make profile ARGS="--help"
```

```bash
make report ARGS="--help"
```

Nếu dùng interpreter khác, truyền biến `PYTHON`; interpreter đó phải có đầy đủ dependencies:

```bash
make help PYTHON=python3
```

## 7. Lỗi thường gặp và cách xử lý

| Hiện tượng | Kiểm tra/cách xử lý |
|---|---|
| Không tìm thấy Makefile hoặc target | Chạy ở `/home/yoo/HDH`, hoặc dùng `make -C /home/yoo/HDH ...` |
| Không có `.venv/bin/python` | Tạo `.venv`, cài `requirements.txt`, hoặc truyền `PYTHON` hợp lệ |
| Thiếu Matplotlib khi report | Cài lại dependencies từ `requirements.txt` vào đúng interpreter |
| `POLICY=resource-aware` bị từ chối | CLI hiện chỉ có cpu-only, gpu-only, round-robin |
| CPU-only vẫn lỗi GPU prepare | Pipeline hiện prepare cả hai backend; dùng CPU profiling nếu chỉ cần kiểm tra CPU |
| VRAM/utilization là `—` hoặc CSV trống | Đọc `errors_json`, resource metadata và trial summary; không tự coi đó là 0 |
| `--require-gpu-metrics` báo lỗi | Kiểm tra NVML/quyền/GPU mapping; nếu mục tiêu chỉ là inference, có thể bỏ điều kiện bắt buộc này và giữ rõ dữ liệu unavailable |
| `CUDA_VISIBLE_DEVICES` yêu cầu UUID | Chỉ rõ `--gpu-uuid` để chọn đúng GPU telemetry, hoặc dùng `--no-gpu-telemetry` |
| GPU chạy ngoài sandbox được nhưng trong sandbox lỗi | Dùng môi trường được phép truy cập GPU; không kết luận driver hỏng chỉ từ lượt bị giới hạn quyền |
| `WINDOW=0`, `UPDATE=nan`, `RUNS=0` bị từ chối | Dùng số dương; các giá trị thời gian phải hữu hạn |
| Recent không có P50/P95 | Chưa có job thành công kết thúc trong cửa sổ; xem phase, queue, failed và số mẫu |
| P95 dao động mạnh | Tăng WINDOW hoặc thu nhiều mẫu hơn; kiểm tra tải nền, không chỉ tăng tốc vẽ bảng |
| Workload timeout | Tăng `--timeout` phù hợp; chương trình vẫn drain job đã nhận và ghi lỗi |
| `make report` không có latest run | Chạy pipeline/profiling trước, hoặc truyền `RUN` tới lượt đã lưu |
| `RUN` trỏ tới lượt không tồn tại | Dùng đường dẫn in ra khi chạy; nếu đã di chuyển thư mục thì truyền vị trí mới |
| Report thiếu artifact | Đợi lượt chạy hoàn tất; kiểm tra các file được yêu cầu trong đúng thư mục lượt |
| Profile không nạp được cho scheduler | Kiểm tra status/completeness; chạy lại không bật `--verify-cuda` nếu muốn calibration |
| Writer lỗi, dropped khác 0 | Kiểm tra quyền ghi, dung lượng đĩa và diagnostics; không coi lượt đó có log đầy đủ |

CLI Python dùng exit code 0 khi thành công, 1 khi chạy/ghi log thất bại, 2 khi tham số không hợp lệ; các luồng có xử lý Ctrl+C dùng 130. Khi gọi qua Make, lỗi recipe thường làm **Make trả 2**, nên luôn đọc thông báo và `summary.json.status`, không chỉ suy từ mã thoát của Make. `--help` không tạo lượt đo.

## 8. Quy trình thực hành đề xuất

### Có CPU và GPU

Chạy batch nhỏ trước:

```bash
make run JOBS=100 WINDOW=5 UPDATE=0.5
```

Tạo report ngay sau lượt pipeline, trước khi tạo lượt mới:

```bash
make report
```

Đo profile với cấu hình thông thường:

```bash
make profile RUNS=100 REPEATS=3
```

Tạo report profiling:

```bash
make report
```

### Chỉ kiểm tra CPU

```bash
make profile DEVICE=cpu RUNS=20 REPEATS=1
```

```bash
make report
```

Các bước này kiểm tra khả năng chạy và quan sát số liệu. So sánh scheduler một cách công bằng còn cần manifest/workload giống nhau, cấu hình cố định và nhiều lần lặp; profile T11 riêng lẻ hoặc một batch T10 chưa thay thế benchmark T15.

Tài liệu liên quan: [README](README.md), [kế hoạch yêu cầu](PLAN.md), [báo cáo triển khai T10/T11](reports/261029_t10_t11_implementation.md).
