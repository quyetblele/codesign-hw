# T02 Report — Setup WSL/CUDA/ONNX Runtime và Smoke Test CPU/CUDA

## 1. Mục tiêu

Thiết lập môi trường chạy AI inference trên WSL2 và xác nhận một model ONNX chạy được trên cả `CPUExecutionProvider` và `CUDAExecutionProvider`, đồng thời xác nhận CUDA inference thực tế.

## 2. Môi trường

| Thành phần | Kết quả |
|---|---|
| OS | Ubuntu 22.04.5 LTS trên WSL2 |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce GTX 1650, 4 GB VRAM |
| NVIDIA driver | 615.78.02 |
| CUDA UMD Version | 13.4 |
| ONNX Runtime | 1.23.2 |
| ONNX Runtime providers | TensorRT, CUDA, CPU |
| CUDA Python packages | CUDA 12.x + cuDNN 9.x đã cài trong `.venv` |

## 3. Thực hiện

- Tạo Python virtual environment `.venv`.
- Cài `onnxruntime-gpu`, các CUDA/cuDNN dependencies và package `onnx`.
- Tải và kiểm tra model `mobilenetv2-12.onnx`.
- Xác nhận model hợp lệ: `ONNX model OK`, `IR version: 7`.
- Chạy smoke test với input tensor `float32`, shape `(1, 3, 224, 224)`.
- Chạy model lần lượt bằng `CPUExecutionProvider` và `CUDAExecutionProvider`.
- Cấu hình `LD_LIBRARY_PATH` để ONNX Runtime tìm được các thư viện CUDA/cuDNN được cài trong virtual environment.

## 4. Kết quả smoke test

### CPU

```text
Session providers: ['CPUExecutionProvider']
Input shape: (1, 3, 224, 224)
Output shape: (1, 1000)
Mean latency: 4.150 ms
Min latency : 3.038 ms
Max latency : 7.299 ms
```

### CUDA

```text
Session providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Input shape: (1, 3, 224, 224)
Output shape: (1, 1000)
Mean latency: 3.469 ms
Min latency : 2.690 ms
Max latency : 5.247 ms
```

CUDA session được tạo thành công với `CUDAExecutionProvider`; không còn lỗi thiếu `libcublasLt.so.12` và không fallback toàn bộ về CPU như lần kiểm tra trước.

### So sánh output

```text
Max absolute output difference: 4.7683716e-06
```

Output CPU và CUDA gần như tương đương, sai khác rất nhỏ do tính toán floating-point.

## 5. Kết luận T02

**T02 đạt.** Môi trường WSL2 đã được thiết lập; MobileNetV2 chạy thành công trên cả CPU và CUDA. Kết quả xác nhận `CUDAExecutionProvider` được khởi tạo và thực thi inference thực tế trên GPU. Cấu hình môi trường được lưu tại `logs/t02_environment.txt`.

> Lưu ý: một số node liên quan đến shape được ONNX Runtime phân bổ cho CPU theo tối ưu nội bộ; điều này không làm mất việc sử dụng CUDA cho inference.
