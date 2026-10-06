# T03 Report — Chọn benchmark models và chuẩn hóa input

## 1. Mục tiêu

T03 thực hiện việc chọn hai pretrained ONNX models có chi phí xử lý khác nhau và chuẩn hóa input/preprocessing để dùng chung cho benchmark CPU/GPU.

Theo kế hoạch project, deliverable của T03 gồm:
- Hai benchmark models.
- Input mẫu.
- Preprocessing dùng chung.
- Xác nhận hai model nhận được input chuẩn và chạy inference thành công.

## 2. Models được chọn

| Model | File | Loại |
|---|---|---|
| MobileNetV2 | `models/mobilenetv2-12.onnx` | Image classification |
| ResNet-50 | `models/resnet50-v1-12.onnx` | Image classification |

Hai model đều sử dụng output 1000 lớp classification.

## 3. Input và preprocessing

Input mẫu:

```text
inputs/test.jpg
```

Pipeline preprocessing dùng chung:

```text
JPEG
  ↓
RGB
  ↓
Resize giữ tỉ lệ
  ↓
Shortest side = 256
  ↓
Center crop
  ↓
224 × 224
  ↓
Chuyển sang float32 và scale [0, 1]
  ↓
ImageNet normalization
  ↓
HWC → CHW
  ↓
Thêm batch dimension
  ↓
(1, 3, 224, 224)
```

Thông số input cuối cùng:

```text
Shape : (1, 3, 224, 224)
Dtype : float32
```

Preprocessing được triển khai dùng chung trong:

```text
scripts/preprocess.py
```

## 4. Kiểm tra thực tế

Lệnh chạy:

```bash
python scripts/T03_model_test.py
```

### MobileNetV2

```text
Model       : mobilenetv2-12.onnx
Input name  : input
Model input : ['batch_size', 3, 224, 224]
Our input   : (1, 3, 224, 224)
dtype       : float32
Output shape: (1, 1000)
Top-1 class : 3
```

### ResNet-50

```text
Model       : resnet50-v1-12.onnx
Input name  : data
Model input : ['N', 3, 224, 224]
Our input   : (1, 3, 224, 224)
dtype       : float32
Output shape: (1, 1000)
Top-1 class : 3
```

## 5. Kết luận

T03 đã hoàn thành.

Kết quả thực tế chứng minh:

- Đã chọn đủ 2 pretrained ONNX models.
- Đã chuẩn bị input mẫu `test.jpg`.
- Hai model sử dụng cùng pipeline preprocessing.
- Input sau preprocessing có shape `(1, 3, 224, 224)` và dtype `float32`.
- MobileNetV2 chạy inference thành công và trả output `(1, 1000)`.
- ResNet-50 chạy inference thành công và trả output `(1, 1000)`.

T03 đủ điều kiện để chuyển sang các task tích hợp worker/scheduler tiếp theo.
