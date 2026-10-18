from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.preprocess import preprocess_image


ROOT = Path(__file__).resolve().parents[1]

IMAGE = ROOT / "inputs" / "test.jpg"

MODELS = {
    "MobileNetV2": ROOT / "models" / "mobilenetv2-12.onnx",
    "ResNet50": ROOT / "models" / "resnet50-v1-12.onnx",
}


def inspect_model(
    model_name: str,
    model_path: Path,
    x: np.ndarray,
):
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]

    print("Model       :", model_path.name)
    print("Input name  :", input_info.name)
    print("Model input :", input_info.shape)
    print("Our input   :", x.shape)
    print("dtype       :", x.dtype)

    outputs = session.run(
        None,
        {input_info.name: x},
    )

    output = outputs[0]

    print("Output shape:", output.shape)

    top1 = int(np.argmax(output[0]))

    print("Top-1 class :", top1)


def main():
    if not IMAGE.exists():
        raise FileNotFoundError(IMAGE)

    x = preprocess_image(IMAGE)

    print("Image:", IMAGE)
    print("Preprocessed shape:", x.shape)
    print("Preprocessed dtype:", x.dtype)

    for model_name, model_path in MODELS.items():
        inspect_model(
            model_name,
            model_path,
            x,
        )


if __name__ == "__main__":
    main()
