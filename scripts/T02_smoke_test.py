from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mobilenetv2-12.onnx"


def make_input() -> np.ndarray:
    """Create a deterministic float32 input tensor."""
    rng = np.random.default_rng(42)

    return rng.random(
        (1, 3, 224, 224),
        dtype=np.float32,
    )


def run_inference(
    provider: str,
    session: ort.InferenceSession,
    x: np.ndarray,
    warmup: int = 5,
    runs: int = 20,
):
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(warmup):
        session.run(
            None,
            {input_name: x},
        )

    latencies_ms = []

    for _ in range(runs):
        t0 = time.perf_counter()

        outputs = session.run(
            None,
            {input_name: x},
        )

        t1 = time.perf_counter()

        latencies_ms.append(
            (t1 - t0) * 1000.0
        )

    output = outputs[0]

    print(f"\n===== {provider} =====")
    print("Session providers:",
          session.get_providers())
    print("Input name:",
          input_name)
    print("Input shape:",
          x.shape)
    print("Output shape:",
          output.shape)
    print(
        f"Mean latency: {np.mean(latencies_ms):.3f} ms"
    )
    print(
        f"Min latency : {np.min(latencies_ms):.3f} ms"
    )
    print(
        f"Max latency : {np.max(latencies_ms):.3f} ms"
    )

    return output


def create_session(provider: str):
    return ort.InferenceSession(
        str(MODEL),
        providers=[provider],
    )


def main():
    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    print("ONNX Runtime:", ort.__version__)

    print("\nAvailable providers:")
    for provider in ort.get_available_providers():
        print(" -", provider)

    x = make_input()

    # CPU
    cpu_session = create_session(
        "CPUExecutionProvider"
    )

    cpu_output = run_inference(
        "CPUExecutionProvider",
        cpu_session,
        x,
    )

    # CUDA
    cuda_session = create_session(
        "CUDAExecutionProvider"
    )

    cuda_output = run_inference(
        "CUDAExecutionProvider",
        cuda_session,
        x,
    )

    # Compare outputs
    max_diff = np.max(
        np.abs(
            cpu_output - cuda_output
        )
    )

    print("\n===== CPU vs CUDA =====")
    print(
        "Max absolute output difference:",
        max_diff,
    )


if __name__ == "__main__":
    main()