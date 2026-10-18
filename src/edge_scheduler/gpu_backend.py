"""CUDA implementation; initialization/execution fallback is rejected."""
from .models import Device
from .onnx_backend import OnnxBackend


class GPUBackend(OnnxBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(Device.GPU, **kwargs)
