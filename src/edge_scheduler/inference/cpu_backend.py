"""CPU implementation of the T04 InferenceBackend protocol."""
from ..core.models import Device
from .onnx_backend import OnnxBackend


class CPUBackend(OnnxBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(Device.CPU, **kwargs)
