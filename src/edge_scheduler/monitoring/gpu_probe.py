"""NVML telemetry is optional; missing metrics never imply backend failure."""
from __future__ import annotations

import math

from .monitoring_models import GPU_METRICS, MIB, ProbeReading


class GPUProbe:
    def __init__(self, *, device_index: int = 0, device_uuid: str | None = None,
                 enabled: bool = True, api=None):
        if type(device_index) is not int or device_index < 0:
            raise ValueError("device_index must be an integer >= 0")
        self.index = device_index
        self.uuid = device_uuid
        self.enabled = enabled
        self.api = api
        self._handle = None
        self._initialized = False
        self._unavailable = "not_started"
        self.metadata = {}

    @staticmethod
    def _reason(exc: Exception) -> str:
        name = type(exc).__name__
        for token, reason in (
            ("NotSupported", "not_supported"), ("NoPermission", "permission_denied"),
            ("DriverNotLoaded", "driver_unavailable"), ("LibraryNotFound", "library_unavailable"),
            ("GpuIsLost", "device_lost"), ("NotFound", "device_not_found"),
        ):
            if token in name:
                return reason
        return f"nvml_error:{name}:{exc}"

    def start(self) -> None:
        if self._initialized:
            return
        self.metadata = {"nvml_index": self.index, "requested_uuid": self.uuid}
        if not self.enabled:
            self._unavailable = "disabled"
            return
        try:
            if self.api is None:
                import pynvml
                self.api = pynvml
            self.api.nvmlInit()
            self._initialized = True
            self._handle = (self.api.nvmlDeviceGetHandleByUUID(self.uuid) if self.uuid else
                            self.api.nvmlDeviceGetHandleByIndex(self.index))
            for key, method in (("name", "nvmlDeviceGetName"), ("uuid", "nvmlDeviceGetUUID")):
                try:
                    value = getattr(self.api, method)(self._handle)
                    self.metadata[key] = value.decode() if isinstance(value, bytes) else value
                except Exception as exc:
                    self.metadata[key + "_error"] = self._reason(exc)
        except ImportError:
            self._unavailable = "library_unavailable"
        except Exception as exc:
            self._unavailable = self._reason(exc)

    def read(self) -> ProbeReading:
        values = dict.fromkeys(GPU_METRICS)
        errors = {}
        if self._handle is None:
            return ProbeReading(values, dict.fromkeys(GPU_METRICS, self._unavailable))
        try:
            memory = self.api.nvmlDeviceGetMemoryInfo(self._handle)
            if not (memory.total > 0 and 0 <= memory.free <= memory.total
                    and 0 <= memory.used <= memory.total):
                raise ValueError("invalid GPU memory metrics")
            values.update(gpu_memory_total_mb=memory.total / MIB,
                          gpu_memory_used_mb=memory.used / MIB,
                          gpu_memory_free_mb=memory.free / MIB)
        except Exception as exc:
            for key in GPU_METRICS[:-1]:
                errors[key] = self._reason(exc)
        try:
            value = float(self.api.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            if math.isfinite(value) and 0 <= value <= 100:
                values["gpu_utilization_pct"] = value
            else:
                errors["gpu_utilization_pct"] = "invalid_value"
        except Exception as exc:
            errors["gpu_utilization_pct"] = self._reason(exc)
        return ProbeReading(values, errors)

    def close(self) -> None:
        self._handle = None
        if self._initialized:
            self._initialized = False
            self.api.nvmlShutdown()
