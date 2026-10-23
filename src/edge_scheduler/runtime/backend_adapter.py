"""Publish backend lifecycle facts without taking ownership of Job state."""
from __future__ import annotations

import time

from ..core.contracts import InferenceBackend
from ..core.models import Device
from ..monitoring.monitoring_models import WorkerRegistry


class MonitoredBackend:
    def __init__(self, backend: InferenceBackend, registry: WorkerRegistry):
        self.backend = backend
        self.registry = registry

    @property
    def device(self) -> Device:
        return self.backend.device

    def prepare(self) -> None:
        self.registry.set_unavailable(self.device)
        self.backend.prepare()
        self.registry.set_ready(self.device)

    def infer(self, model_id: str, input_ref: str):
        self.registry.set_running(self.device, model_id, time.monotonic())
        try:
            output = self.backend.infer(model_id, input_ref)
        except (ValueError, OSError):
            # Invalid requests do not mean the inference device is unhealthy.
            self.registry.set_ready(self.device)
            raise
        except BaseException:
            self.registry.set_unavailable(self.device)
            raise
        self.registry.set_ready(self.device)
        return output

    def close(self) -> None:
        try:
            self.backend.close()
        finally:
            self.registry.set_unavailable(self.device)
