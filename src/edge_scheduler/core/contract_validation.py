"""Small boundary checks for T04; no dependency on actual GPU workers."""
from __future__ import annotations
from typing import Any
from .models import Device, Job
from .contracts import InferenceBackend, JobEvent, WorkerMessage

MODEL_IDS = frozenset({"mobilenetv2", "resnet50"})

def validate_job_request(job: Job) -> None:
    if not isinstance(job.job_id, str) or not job.job_id.strip():
        raise ValueError("job_id must be non-empty str")
    if job.model_id not in MODEL_IDS:
        raise ValueError(f"unsupported model_id: {job.model_id!r}")
    if not isinstance(job.input_ref, str) or not job.input_ref.strip():
        raise ValueError("input_ref must be non-empty repo-relative path")
    if not isinstance(job.priority, int):
        raise TypeError("priority must be int")
    if job.arrival_offset_s < 0:
        raise ValueError("arrival_offset_s must be >= 0")

def validate_backend(backend: InferenceBackend, expected_device: Device) -> None:
    if not isinstance(backend, InferenceBackend):
        raise TypeError("backend does not implement the InferenceBackend protocol")
    if backend.device is not expected_device:
        raise ValueError("backend device mismatch")

def validate_backend_output(output: Any) -> None:
    # Keep numpy optional until T06 is installed; validate using numpy only here.
    import numpy as np
    if not isinstance(output, np.ndarray):
        raise TypeError("backend output must be numpy.ndarray")
    if output.shape != (1, 1000):
        raise ValueError(f"expected output shape (1, 1000), got {output.shape}")
    if not np.issubdtype(output.dtype, np.number):
        raise TypeError("backend output must contain numeric data")
