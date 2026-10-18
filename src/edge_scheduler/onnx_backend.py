"""Reusable sessions owned by one worker; no Job or queue mutation.

Use each instance serially from its owning worker. prepare/close must not race
with infer. Importing edge_scheduler's contract package does not import ORT.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import onnxruntime as ort

from .contract_validation import validate_backend_output
from .model_registry import INPUT_SHAPE, MODEL_REGISTRY, PROJECT_ROOT, repository_file
from .models import Device
from .preprocess import preprocess_image


class OnnxBackend:
    def __init__(
        self,
        device: Device,
        *,
        root: str | Path = PROJECT_ROOT,
        warmup_runs: int = 5,
        intra_op_num_threads: int = 2,
        device_id: int = 0,
        profile_dir: str | Path | None = None,
        preload_cuda: bool = True,
    ) -> None:
        if not isinstance(device, Device):
            raise ValueError("device must be Device.CPU or Device.GPU")
        for name, value, minimum in (
            ("warmup_runs", warmup_runs, 1),
            ("intra_op_num_threads", intra_op_num_threads, 1),
            ("device_id", device_id, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self._device = device
        self.root = Path(root).resolve()
        self.warmup_runs = warmup_runs
        self.intra_op_num_threads = intra_op_num_threads
        self.device_id = device_id
        self.profile_dir = Path(profile_dir).resolve() if profile_dir is not None else None
        self.preload_cuda = preload_cuda
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._input_names: dict[str, str] = {}
        self._output_names: dict[str, str] = {}
        self._profile_paths: dict[str, str] = {}
        self._ready = False

    @property
    def device(self) -> Device:
        return self._device

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def session_providers(self) -> dict[str, list[str]]:
        return {model: list(session.get_providers()) for model, session in self._sessions.items()}

    @property
    def profile_paths(self) -> dict[str, str]:
        """Trace paths become available after close(); retained until prepare()."""
        return dict(self._profile_paths)

    @property
    def _provider(self) -> str:
        return "CUDAExecutionProvider" if self.device is Device.GPU else "CPUExecutionProvider"

    def _check_provider(self, session: ort.InferenceSession) -> None:
        actual = session.get_providers()
        if not actual or actual[0] != self._provider:
            raise RuntimeError(f"expected {self._provider} first, got {actual}; device fallback rejected")

    @staticmethod
    def _check_metadata(metadata, expected: tuple[int, ...], label: str) -> None:
        shape = metadata.shape
        if metadata.type != "tensor(float)" or len(shape) != len(expected):
            raise ValueError(f"invalid {label}: expected float32 {expected}, got {metadata.type} {shape}")
        for index, (actual, wanted) in enumerate(zip(shape, expected)):
            # Only the batch dimension is allowed to be dynamic for T03 models.
            if actual != wanted and not (index == 0 and (actual is None or isinstance(actual, str))):
                raise ValueError(f"invalid {label} shape: expected {expected}, got {shape}")

    @staticmethod
    def _validate_output(output: np.ndarray) -> None:
        validate_backend_output(output)
        if output.dtype != np.float32 or not np.isfinite(output).all():
            raise ValueError("inference output must be finite float32 scores")

    def prepare(self) -> None:
        """Load/warm both models atomically; a second call while ready is a no-op."""
        if self.ready:
            return
        self._profile_paths.clear()
        try:
            if self._provider not in ort.get_available_providers():
                raise RuntimeError(f"{self._provider} is unavailable in this ONNX Runtime build")
            paths = {model: repository_file(self.root, spec.filename)
                     for model, spec in MODEL_REGISTRY.items()}
            if self.device is Device.GPU and self.preload_cuda:
                # ORT >= 1.21: load CUDA/cuDNN from installed NVIDIA site packages.
                # System libraries remain available through the normal loader.
                ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
            if self.profile_dir is not None:
                self.profile_dir.mkdir(parents=True, exist_ok=True)
            generation = uuid4().hex[:12]
            for model, path in paths.items():
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.intra_op_num_threads
                options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                if self.profile_dir is not None:
                    options.enable_profiling = True
                    options.profile_file_prefix = str(
                        self.profile_dir / f"{self.device.value}_{model}_{generation}"
                    )
                providers = [self._provider] if self.device is Device.CPU else [
                    (self._provider, {"device_id": self.device_id}),
                    "CPUExecutionProvider",  # ORT may place small shape nodes on CPU.
                ]
                session = ort.InferenceSession(
                    str(path), sess_options=options, providers=providers,
                    enable_fallback=False,  # Supported by the pinned ORT 1.23.2.
                )
                self._sessions[model] = session
                session.disable_fallback()
                self._check_provider(session)
                inputs, outputs = session.get_inputs(), session.get_outputs()
                if len(inputs) != 1 or len(outputs) != 1:
                    raise ValueError(f"{model} must have one input and one output")
                spec = MODEL_REGISTRY[model]
                self._check_metadata(inputs[0], spec.input_shape, f"{model} input")
                self._check_metadata(outputs[0], spec.output_shape, f"{model} output")
                self._input_names[model] = inputs[0].name
                self._output_names[model] = outputs[0].name
                tensor = np.zeros(INPUT_SHAPE, dtype=np.float32)
                for _ in range(self.warmup_runs):
                    output = session.run([outputs[0].name], {inputs[0].name: tensor})[0]
                    self._validate_output(output)
                self._check_provider(session)
            self._ready = True
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"{self.device.value} prepare failed: {exc}; cleanup failed: {cleanup_error}"
                ) from exc
            raise RuntimeError(f"{self.device.value} prepare failed: {exc}") from exc

    def infer(self, model_id: str, input_ref: str) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("backend is not prepared; call prepare() before infer()")
        if not isinstance(model_id, str) or model_id not in MODEL_REGISTRY:
            raise ValueError(f"unsupported model_id: {model_id!r}")
        image_path = repository_file(self.root, input_ref)
        tensor = preprocess_image(image_path)
        session = self._sessions[model_id]
        try:
            self._check_provider(session)
            output = session.run(
                [self._output_names[model_id]], {self._input_names[model_id]: tensor}
            )[0]
            self._check_provider(session)
            self._validate_output(output)
            return output
        except Exception as exc:
            raise RuntimeError(f"{self.device.value} inference failed for {model_id}: {exc}") from exc

    def close(self) -> None:
        """End traces and release all session references, even if finalizing fails."""
        self._ready = False
        errors = []
        try:
            if self.profile_dir is not None:
                for model, session in self._sessions.items():
                    try:
                        self._profile_paths[model] = session.end_profiling()
                    except Exception as exc:
                        errors.append(f"{model}: {exc}")
        finally:
            self._sessions.clear()
            self._input_names.clear()
            self._output_names.clear()
        if errors:
            raise RuntimeError("could not finalize profiling: " + "; ".join(errors))
