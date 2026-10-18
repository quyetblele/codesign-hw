import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler import Device, InferenceBackend
from edge_scheduler.cpu_backend import CPUBackend
from edge_scheduler.gpu_backend import GPUBackend
from edge_scheduler.model_registry import MODEL_REGISTRY
from edge_scheduler.verification import summarize_profile


class BackendUnitTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for spec in MODEL_REGISTRY.values():
            path = self.root / spec.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        Image.new("RGB", (256, 256), "red").save(self.root / "input.png")
        self.sessions = []
        self.factory = self.enter_patch("InferenceSession", side_effect=self.make_session)
        self.providers = self.enter_patch("get_available_providers", return_value=[
            "CPUExecutionProvider", "CUDAExecutionProvider",
        ])
        self.preload = self.enter_patch("preload_dlls")

    def enter_patch(self, name, **kwargs):
        patcher = patch(f"edge_scheduler.onnx_backend.ort.{name}", **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def make_session(self, path, **kwargs):
        model_is_mobile = "mobilenet" in path
        session = MagicMock()
        requested = kwargs["providers"][0]
        provider = requested if isinstance(requested, str) else requested[0]
        session.get_providers.return_value = [provider]
        session.get_inputs.return_value = [SimpleNamespace(
            name="input" if model_is_mobile else "data",
            shape=["N", 3, 224, 224], type="tensor(float)",
        )]
        session.get_outputs.return_value = [SimpleNamespace(
            name="scores", shape=["N", 1000], type="tensor(float)",
        )]
        session.run.return_value = [np.zeros((1, 1000), dtype=np.float32)]
        session.end_profiling.return_value = str(self.root / f"{len(self.sessions)}.json")
        self.sessions.append(session)
        return session

    def backend(self, cls=CPUBackend, **kwargs):
        backend = cls(root=self.root, warmup_runs=1, **kwargs)
        self.addCleanup(backend.close)
        return backend

    def test_protocol_and_fixed_devices(self):
        for cls, device in ((CPUBackend, Device.CPU), (GPUBackend, Device.GPU)):
            backend = self.backend(cls)
            self.assertIsInstance(backend, InferenceBackend)
            self.assertIs(backend.device, device)

    def test_prepare_once_and_reuse_sessions_with_correct_input_names(self):
        backend = self.backend()
        backend.prepare()
        backend.prepare()
        self.assertEqual(self.factory.call_count, 2)
        for _ in range(3):
            for model in MODEL_REGISTRY:
                self.assertEqual(backend.infer(model, "input.png").shape, (1, 1000))
        self.assertEqual(self.factory.call_count, 2)
        for session, input_name in zip(self.sessions, ("input", "data")):
            self.assertEqual(session.run.call_count, 4)  # One warmup plus three jobs.
            self.assertEqual(list(session.run.call_args.args[1]), [input_name])
            session.disable_fallback.assert_called_once()
        for call in self.factory.call_args_list:
            self.assertFalse(call.kwargs["enable_fallback"])
            self.assertEqual(call.kwargs["providers"], ["CPUExecutionProvider"])
        self.preload.assert_not_called()

    def test_lifecycle_and_reprepare_after_close(self):
        backend = self.backend()
        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            backend.infer("mobilenetv2", "input.png")
        backend.prepare()
        backend.close()
        backend.close()
        self.assertFalse(backend.ready)
        self.assertEqual(backend.session_providers, {})
        with self.assertRaises(RuntimeError):
            backend.infer("mobilenetv2", "input.png")
        backend.prepare()
        self.assertEqual(self.factory.call_count, 4)

    def test_gpu_provider_absent(self):
        self.providers.return_value = ["CPUExecutionProvider"]
        backend = self.backend(GPUBackend)
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider.*unavailable"):
            backend.prepare()
        self.factory.assert_not_called()
        self.assertFalse(backend.ready)

    def test_gpu_session_fallback_is_rejected(self):
        def cpu_only(path, **kwargs):
            session = self.make_session(path, **kwargs)
            session.get_providers.return_value = ["CPUExecutionProvider"]
            return session
        self.factory.side_effect = cpu_only
        backend = self.backend(GPUBackend)
        with self.assertRaisesRegex(RuntimeError, "fallback rejected"):
            backend.prepare()
        self.assertFalse(backend.ready)
        self.assertEqual(backend.session_providers, {})

    def test_partial_prepare_failure_cleans_up_and_can_retry(self):
        def failing(path, **kwargs):
            if "resnet" in path:
                raise RuntimeError("broken second model")
            return self.make_session(path, **kwargs)
        self.factory.side_effect = failing
        backend = self.backend(profile_dir=self.root / "profiles")
        with self.assertRaisesRegex(RuntimeError, "broken second model"):
            backend.prepare()
        self.assertFalse(backend.ready)
        self.assertEqual(backend.session_providers, {})
        self.sessions[0].end_profiling.assert_called_once()
        self.factory.side_effect = self.make_session
        backend.prepare()
        self.assertTrue(backend.ready)

    def test_missing_model_does_not_create_sessions(self):
        (self.root / MODEL_REGISTRY["resnet50"].filename).unlink()
        backend = self.backend()
        with self.assertRaisesRegex(RuntimeError, "file not found"):
            backend.prepare()
        self.factory.assert_not_called()

    def test_bad_metadata_is_rejected(self):
        for shape, dtype in (([1, 224, 224, 3], "tensor(float)"),
                             ([1, 3, 224, 224], "tensor(double)")):
            def malformed(path, **kwargs):
                session = self.make_session(path, **kwargs)
                session.get_inputs.return_value[0].shape = shape
                session.get_inputs.return_value[0].type = dtype
                return session
            self.factory.side_effect = malformed
            backend = self.backend()
            with self.subTest(shape=shape, dtype=dtype), self.assertRaisesRegex(RuntimeError, "invalid"):
                backend.prepare()
            self.assertFalse(backend.ready)

    def test_bad_requests_do_not_poison_next_job(self):
        backend = self.backend()
        backend.prepare()
        for model in ("unknown", None, []):
            with self.subTest(model=model), self.assertRaises(ValueError):
                backend.infer(model, "input.png")
        with self.assertRaises(FileNotFoundError):
            backend.infer("mobilenetv2", "missing.png")
        with self.assertRaises(ValueError):
            backend.infer("mobilenetv2", "../outside.png")
        (self.root / "bad.jpg").write_bytes(b"bad image")
        with self.assertRaises(OSError):
            backend.infer("mobilenetv2", "bad.jpg")
        backend.infer("mobilenetv2", "input.png")

    def test_input_root_is_independent_of_cwd(self):
        backend = self.backend()
        backend.prepare()
        previous = Path.cwd()
        try:
            os.chdir(self.root / "models")
            backend.infer("resnet50", "input.png")
        finally:
            os.chdir(previous)

    def test_gpu_run_failure_is_not_retried_on_cpu(self):
        backend = self.backend(GPUBackend, device_id=2)
        backend.prepare()
        self.preload.assert_called_once()
        self.assertEqual(self.factory.call_args.kwargs["providers"][0],
                         ("CUDAExecutionProvider", {"device_id": 2}))
        session = self.sessions[0]
        count = session.run.call_count
        session.run.side_effect = RuntimeError("CUDA device lost")
        with self.assertRaisesRegex(RuntimeError, "gpu inference.*CUDA device lost"):
            backend.infer("mobilenetv2", "input.png")
        self.assertEqual(session.run.call_count, count + 1)
        self.assertEqual(self.factory.call_count, 2)
        session.set_providers.assert_not_called()

    def test_invalid_output_is_rejected(self):
        backend = self.backend()
        backend.prepare()
        for output in (np.zeros((1, 999), np.float32), np.zeros((1, 1000), np.float64),
                       np.full((1, 1000), np.nan, np.float32), "invalid"):
            self.sessions[0].run.return_value = [output]
            with self.subTest(output_type=type(output)), self.assertRaises(RuntimeError):
                backend.infer("mobilenetv2", "input.png")

    def test_profile_finalization_failure_still_releases_sessions(self):
        backend = self.backend(profile_dir=self.root / "profiles")
        backend.prepare()
        self.sessions[0].end_profiling.side_effect = RuntimeError("disk error")
        with self.assertRaisesRegex(RuntimeError, "disk error"):
            backend.close()
        self.sessions[1].end_profiling.assert_called_once()
        self.assertFalse(backend.ready)
        self.assertEqual(backend.session_providers, {})
        backend.close()

    def test_invalid_configuration(self):
        for options in ({"warmup_runs": 0}, {"intra_op_num_threads": 0},
                        {"device_id": -1}, {"warmup_runs": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CPUBackend(**options)

    def test_trace_requires_cuda_compute_not_copy_or_provider_list(self):
        path = self.root / "trace.json"
        events = [
            {"cat": "Session", "args": {"provider": "CUDAExecutionProvider"}},
            {"cat": "Node", "args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}},
            {"cat": "Node", "args": {"provider": "CUDAExecutionProvider", "op_name": "MemcpyFromHost"}},
        ]
        path.write_text(json.dumps(events))
        self.assertFalse(summarize_profile(path)["cuda_compute_verified"])
        events.append({"cat": "Node", "args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"}})
        path.write_text(json.dumps(events))
        self.assertTrue(summarize_profile(path)["cuda_compute_verified"])


if __name__ == "__main__":
    unittest.main()
