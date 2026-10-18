"""Real CPU tests by default. T06_TEST_GPU=1 requires real CUDA, never skips it."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.cpu_backend import CPUBackend
from edge_scheduler.gpu_backend import GPUBackend
from edge_scheduler.model_registry import MODEL_REGISTRY
from edge_scheduler.verification import summarize_profile


class CPUIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = CPUBackend(warmup_runs=1)
        cls.addClassCleanup(cls.backend.close)
        cls.backend.prepare()

    def test_both_models_and_repeated_mixed_jobs(self):
        reference = {}
        for _ in range(3):
            for model in MODEL_REGISTRY:
                with self.subTest(model=model):
                    output = self.backend.infer(model, "inputs/test.jpg")
                    self.assertEqual(output.shape, (1, 1000))
                    self.assertEqual(output.dtype, np.float32)
                    self.assertTrue(np.isfinite(output).all())
                    if model in reference:
                        np.testing.assert_allclose(output, reference[model], rtol=1e-5, atol=1e-5)
                    reference[model] = output

    def test_bad_input_then_success(self):
        with self.assertRaises(FileNotFoundError):
            self.backend.infer("resnet50", "inputs/does-not-exist.jpg")
        self.backend.infer("resnet50", "inputs/test.jpg")


@unittest.skipUnless(os.environ.get("T06_TEST_GPU") == "1",
                     "set T06_TEST_GPU=1 to require CUDA execution and comparison")
class GPUIntegrationTest(unittest.TestCase):
    def test_both_models_match_cpu_and_execute_cuda_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = CPUBackend(warmup_runs=1)
            gpu = GPUBackend(warmup_runs=1, profile_dir=directory)
            self.addCleanup(cpu.close)
            self.addCleanup(gpu.close)
            try:
                cpu.prepare()
                gpu.prepare()
                # One owner per backend, with CPU and GPU jobs running together.
                with ThreadPoolExecutor(max_workers=2) as workers:
                    for _ in range(2):
                        for model in MODEL_REGISTRY:
                            with self.subTest(model=model):
                                cpu_job = workers.submit(cpu.infer, model, "inputs/test.jpg")
                                gpu_job = workers.submit(gpu.infer, model, "inputs/test.jpg")
                                np.testing.assert_allclose(
                                    gpu_job.result(timeout=60), cpu_job.result(timeout=60),
                                    rtol=1e-4, atol=1e-4,
                                )
            finally:
                gpu.close()
                cpu.close()
            self.assertEqual(set(gpu.profile_paths), set(MODEL_REGISTRY))
            for model, path in gpu.profile_paths.items():
                with self.subTest(model=model):
                    self.assertTrue(summarize_profile(path)["cuda_compute_verified"])


if __name__ == "__main__":
    unittest.main()
