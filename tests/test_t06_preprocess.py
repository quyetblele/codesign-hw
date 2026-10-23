import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_scheduler.inference.preprocess import MEAN, STD, preprocess_image
from edge_scheduler.inference.model_registry import repository_file


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_constant_color_matches_imagenet_normalization(self):
        path = self.root / "rgb.png"
        Image.new("RGB", (400, 280), (255, 128, 0)).save(path)
        output = preprocess_image(path)
        expected = (np.array([1, 128 / 255, 0], dtype=np.float32) - MEAN) / STD
        self.assertEqual(output.shape, (1, 3, 224, 224))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(output.flags.c_contiguous)
        self.assertTrue(np.isfinite(output).all())
        for channel in range(3):
            np.testing.assert_allclose(output[0, channel], expected[channel], atol=1e-6)

    def test_center_crop_preserves_center_for_both_orientations(self):
        for size in ((512, 256), (256, 512)):
            with self.subTest(size=size):
                pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                x, y = size[0] // 2, size[1] // 2
                pixels[y-112:y+112, x-112:x+112] = 255
                path = self.root / "center.png"
                Image.fromarray(pixels).save(path)
                output = preprocess_image(path)
                expected = (np.ones(3, dtype=np.float32) - MEAN) / STD
                np.testing.assert_allclose(output[0, :, 0, 0], expected, atol=1e-6)
                np.testing.assert_allclose(output[0, :, -1, -1], expected, atol=1e-6)

    def test_grayscale_and_rgba(self):
        for mode, color in (("L", 128), ("RGBA", (128, 128, 128, 10))):
            with self.subTest(mode=mode):
                path = self.root / f"{mode}.png"
                Image.new(mode, (180, 300), color).save(path)
                output = preprocess_image(path)
                expected = (np.full(3, 128 / 255, dtype=np.float32) - MEAN) / STD
                np.testing.assert_allclose(output[0, :, 100, 100], expected, atol=1e-6)

    def test_corrupt_image(self):
        path = self.root / "corrupt.jpg"
        path.write_bytes(b"not an image")
        with self.assertRaises(UnidentifiedImageError):
            preprocess_image(path)

    def test_repository_path_validation(self):
        path = self.root / "input.png"
        path.touch()
        self.assertEqual(repository_file(self.root, "input.png"), path)
        for invalid in ("", "  ", str(path), "../outside.jpg", None):
            with self.subTest(reference=invalid), self.assertRaises(ValueError):
                repository_file(self.root, invalid)
        with self.assertRaises(FileNotFoundError):
            repository_file(self.root, "missing.jpg")
        with self.assertRaises(FileNotFoundError):
            repository_file(self.root, ".")

    def test_symlink_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "image.jpg"
            target.touch()
            (self.root / "link.jpg").symlink_to(target)
            with self.assertRaises(ValueError):
                repository_file(self.root, "link.jpg")


if __name__ == "__main__":
    unittest.main()
