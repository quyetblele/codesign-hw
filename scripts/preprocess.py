"""Compatibility import for T03; new code uses edge_scheduler.preprocess."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_scheduler.preprocess import (  # noqa: E402,F401
    IMAGE_SIZE, RESIZE_SIZE, MEAN, STD, preprocess_image,
)
