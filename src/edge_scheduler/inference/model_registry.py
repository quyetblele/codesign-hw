"""The two T03 models and repository-relative asset resolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


# This module lives at <project>/src/edge_scheduler/inference/model_registry.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 1000)


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    input_shape: tuple[int, ...] = INPUT_SHAPE
    output_shape: tuple[int, ...] = OUTPUT_SHAPE


MODEL_REGISTRY = MappingProxyType({
    "mobilenetv2": ModelSpec("models/mobilenetv2-12.onnx"),
    "resnet50": ModelSpec("models/resnet50-v1-12.onnx"),
})


def repository_file(root: Path, reference: str) -> Path:
    """Resolve an existing file under root, including symlink containment."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("file reference must be a non-empty repository-relative str")
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("absolute file references are not allowed")
    root = root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"file reference escapes project root: {reference!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"file not found: {reference!r} (root: {root})")
    return resolved
