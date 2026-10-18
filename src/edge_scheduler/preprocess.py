"""Shared T03 image preprocessing for both inference devices."""
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = 224
RESIZE_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width < height:
        new_width = RESIZE_SIZE
        new_height = round(height * RESIZE_SIZE / width)
    else:
        new_height = RESIZE_SIZE
        new_width = round(width * RESIZE_SIZE / height)
    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    left = (new_width - IMAGE_SIZE) // 2
    top = (new_height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
