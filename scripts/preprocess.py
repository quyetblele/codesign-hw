from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = 224
RESIZE_SIZE = 256

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    # Resize shortest side to 256,
    # keeping aspect ratio.
    if width < height:
        new_width = RESIZE_SIZE
        new_height = round(height * RESIZE_SIZE / width)
    else:
        new_height = RESIZE_SIZE
        new_width = round(width * RESIZE_SIZE / height)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR,
    )

    # Center crop 224x224.
    left = (new_width - IMAGE_SIZE) // 2
    top = (new_height - IMAGE_SIZE) // 2

    image = image.crop(
        (
            left,
            top,
            left + IMAGE_SIZE,
            top + IMAGE_SIZE,
        )
    )

    # HWC uint8 -> HWC float32 [0, 1]
    x = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    # ImageNet normalization.
    x = (x - MEAN) / STD

    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))

    # CHW -> NCHW
    x = np.expand_dims(x, axis=0)

    return np.ascontiguousarray(
        x,
        dtype=np.float32,
    )