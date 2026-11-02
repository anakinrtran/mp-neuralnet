"""Encode an image into a grid of handwritten characters, and decode predictions back.

Encoding: each pixel is snapped to the nearest palette color (its class), then
replaced by a distinct EMNIST test character of that class.
Decoding: each predicted class is mapped back to its palette color.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from shared.palette import CLASSES, PALETTE, nearest_class


def load_image(path: str | Path) -> np.ndarray:
    """Load a pixel-art image as (H, W, 3) uint8. Transparent pixels become white."""
    rgba = np.array(Image.open(path).convert("RGBA"))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] < 128] = 255  # RGB under transparent pixels is often junk
    return rgb


def image_to_classes(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) image -> (H, W) class indices."""
    return nearest_class(rgb)


def classes_to_image(classes: np.ndarray) -> np.ndarray:
    """(H, W) class indices -> (H, W, 3) palette image."""
    return PALETTE[classes]


def build_grid(classes: np.ndarray, pool_y: np.ndarray, seed: int,
               exclude: np.ndarray | None = None) -> np.ndarray:
    """Pick a distinct pool index for every pixel, matching its class.

    Returns an (H, W) array of indices into the character pool. `exclude` lists
    pool indices that must not be used (e.g. characters used by another grid).
    """
    rng = np.random.default_rng(seed)
    available = np.ones(len(pool_y), dtype=bool)
    if exclude is not None:
        available[exclude] = False

    flat = classes.ravel()
    picks = np.empty(flat.shape, dtype=np.int64)
    shortages = []
    for c in np.unique(flat):
        need = np.flatnonzero(flat == c)
        have = np.flatnonzero(available & (pool_y == c))
        if len(have) < len(need):
            shortages.append(f"{CLASSES[c]}: need {len(need)}, have {len(have)}")
            continue
        picks[need] = rng.choice(have, size=len(need), replace=False)
    if shortages:
        raise ValueError("Not enough test characters for: " + "; ".join(shortages))
    return picks.reshape(classes.shape)


def upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor upscale for display."""
    return rgb.repeat(scale, axis=0).repeat(scale, axis=1)


def mosaic(chars: np.ndarray) -> np.ndarray:
    """(H, W, 28, 28) characters -> one (H*28, W*28) grayscale image, dark ink on white."""
    h, w, ch, cw = chars.shape
    return 255 - chars.transpose(0, 2, 1, 3).reshape(h * ch, w * cw)
