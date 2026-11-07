"""Encode an image into a grid of handwritten characters, and decode predictions back.

Encoding: each pixel gets the color group of its nearest palette color (see palette.py),
then a distinct EMNIST test character from that group (either class of a look-alike pair).
Decoding: each predicted class is mapped back to its palette color.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from shared.palette import VISIBLE_GROUP, group_name


def load_image(path: str | Path) -> np.ndarray:
    """Load a pixel-art image as (H, W, 3) uint8. Transparent pixels become white."""
    rgba = np.array(Image.open(path).convert("RGBA"))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] < 128] = 255  # RGB under transparent pixels is often junk
    return rgb


def classes_to_image(classes: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H, W) class indices -> (H, W, 3) image in the given (36, 3) palette."""
    return palette[classes]


def build_grid(groups: np.ndarray, pool_y: np.ndarray, seed: int,
               exclude: np.ndarray | None = None, allow_reuse: bool = False) -> np.ndarray:
    """Pick a distinct pool index for every pixel, from the pixel's color group.

    Returns an (H, W) array of indices into the character pool. `exclude` lists
    pool indices that must not be used (e.g. characters used by another grid).
    allow_reuse: a group without enough characters reuses some instead of failing
    (Workspace 4's own images, encoded with the small validation set).
    """
    rng = np.random.default_rng(seed)
    available = np.ones(len(pool_y), dtype=bool)
    if exclude is not None:
        available[exclude] = False

    pool_group = VISIBLE_GROUP[pool_y]
    flat = groups.ravel()
    picks = np.empty(flat.shape, dtype=np.int64)
    shortages = []
    for g in np.unique(flat):
        need = np.flatnonzero(flat == g)
        have = np.flatnonzero(available & (pool_group == g))
        short = len(have) < len(need)
        if short and not (allow_reuse and len(have)):
            shortages.append(f"{group_name(g)}: need {len(need)}, have {len(have)}")
            continue
        # reusing: every character is used about equally often
        picks[need] = (rng.permutation(np.resize(have, len(need))) if short
                       else rng.choice(have, size=len(need), replace=False))
    if shortages:
        raise ValueError("Not enough test characters for: " + "; ".join(shortages))
    return picks.reshape(groups.shape)


def upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor upscale for display."""
    return rgb.repeat(scale, axis=0).repeat(scale, axis=1)


def mosaic(chars: np.ndarray) -> np.ndarray:
    """(H, W, 28, 28) characters -> one (H*28, W*28) grayscale image, dark ink on white."""
    h, w, ch, cw = chars.shape
    return 255 - chars.transpose(0, 2, 1, 3).reshape(h * ch, w * cw)
