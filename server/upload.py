"""Workspace 5's "Your image": encode any picture in handwritten characters, the same way the
course images are encoded, and decode it with the student's model.

The picture is shrunk to at most 64 pixels on its long side, reduced to at most 30 colors and
given a fitted palette, then every pixel gets a validation character from its color group.
Validation characters, not test ones: the encoder needs labels, and test labels never enter the
workspace. There are only 100 per class (200 per look-alike pair), so a color that covers more
pixels than its group has characters reuses some; the result says which.
"""

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from shared import harness
from shared.grids import build_grid, mosaic, upscale
from shared.palette import (CLASSES, VISIBLE_GROUP, fit_palette, group_name, image_to_groups,
                            reduce_colors)

MAX_SIDE = 64
PREVIEW_SIDE = 256  # the original, shown at most this big
SCALE = 6  # display scale for 64-pixel images
MAX_BYTES = 20_000_000


def _png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _open(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise ValueError("That file is too big (over 20 MB). Try a smaller picture.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("That file couldn't be read as a picture. Try a PNG or JPEG.") from None
    rgba = np.array(img.convert("RGBA"))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] < 128] = 255  # transparent pixels count as white, as for the course images
    return Image.fromarray(rgb)


def encode_and_decode(data: bytes, run_dir: Path) -> dict:
    img = _open(data)
    preview = img.copy()
    preview.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    w, h = img.size
    k = MAX_SIDE / max(w, h)
    small = np.array(img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS) if k < 1 else img)
    reduced, colors = reduce_colors(small)

    val = np.load(harness.DATA_DIR / "val.npz")
    capacity = np.bincount(VISIBLE_GROUP[val["y"]], minlength=len(CLASSES))
    try:
        palette = fit_palette(reduced, capacity)
    except ValueError:  # a color too big for every group: fit by color alone, reuse characters
        palette = fit_palette(reduced, np.full(len(CLASSES), reduced.size))
    groups = image_to_groups(reduced, palette)
    idx = build_grid(groups, val["y"], seed=0, allow_reuse=True)
    chars, labels = val["x"][idx], val["y"][idx]
    need = np.bincount(groups.ravel(), minlength=len(CLASSES))
    reused = [f"{group_name(g)} ({need[g]} pixels, {capacity[g]} characters)"
              for g in np.flatnonzero(need > capacity)]

    model = harness.load_trained_model(run_dir / "model.py", run_dir / "model.pt")
    with torch.no_grad():
        pred = harness.predict(model, torch.from_numpy(chars.reshape(-1, 1, 28, 28))).reshape(groups.shape)
    right = VISIBLE_GROUP[pred] == VISIBLE_GROUP[labels]
    target, decoded = palette[groups], palette[pred]
    gap = np.full((groups.shape[0] * SCALE, SCALE, 3), 255, np.uint8)
    return {
        "size": [int(groups.shape[1]), int(groups.shape[0])],
        "colors_before": int(colors),
        "colors": int(len(np.unique(groups))),
        "reused": reused,
        "accuracy": float(right.mean()),
        "images": {
            "original": _png(np.array(preview)),
            "small": _png(upscale(small, SCALE)),
            "palette": _png(upscale(target, SCALE)),
            "characters": _png(mosaic(chars)),
            "decoded": _png(upscale(decoded, SCALE)),
            "side_by_side": _png(np.concatenate([upscale(target, SCALE), gap, upscale(decoded, SCALE)], axis=1)),
        },
    }
