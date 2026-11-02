"""Save a contact sheet (one row per class) to check that characters are upright.

Run:  python data/preview_samples.py OUT.png
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PUBLIC = Path(__file__).resolve().parent / "processed" / "public"
PER_ROW = 12


def main(out_path: str):
    data = np.load(PUBLIC / "train.npz")
    x, y = data["x"], data["y"]
    rows = []
    for c in range(36):
        imgs = x[np.flatnonzero(y == c)[:PER_ROW]]
        rows.append(np.concatenate(list(imgs), axis=1))
    sheet = 255 - np.concatenate(rows, axis=0)  # dark ink on white
    Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.NEAREST).save(out_path)


if __name__ == "__main__":
    main(sys.argv[1])
