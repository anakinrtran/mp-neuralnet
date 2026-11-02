"""Encode a workspace's practice image into a grid of EMNIST test characters.

Writes, for workspace WS (settings in shared/workspaces.py):
  data/processed/public/practice_WS.npz     x: (32, 32, 28, 28) characters only (goes in the workspace)
  data/processed/private/practice_WS.npz    y: (32, 32) labels, idx: test-pool indices (grader only)
  questions/<question>/tests/practice_labels.npz
                                            y: labels, x_sha256: fingerprint of the characters above,
                                            so the grader can tell if its labels and characters disagree

Run:  python data/make_practice.py WS [--preview DIR]
With --preview, also saves the palette-snapped image and the character mosaic for checking.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from shared import workspaces  # noqa: E402
from shared.grids import build_grid, classes_to_image, image_to_classes, load_image, mosaic, upscale  # noqa: E402
from shared.palette import CLASSES  # noqa: E402

IMAGES = ROOT.parent / "practice_images"
PUBLIC = ROOT / "processed" / "public"
PRIVATE = ROOT / "processed" / "private"
SEED = 1


def practice_fingerprint(chars: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(chars, dtype=np.uint8).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", choices=list(workspaces.WORKSPACES))
    parser.add_argument("--preview", type=Path, help="also save preview images here")
    args = parser.parse_args()
    ws = workspaces.get(args.workspace)

    test = np.load(PRIVATE / "test.npz")
    classes = image_to_classes(load_image(IMAGES / ws.practice_image))
    idx = build_grid(classes, test["y"], seed=SEED)
    chars = test["x"][idx]

    np.savez_compressed(PUBLIC / ws.practice_file, x=chars)
    np.savez_compressed(PRIVATE / ws.practice_file, y=classes, idx=idx)
    ws.tests_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ws.tests_dir / "practice_labels.npz", y=classes,
                        x_sha256=practice_fingerprint(chars))

    counts = np.bincount(classes.ravel(), minlength=len(CLASSES))
    print(f"{ws.id} classes used:", {CLASSES[c]: int(n) for c, n in enumerate(counts) if n})

    if args.preview:
        Image.fromarray(upscale(classes_to_image(classes), 12)).save(args.preview / f"practice_{ws.id}_snapped.png")
        Image.fromarray(mosaic(chars)).save(args.preview / f"practice_{ws.id}_mosaic.png")


if __name__ == "__main__":
    main()
