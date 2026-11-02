"""Encode a workspace's practice image into a grid of EMNIST test characters.

Writes, for workspace WS (settings in shared/workspaces.py):
  shared/palettes/WS.json                   the image's palette (fitted the first time, or with
                                            --refit; reused otherwise so it can be tuned by hand)
  data/processed/public/practice_WS.npz     x: (32, 32, 28, 28) characters only (goes in the workspace)
  data/processed/private/practice_WS.npz    y: (32, 32) labels, idx: test-pool indices (grader only)
  questions/<question>/tests/practice_labels.npz
                                            y: labels, x_sha256: fingerprint of the characters above,
                                            so the grader can tell if its labels and characters disagree

Run:  python data/make_practice.py WS [--refit] [--preview DIR]
With --preview, also saves the original, the image in its palette, and the character mosaic.
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
from shared.grids import build_grid, classes_to_image, load_image, mosaic, upscale  # noqa: E402
from shared.palette import (CLASSES, MAX_COLORS, PALETTES_DIR, VISIBLE_GROUP, fit_palette,  # noqa: E402
                            group_name, image_to_groups, load_palette, reduce_colors, save_palette)

IMAGES = ROOT.parent / "practice_images"
PUBLIC = ROOT / "processed" / "public"
PRIVATE = ROOT / "processed" / "private"
SEED = 1


def practice_fingerprint(chars: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(chars, dtype=np.uint8).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", choices=list(workspaces.WORKSPACES))
    parser.add_argument("--refit", action="store_true", help="fit a new palette even if one is saved")
    parser.add_argument("--preview", type=Path, help="also save preview images here")
    args = parser.parse_args()
    ws = workspaces.get(args.workspace)

    test = np.load(PRIVATE / "test.npz")
    rgb, n_colors = reduce_colors(load_image(IMAGES / ws.practice_image))
    if n_colors > MAX_COLORS:
        print(f"{ws.practice_image}: reduced {n_colors} colors to {MAX_COLORS}")

    if args.refit or not (PALETTES_DIR / f"{ws.id}.json").exists():
        capacity = np.bincount(VISIBLE_GROUP[test["y"]], minlength=len(CLASSES))
        save_palette(ws.id, fit_palette(rgb, capacity), source=ws.practice_image)
        print(f"fitted a palette: shared/palettes/{ws.id}.json")
    palette = load_palette(ws.id)
    groups = image_to_groups(rgb, palette)
    if not np.array_equal(palette[groups], rgb):
        print("warning: some image colors aren't in the saved palette; they were snapped to the "
              "nearest one. Use --refit if the image changed.")

    idx = build_grid(groups, test["y"], seed=SEED)
    chars, labels = test["x"][idx], test["y"][idx]

    np.savez_compressed(PUBLIC / ws.practice_file, x=chars)
    np.savez_compressed(PRIVATE / ws.practice_file, y=labels, idx=idx)
    ws.tests_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ws.tests_dir / "practice_labels.npz", y=labels,
                        x_sha256=practice_fingerprint(chars))

    counts = np.bincount(groups.ravel(), minlength=len(CLASSES))
    print(f"{ws.id} color groups used:", {group_name(g): int(n) for g, n in enumerate(counts) if n})

    if args.preview:
        side = np.full((rgb.shape[0] * 8, 8, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([upscale(load_image(IMAGES / ws.practice_image), 8), side,
                                        upscale(classes_to_image(labels, palette), 8)], axis=1)
                        ).save(args.preview / f"practice_{ws.id}_palette.png")
        Image.fromarray(mosaic(chars)).save(args.preview / f"practice_{ws.id}_mosaic.png")


if __name__ == "__main__":
    main()
