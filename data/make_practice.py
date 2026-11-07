"""Encode a workspace's practice image into a grid of EMNIST test characters.

Writes, for workspace WS (settings in shared/workspaces.py):
  shared/palettes/WS.json                   the image's palette (fitted the first time, or with
                                            --refit; reused otherwise so it can be tuned by hand)
  data/processed/public/practice_WS.npz     x: (32, 32, 28, 28) characters only (goes in the workspace)
  data/processed/private/practice_WS.npz    y: (32, 32) labels, idx: test-pool indices (grader only)
  questions/<question>/tests/practice_labels.npz
                                            y: labels, x_sha256: fingerprint of the characters above,
                                            so the grader can tell if its labels and characters disagree

Workspaces graded on another grid (shared/workspaces.py: graded_grid "heldout" or "hidden")
also get, from their graded_image:
  data/processed/grader/GRID_WS.npz         x: characters (copied only into the grader image)
  data/processed/private/GRID_WS.npz        y, idx, palette
  questions/<question>/tests/GRID_labels.npz
                                            y, x_sha256, palette (the grid's own fitted palette)
An image wider than MAX_GRADED_WIDTH (the class image) is shrunk to the largest of
GRADED_SIZES whose colors all fit, and the shrunk version is saved next to the source.
Graded grids use characters no other grid uses (if they fit); practice grids avoid graded ones.

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
GRADER = ROOT / "processed" / "grader"
SEED = 1
MAX_GRADED_WIDTH = 128
GRADED_SIZES = [(128, 88), (96, 66), (64, 44)]  # ~3:2, like the class image


def practice_fingerprint(chars: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(chars, dtype=np.uint8).tobytes()).hexdigest()


def used_indices(skip: str, graded_only: bool) -> np.ndarray:
    """Test-pool indices used by other saved grids. graded_only (for a practice grid): only other
    workspaces' graded grids; its own graded grid is rebuilt right after it, avoiding it."""
    ws_id = skip.split("_")[-1]
    files = [f for f in PRIVATE.glob("*_ws*.npz") if f.name != skip
             and (not graded_only or not (f.name.startswith("practice_") or f.name.endswith(ws_id)))]
    return np.concatenate([np.load(f)["idx"].ravel() for f in files] or [np.array([], int)])


def capacity(test_y: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    keep = np.ones(len(test_y), bool)
    keep[exclude] = False
    return np.bincount(VISIBLE_GROUP[test_y[keep]], minlength=len(CLASSES))


def graded_candidates(source: np.ndarray):
    """(image, note) to try, best first: the largest size, then the most colors (a large
    image like the class image is shrunk; fewer colors leave out the groups with few characters)."""
    sizes = [None] if source.shape[1] <= MAX_GRADED_WIDTH else GRADED_SIZES
    for size in sizes:
        small = source if size is None else np.array(Image.fromarray(source).resize(size, Image.LANCZOS))
        for colors in range(MAX_COLORS, 19, -2):
            yield reduce_colors(small, colors)[0], size


def make_graded(ws, test):
    """Encode the workspace's graded_image into a grid only the grader sees."""
    name = ws.graded_file
    source = load_image(IMAGES / ws.graded_image)
    exclude_all = used_indices(name, graded_only=False)
    for exclude in (exclude_all, np.array([], int)):  # disjoint from other grids if it fits at all
        for rgb, size in graded_candidates(source):
            try:
                palette = fit_palette(rgb, capacity(test["y"], exclude))
                idx = build_grid(image_to_groups(rgb, palette), test["y"], seed=SEED + 1, exclude=exclude)
            except ValueError:
                continue
            if not len(exclude) and len(exclude_all):
                print(f"warning: {name} shares characters with other grids")
            if size:
                Image.fromarray(rgb).save(IMAGES / f"{Path(ws.graded_image).stem}_{size[0]}x{size[1]}.png")
            chars, labels = test["x"][idx], test["y"][idx]
            GRADER.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(GRADER / name, x=chars)
            np.savez_compressed(PRIVATE / name, y=labels, idx=idx, palette=palette)
            np.savez_compressed(ws.tests_dir / ws.labels_file, y=labels, palette=palette,
                                x_sha256=practice_fingerprint(chars))
            print(f"{name}: {labels.shape[1]}x{labels.shape[0]} grid, {len(np.unique(rgb.reshape(-1, 3), axis=0))} "
                  f"colors, {len(np.unique(VISIBLE_GROUP[labels]))} color groups")
            return
    raise SystemExit(f"{ws.graded_image} doesn't fit at any size.")


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

    idx = build_grid(groups, test["y"], seed=SEED, exclude=used_indices(ws.practice_file, graded_only=True))
    chars, labels = test["x"][idx], test["y"][idx]

    np.savez_compressed(PUBLIC / ws.practice_file, x=chars)
    np.savez_compressed(PRIVATE / ws.practice_file, y=labels, idx=idx)
    ws.tests_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ws.tests_dir / "practice_labels.npz", y=labels,
                        x_sha256=practice_fingerprint(chars))

    counts = np.bincount(groups.ravel(), minlength=len(CLASSES))
    print(f"{ws.id} color groups used:", {group_name(g): int(n) for g, n in enumerate(counts) if n})
    if ws.graded_grid != "practice":
        make_graded(ws, test)

    if args.preview:
        side = np.full((rgb.shape[0] * 8, 8, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([upscale(load_image(IMAGES / ws.practice_image), 8), side,
                                        upscale(classes_to_image(labels, palette), 8)], axis=1)
                        ).save(args.preview / f"practice_{ws.id}_palette.png")
        Image.fromarray(mosaic(chars)).save(args.preview / f"practice_{ws.id}_mosaic.png")


if __name__ == "__main__":
    main()
