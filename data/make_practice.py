"""Encode the Workspace 1 practice image into a grid of EMNIST test characters.

Writes:
  data/processed/public/practice_ws1.npz    x: (32, 32, 28, 28) characters only (goes in the workspace)
  data/processed/private/practice_ws1.npz   y: (32, 32) labels, idx: test-pool indices (grader only)
  questions/ws1-basic-network/tests/practice_labels.npz
                                            y: labels, x_sha256: fingerprint of the characters above,
                                            so the grader can tell if its labels and characters disagree

Run:  python data/make_practice.py [PREVIEW_DIR]
With PREVIEW_DIR, also saves the palette-snapped image and the character mosaic for checking.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from shared.grids import build_grid, classes_to_image, image_to_classes, load_image, mosaic, upscale  # noqa: E402
from shared.palette import CLASSES  # noqa: E402

SOURCE = ROOT.parent / "practice_images" / "workspace1_dog.png"
PUBLIC = ROOT / "processed" / "public"
PRIVATE = ROOT / "processed" / "private"
QUESTION_TESTS = ROOT.parent / "questions" / "ws1-basic-network" / "tests"
SEED = 1


def practice_fingerprint(chars: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(chars, dtype=np.uint8).tobytes()).hexdigest()


def main(preview_dir: str | None):
    test = np.load(PRIVATE / "test.npz")
    classes = image_to_classes(load_image(SOURCE))
    idx = build_grid(classes, test["y"], seed=SEED)
    chars = test["x"][idx]

    np.savez_compressed(PUBLIC / "practice_ws1.npz", x=chars)
    np.savez_compressed(PRIVATE / "practice_ws1.npz", y=classes, idx=idx)
    QUESTION_TESTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(QUESTION_TESTS / "practice_labels.npz", y=classes,
                        x_sha256=practice_fingerprint(chars))

    counts = np.bincount(classes.ravel(), minlength=len(CLASSES))
    print("classes used:", {CLASSES[c]: int(n) for c, n in enumerate(counts) if n})

    if preview_dir:
        out = Path(preview_dir)
        Image.fromarray(upscale(classes_to_image(classes), 12)).save(out / "practice_snapped.png")
        Image.fromarray(mosaic(chars)).save(out / "practice_mosaic.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
