"""Build the 36-class EMNIST dataset (digits 0-9 + uppercase A-Z).

Downloads EMNIST ByClass once with torchvision (needs network), then writes:

  data/processed/public/train.npz   balanced training pool (baked into the workspace)
  data/processed/public/val.npz     balanced validation set (baked into the workspace)
  data/processed/private/test.npz   all 36-class test characters (course side only;
                                    practice and hidden grids are drawn from here)

Run:  python data/prepare_emnist.py
"""

from pathlib import Path

import numpy as np
from torchvision.datasets import EMNIST

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
PUBLIC = ROOT / "processed" / "public"
PRIVATE = ROOT / "processed" / "private"

NUM_CLASSES = 36  # ByClass labels 0-9 are digits, 10-35 are A-Z, 36-61 are a-z
VAL_PER_CLASS = 100
SEED = 124


def load_split(train: bool) -> tuple[np.ndarray, np.ndarray]:
    ds = EMNIST(RAW, split="byclass", train=train, download=True)
    x = ds.data.numpy()
    y = ds.targets.numpy()
    keep = y < NUM_CLASSES
    # EMNIST is stored transposed relative to MNIST; flip it back so characters are upright.
    x = np.ascontiguousarray(x[keep].transpose(0, 2, 1))
    return x.astype(np.uint8), y[keep].astype(np.int64)


def balanced_split(x, y, rng):
    """Return (train_x, train_y, val_x, val_y) with equal counts per class."""
    per_class = np.bincount(y, minlength=NUM_CLASSES)
    train_per_class = per_class.min() - VAL_PER_CLASS
    tr_idx, va_idx = [], []
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(y == c))
        va_idx.append(idx[:VAL_PER_CLASS])
        tr_idx.append(idx[VAL_PER_CLASS:VAL_PER_CLASS + train_per_class])
    tr_idx = rng.permutation(np.concatenate(tr_idx))
    va_idx = rng.permutation(np.concatenate(va_idx))
    return x[tr_idx], y[tr_idx], x[va_idx], y[va_idx]


def main():
    rng = np.random.default_rng(SEED)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)

    x, y = load_split(train=True)
    print("train split counts per class:", np.bincount(y, minlength=NUM_CLASSES).tolist())
    tx, ty, vx, vy = balanced_split(x, y, rng)
    np.savez_compressed(PUBLIC / "train.npz", x=tx, y=ty)
    np.savez_compressed(PUBLIC / "val.npz", x=vx, y=vy)
    print(f"train pool: {len(ty)} ({len(ty) // NUM_CLASSES}/class), val: {len(vy)}")

    x, y = load_split(train=False)
    print("test split counts per class:", np.bincount(y, minlength=NUM_CLASSES).tolist())
    np.savez_compressed(PRIVATE / "test.npz", x=x, y=y)
    print(f"test pool: {len(y)}")


if __name__ == "__main__":
    main()
