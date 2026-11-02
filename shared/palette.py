"""Class colors. Each practice image gets its own palette, fitted to its colors.

How fit_palette() fits a palette to an image:
- Look-alike pairs (0/O, 1/I, 2/Z, 8/B, 6/G, 5/S) always share one color, so confusing
  them doesn't show in a decoded image, and a large area can use characters of both
  classes. That leaves 30 color groups, so an image can have at most 30 colors.
- Images with more than 30 colors are reduced to 30 first (reduce_colors).
- Image colors are assigned largest area first, each to the free group that has enough
  EMNIST test characters and whose default color is closest.
- Groups the image doesn't use keep their default color, so a wrong prediction into one
  of them shows up as an off-color pixel.

Course images' palettes are saved in shared/palettes/<workspace>.json by
data/make_practice.py, so they stay fixed and can be tuned by hand.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PALETTES_DIR = Path(__file__).resolve().parent / "palettes"

# Default color of each class: used for classes an image doesn't use.
DEFAULT_HEX = {
    # look-alike pairs (each pair always shares its first class's color)
    "0": "#ffffff", "O": "#ffffff",  # white
    "1": "#000000", "I": "#000000",  # black
    "2": "#5dade2", "Z": "#5dade2",  # sky blue
    "8": "#a9dfbf", "B": "#a9dfbf",  # mint
    "6": "#f8c471", "G": "#f8c471",  # amber
    "5": "#c39bd3", "S": "#c39bd3",  # lilac
    # gray ramp
    "3": "#303030", "4": "#4a4a4a", "7": "#646464", "9": "#7e7e7e",
    "A": "#989898", "C": "#b2b2b2", "D": "#cccccc", "E": "#e0e0e0",
    # reds, pinks, browns (Y is a dusty rose)
    "F": "#8b2a2e", "H": "#c0392b", "J": "#e8806e", "K": "#f5b7b1",
    "Y": "#b4585c", "M": "#8d5524", "N": "#f1c27d",
    # yellows, greens, blues, purples
    "P": "#f1c40f", "L": "#e67e22", "Q": "#27ae60", "R": "#145a32",
    "T": "#1abc9c", "U": "#2e86c1", "V": "#1b2f6e", "W": "#6c3483",
    "X": "#bb8fce",
}
assert sorted(DEFAULT_HEX) == sorted(CLASSES)

LOOKALIKES = ["0O", "1I", "2Z", "8B", "6G", "5S"]

# VISIBLE_GROUP[c] is the color group of class c: the same for look-alike classes, so
# confusing them counts as "looks right" in the decoded image.
VISIBLE_GROUP = np.arange(len(CLASSES))
for _a, _b in LOOKALIKES:
    VISIBLE_GROUP[CLASSES.index(_b)] = CLASSES.index(_a)
GROUPS = np.unique(VISIBLE_GROUP)  # 30 group ids (each is its first class's index)
MAX_COLORS = len(GROUPS)


def hex_to_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in rgb)


DEFAULT_PALETTE = np.array([hex_to_rgb(DEFAULT_HEX[ch]) for ch in CLASSES], dtype=np.uint8)


def group_name(g: int) -> str:
    """"0/O" for a look-alike group, "3" for a single class."""
    return "/".join(CLASSES[c] for c in np.flatnonzero(VISIBLE_GROUP == g))


def reduce_colors(rgb: np.ndarray, max_colors: int = MAX_COLORS) -> tuple[np.ndarray, int]:
    """Reduce an image to at most max_colors colors (median cut). Returns (image, colors before)."""
    before = len(np.unique(rgb.reshape(-1, 3), axis=0))
    if before <= max_colors:
        return rgb, before
    reduced = Image.fromarray(rgb).quantize(max_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    return np.array(reduced), before


def fit_palette(rgb: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Fit a palette to an image with at most MAX_COLORS colors.

    capacity[g] is how many test characters color group g has. Returns a (36, 3) palette.
    """
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    if len(colors) > MAX_COLORS:
        raise ValueError(f"The image has {len(colors)} colors; reduce it to {MAX_COLORS} first.")
    palette = DEFAULT_PALETTE.copy()
    free = set(GROUPS.tolist())
    for i in np.argsort(-counts, kind="stable"):
        fits = [g for g in free if capacity[g] >= counts[i]]
        if not fits:
            raise ValueError(f"No color group has enough test characters for {counts[i]} pixels "
                             f"of {rgb_to_hex(colors[i])}.")
        dist = [((DEFAULT_PALETTE[g].astype(int) - colors[i]) ** 2).sum() for g in fits]
        g = fits[int(np.argmin(dist))]
        palette[VISIBLE_GROUP == g] = colors[i]
        free.remove(g)
    return palette


def image_to_groups(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H, W, 3) image -> (H, W) color group of each pixel's nearest palette color."""
    diff = rgb[..., None, :].astype(np.int32) - palette.astype(np.int32)
    return VISIBLE_GROUP[np.argmin((diff ** 2).sum(axis=-1), axis=-1)]


def save_palette(ws_id: str, palette: np.ndarray, source: str):
    PALETTES_DIR.mkdir(exist_ok=True)
    data = {"source": source, "colors": {ch: rgb_to_hex(palette[i]) for i, ch in enumerate(CLASSES)}}
    (PALETTES_DIR / f"{ws_id}.json").write_text(json.dumps(data, indent=2) + "\n")


def load_palette(ws_id: str) -> np.ndarray:
    """A workspace's (36, 3) palette, from shared/palettes/<ws_id>.json."""
    path = PALETTES_DIR / f"{ws_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No palette for {ws_id}. Run: python data/make_practice.py {ws_id}")
    colors = json.loads(path.read_text())["colors"]
    return np.array([hex_to_rgb(colors[ch]) for ch in CLASSES], dtype=np.uint8)
