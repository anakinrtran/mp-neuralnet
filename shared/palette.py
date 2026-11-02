"""36-color palette: one color per class, shared by every workspace and the grader.

Look-alike pairs (0/O, 1/I, 2/Z, 8/B, 6/G, 5/S) get near-identical shades so their
confusions barely show in a decoded image. Digits have thousands of EMNIST test
characters each, so the colors that cover large areas (white background, black
outline) belong to digits.
"""

import numpy as np

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PALETTE_HEX = {
    # look-alike pairs
    "0": "#ffffff", "O": "#f3efe6",  # white / off-white
    "1": "#000000", "I": "#1c1c1c",  # black / near-black
    "2": "#5dade2", "Z": "#4f9bd0",  # sky blue
    "8": "#a9dfbf", "B": "#97cfad",  # mint
    "6": "#f8c471", "G": "#e6b25f",  # amber
    "5": "#c39bd3", "S": "#b38bc3",  # lilac
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

assert sorted(PALETTE_HEX) == sorted(CLASSES)

# PALETTE[c] is the RGB color of class index c.
PALETTE = np.array(
    [[int(PALETTE_HEX[ch][i:i + 2], 16) for i in (1, 3, 5)] for ch in CLASSES],
    dtype=np.uint8,
)


LOOKALIKES = ["0O", "1I", "2Z", "8B", "6G", "5S"]

# VISIBLE_GROUP[c] is the same for look-alike classes, so confusing them counts as
# "looks right" in the decoded image.
VISIBLE_GROUP = np.arange(len(CLASSES))
for _a, _b in LOOKALIKES:
    VISIBLE_GROUP[CLASSES.index(_b)] = CLASSES.index(_a)


def nearest_class(rgb: np.ndarray) -> np.ndarray:
    """Map an (..., 3) RGB array to the class index of the nearest palette color."""
    diff = rgb[..., None, :].astype(np.int32) - PALETTE.astype(np.int32)
    return np.argmin((diff ** 2).sum(axis=-1), axis=-1)
