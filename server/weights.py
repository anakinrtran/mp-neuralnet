"""First-layer weights of a trained model, shaped as 28x28 images for display.

"templates": the model is a single Linear layer, so each of its 36 output units is
             a class template.
"features":  the first Linear layer's units (hidden features), whatever the model is.
"""

from pathlib import Path

import torch

MAX_FEATURES = 64


class WeightsUnavailable(Exception):
    pass


def first_layer_units(run_dir: Path, kind: str) -> dict:
    path = run_dir / "model.pt"
    if not path.exists():
        raise WeightsUnavailable("This run has no trained weights.")
    matrices = [w for k, w in torch.load(path, weights_only=True).items()
                if k.endswith("weight") and w.dim() == 2]
    if not matrices or matrices[0].shape[1] != 28 * 28:
        raise WeightsUnavailable("The first layer doesn't read 28x28 pixels directly.")

    first = matrices[0]
    if kind == "templates":
        if len(matrices) != 1:
            raise WeightsUnavailable("Class templates only exist for a model with a single Linear layer.")
        units = first
    elif kind == "features":
        units = first[:MAX_FEATURES]
    else:
        raise ValueError(kind)

    # scale each unit to integers in [-100, 100] so positive and negative weights show
    # with equal contrast (integers keep the JSON small). Scaling by the 98th percentile
    # instead of the max keeps a few extreme edge pixels from washing out the picture.
    scale = units.abs().quantile(0.98, dim=1, keepdim=True).clamp_min(1e-8)
    units = (units / scale).clamp(-1, 1).mul(100).round().int()
    return {"kind": kind, "total_units": first.shape[0], "units": units.tolist()}
