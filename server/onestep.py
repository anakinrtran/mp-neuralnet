"""One training step, taken apart: what the "One step" tab shows.

On a copy of a trained model: the guesses for one character it gets wrong, the loss, which
input pixels the loss depends on most (input gradient), how much one plain gradient-descent
step changes each layer (from this character alone, and from a batch of 128 that includes it),
and the guesses after taking that batch step, as training does. Nothing is saved; the run's
model is unchanged.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared import harness
from shared.palette import CLASSES, VISIBLE_GROUP

# Plain SGD on a batch: a single character's step overshoots (any rate tried fixed it to 100%),
# while 0.05 on a batch of 128 visibly moves the guesses, sometimes the wrong way.
STEP_LR = 0.05
BATCH = 128
TOP = 5


def _top(probs: torch.Tensor) -> list:
    p, c = probs.topk(TOP)
    return [[CLASSES[i], round(float(v), 4)] for v, i in zip(p, c)]


def _layer_changes(model: nn.Module, loss: torch.Tensor) -> list[float]:
    """Relative size of one SGD step, ||lr * grad|| / ||w||, for each layer with weights."""
    model.zero_grad()
    loss.backward()
    return [float(STEP_LR * m.weight.grad.norm() / m.weight.detach().norm().clamp_min(1e-12))
            for m in model.children() if isinstance(m, (nn.Linear, nn.Conv2d))]


def one_step(run_dir: Path, start: int = 0) -> dict:
    """Take apart one step on the first validation character at or after `start` that the
    model gets wrong (a step that fixes a mistake shows more than one on a sure guess)."""
    model = harness.load_trained_model(run_dir / "model.py", run_dir / "model.pt")
    val = np.load(harness.DATA_DIR / "val.npz")
    images = torch.from_numpy(val["x"]).float().div(255).unsqueeze(1)
    labels = torch.from_numpy(val["y"]).long()
    # mix-ups of look-alikes (0/O, ...) count as right in this course, so skip those
    wrong = np.flatnonzero(VISIBLE_GROUP[harness.predict(model, images)] != VISIBLE_GROUP[labels.numpy()])
    if len(wrong) == 0:
        raise ValueError("This model gets every validation character right.")
    index = int(wrong[np.searchsorted(wrong, start % len(labels)) % len(wrong)])
    x, y = images[index:index + 1].clone().requires_grad_(True), labels[index:index + 1]
    loss_fn = nn.CrossEntropyLoss()
    model.eval()  # no dropout or batch norm yet, but keep the step deterministic

    logits = model(x)
    loss = loss_fn(logits, y)
    before = logits.softmax(1)[0].detach()
    changes = _layer_changes(model, loss)
    saliency = x.grad[0, 0].abs()

    others = np.random.default_rng(index).choice(np.delete(np.arange(len(labels)), index), BATCH - 1, replace=False)
    batch = torch.from_numpy(np.concatenate([[index], others]))
    batch_changes = _layer_changes(model, loss_fn(model(images[batch]), labels[batch]))

    with torch.no_grad():  # the step itself, with the batch's gradients
        for p in model.parameters():
            p -= STEP_LR * p.grad
        after_logits = model(x)
    after = after_logits.softmax(1)[0]

    weighted = [n for n, m in zip(harness.describe_layers(model), model.children())
                if isinstance(m, (nn.Linear, nn.Conv2d))]
    true = int(y)
    return {
        "index": index,
        "label": CLASSES[true],
        "image": (images[index, 0] * 255).round().int().tolist(),
        "before": _top(before),
        "true_before": round(float(before[true]), 4),
        "loss": round(float(loss.detach()), 4),
        "saliency": (saliency / saliency.max().clamp_min(1e-12) * 100).round().int().tolist(),
        "layers": [{"name": n, "change": c, "batch_change": b}
                   for n, c, b in zip(weighted, changes, batch_changes)],
        "after": _top(after),
        "true_after": round(float(after[true]), 4),
        "loss_after": round(float(loss_fn(after_logits, y)), 4),
        "lr": STEP_LR,
    }
