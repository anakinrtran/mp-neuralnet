"""Grade runs the way the grader will: rebuild the model from its code, load the
saved weights (no retraining), decode the practice grid and score it against the private labels.

  python scripts/grade_run.py runs/ws1/20261101-190602 [more run dirs] [--image out.png]

The workspace comes from MPNN_WORKSPACE (default ws1). For workspaces graded on another grid
(held-out or hidden), this scores only the practice grid and shows no pass/fail; use
scripts/benchmark_ws45.py for the graded grid.

Staff only: it reads the private practice labels. --image writes, for each run,
decoded | answer | wrong pixels (red), scaled up.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from shared import harness  # noqa: E402
from shared.grids import classes_to_image, upscale  # noqa: E402
from shared.palette import VISIBLE_GROUP, load_palette  # noqa: E402

LABELS = REPO / "data" / "processed" / "private" / harness.WORKSPACE.practice_file
SCALE = 8


def grade(run_dir: Path, labels: np.ndarray) -> dict:
    model = harness.load_trained_model(run_dir / "model.py", run_dir / "model.pt")
    pred = harness.predict_practice(model)
    saved = run_dir / "practice_pred.npy"
    visible = float((VISIBLE_GROUP[pred] == VISIBLE_GROUP[labels]).mean())
    return {
        "layers": harness.describe_layers(model),
        "raw": float((pred == labels).mean()),
        "visible": visible,
        # the threshold applies to the graded grid, which is the practice grid only in some workspaces
        "passed": visible >= harness.WORKSPACE.threshold if harness.WORKSPACE.graded_grid == "practice" else None,
        # the image the student saw must be the one the grader scores
        "matches_decoded": bool(np.array_equal(np.load(saved), pred)) if saved.exists() else None,
        "pred": pred,
    }


def review_image(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    wrong = VISIBLE_GROUP[pred] != VISIBLE_GROUP[labels]
    mask = np.where(wrong[..., None], [220, 40, 40], [245, 245, 245]).astype(np.uint8)
    gap = np.full((labels.shape[0] * SCALE, SCALE, 3), 255, np.uint8)
    palette = load_palette(harness.WORKSPACE.id)
    parts = [upscale(classes_to_image(pred, palette), SCALE), gap,
             upscale(classes_to_image(labels, palette), SCALE), gap, upscale(mask, SCALE)]
    return np.concatenate(parts, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--image", type=Path, help="write a review image of all runs")
    args = parser.parse_args()

    labels = np.load(LABELS)["y"]
    rows = []
    print(f"{harness.WORKSPACE.id} threshold: {harness.WORKSPACE.threshold:.1%} visible accuracy")
    for run_dir in args.runs:
        r = grade(run_dir, labels)
        seconds = json.loads((run_dir / "summary.json").read_text()).get("seconds")
        print(f"{run_dir.name}  {' > '.join(r['layers']):<38} raw {r['raw']:.3f}  visible {r['visible']:.3f}  "
              f"{'-' if r['passed'] is None else 'PASS' if r['passed'] else 'fail'}  run {seconds}s  matches decoded: {r['matches_decoded']}")
        rows.append(review_image(r["pred"], labels))

    if args.image:
        gap = np.full((SCALE, rows[0].shape[1], 3), 255, np.uint8)
        stacked = np.concatenate([x for row in rows for x in (row, gap)][:-1], axis=0)
        Image.fromarray(stacked).save(args.image)
        print("review image:", args.image)


if __name__ == "__main__":
    main()
