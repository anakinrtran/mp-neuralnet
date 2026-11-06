"""Baselines for Workspaces 4 and 5: visible accuracy on the graded grid (Workspace 4's held-out
grid, Workspace 5's hidden grid) and the practice grid, per setup. Staff only: reads private labels.
Results set `threshold` in shared/workspaces.py.

    python scripts/benchmark_ws45.py ws4|ws5 [--seeds 0 1 2] [--only NAME ...]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from shared import workspaces  # noqa: E402
from shared.palette import VISIBLE_GROUP  # noqa: E402

PRIVATE = REPO / "data/processed/private"
GRADER_DATA = REPO / "data/processed/grader"

HEADER = """from torch import nn, optim
from harness import load_data, train, evaluate, decode_practice
"""
MLP = "nn.Flatten(), nn.LazyLinear(512), nn.ReLU(), {drop}nn.LazyLinear(512), nn.ReLU(), {drop}nn.LazyLinear(36)"
CNN = ("nn.LazyConv2d(32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), "
       "nn.LazyLinear(256), nn.ReLU(), {drop}nn.LazyLinear(36)")  # Workspace 4's starting model
CNN2 = ("nn.LazyConv2d(16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), "
        "nn.LazyConv2d(32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), "
        "nn.LazyLinear(128), nn.ReLU(), {drop}nn.LazyLinear(36)")
CNN1 = ("nn.LazyConv2d(16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), "
        "nn.LazyLinear(128), nn.ReLU(), {drop}nn.LazyLinear(36)")
LINEAR = "nn.Flatten(), nn.LazyLinear(36)"


def code(model, epochs, drop=0.0, wd=0.0, augment=(), patience=None, per_class=None, lr=0.001, opt="Adam"):
    load = f"load_data(train_per_class={per_class})" if per_class else "load_data()"
    extra = (f", augment={list(augment)}" if augment else "") + (f", early_stopping={patience}" if patience else "")
    wd_arg = f", weight_decay={wd}" if wd else ""
    return (HEADER + f"train_data, val_data = {load}\n"
            + "model = nn.Sequential(" + model.format(drop=f"nn.Dropout({drop}), " if drop else "") + ")\n"
            + f"optimizer = optim.Adam(model.parameters(), lr={lr}{wd_arg})\n"
            + "loss_fn = nn.CrossEntropyLoss()\n"
            + f"train(model, train_data, val_data, epochs={epochs}, batch_size=128, optimizer=optimizer, "
            + f"loss_fn=loss_fn{extra})\nevaluate(model, val_data)\ndecode_practice(model)\n")


SETUPS = {
    "ws4": {
        "start MLP": code(MLP, 20),
        "start CNN": code(CNN, 20),
        "CNN dropout 0.5": code(CNN, 20, drop=0.5),
        "CNN wd 1e-3": code(CNN, 20, wd=1e-3),
        "CNN rotate+shift": code(CNN, 20, augment=("rotate", "shift")),
        "CNN flip": code(CNN, 20, augment=("flip",)),
        "CNN early 3": code(CNN, 20, patience=3),
        "CNN dropout + rotate+shift": code(CNN, 20, drop=0.5, augment=("rotate", "shift")),
        "CNN dropout + rotate+shift 40": code(CNN, 40, drop=0.5, augment=("rotate", "shift"), patience=8),
    },
    "ws5": {  # Workspace 4's small training set (100 per class); the CNN is the starting model
        "linear": code(LINEAR, 20),
        "start MLP": code(MLP, 20),
        "CNN": code(CNN, 20),
        "CNN dropout 0.3": code(CNN, 20, drop=0.3),
        "CNN dropout 0.5": code(CNN, 20, drop=0.5),
        "CNN dropout 0.7": code(CNN, 20, drop=0.7),
        "CNN wd 1e-3": code(CNN, 20, wd=1e-3),
        "CNN rotate": code(CNN, 20, augment=("rotate",)),
        "CNN shift": code(CNN, 20, augment=("shift",)),
        "CNN rotate+shift": code(CNN, 20, augment=("rotate", "shift")),
        "CNN rotate+shift+flip": code(CNN, 20, augment=("rotate", "shift", "flip")),
        "CNN early 5": code(CNN, 20, patience=5),
        "CNN dropout + rotate+shift": code(CNN, 20, drop=0.5, augment=("rotate", "shift")),
        "CNN dropout + rotate+shift 40": code(CNN, 40, drop=0.5, augment=("rotate", "shift"), patience=8),
        "CNN2 dropout + rotate+shift 40": code(CNN2, 40, drop=0.5, augment=("rotate", "shift"), patience=8),
    },
}


def grid_accuracy(run: Path, ws, labels_name: str, chars: Path) -> float:
    """Visible accuracy of the run's model on a grid, loaded the way the grader does."""
    out = subprocess.run([sys.executable, "-c", (
        "import sys, json; sys.path.insert(0, r'%s'); from shared import harness; "
        "m = harness.load_trained_model(r'%s', r'%s'); "
        "print(json.dumps(harness.predict_grid(m, r'%s').tolist()))") % (REPO, run / "model.py", run / "model.pt", chars)],
        env={**os.environ, "MPNN_WORKSPACE": ws.id}, capture_output=True, text=True)
    if out.returncode:
        return float("nan")
    pred = np.array(json.loads(out.stdout.splitlines()[-1]))
    labels = np.load(PRIVATE / labels_name)["y"]
    return float((VISIBLE_GROUP[pred] == VISIBLE_GROUP[labels]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", choices=["ws4", "ws5"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    ws = workspaces.get(args.workspace)
    out_dir = Path(tempfile.gettempdir()) / f"mpnn_benchmark_{ws.id}"
    env = {**os.environ, "MPNN_WORKSPACE": ws.id}
    for name, source in SETUPS[ws.id].items():
        if args.only and name not in args.only:
            continue
        rows = []
        for seed in args.seeds:
            run = out_dir / f"{name.replace(' ', '_').replace('+', '')}_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "model.py").write_text(source)
            t = time.time()
            proc = subprocess.run([sys.executable, str(REPO / "shared/harness.py"), str(run / "model.py"),
                                   "--run-dir", str(run), "--seed", str(seed)], env=env, capture_output=True, text=True)
            secs = time.time() - t
            events = [json.loads(line[2:]) for line in proc.stdout.splitlines() if line.startswith("@@")]
            vals = [e for e in events if e["event"] == "val"]
            stop = next((e for e in events if e["event"] == "early_stop"), None)
            rows.append({
                "seed": seed,
                "graded": round(grid_accuracy(run, ws, ws.graded_file, GRADER_DATA / ws.graded_file), 4),
                "practice": round(grid_accuracy(run, ws, ws.practice_file, REPO / "data/processed/public" / ws.practice_file), 4),
                "val_final": vals[-1]["accuracy"] if vals else None,
                "val_best": max(v["accuracy"] for v in vals) if vals else None,
                "train_final": vals[-1].get("train_accuracy") if vals else None,
                "stopped_at": stop and stop["epoch"],
                "seconds": round(secs),
            })
        print(json.dumps({"setup": name, "runs": rows}), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
