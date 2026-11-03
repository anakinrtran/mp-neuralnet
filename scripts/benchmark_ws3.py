"""Baselines for Workspace 3: visible practice accuracy (2 epochs) per training setup.
Staff only: reads the private practice labels. Results set `threshold` in shared/workspaces.py.

    python scripts/benchmark_ws3.py [SEED ...]     (default seeds 0 1 2)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from shared.palette import VISIBLE_GROUP  # noqa: E402

OUT = Path(tempfile.gettempdir()) / "mpnn_benchmark_ws3"
LABELS = np.load(REPO / "data/processed/private/practice_ws3.npz")["y"]

MODEL = """nn.LazyConv2d(16, kernel_size=3, padding=1), {bn2}nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.LazyLinear(128), {bn1}nn.ReLU(), nn.LazyLinear(36)"""
CODE = """from torch import nn, optim
from harness import load_data, train, evaluate, decode_practice
train_data, val_data = load_data()
model = nn.Sequential({model})
optimizer = {opt}
{sched}loss_fn = {loss}
train(model, train_data, val_data, epochs=2, batch_size={batch}, optimizer=optimizer, loss_fn=loss_fn{sched_arg})
evaluate(model, val_data)
decode_practice(model)
"""
STEPS = 666  # per epoch at batch 128

SETUPS = {
    "start: SGD lr 10": dict(opt="optim.SGD(model.parameters(), lr=10)"),
    "SGD lr 0.1": dict(opt="optim.SGD(model.parameters(), lr=0.1)"),
    "SGD+mom lr 0.05": dict(opt="optim.SGD(model.parameters(), lr=0.05, momentum=0.9)"),
    "Adam 1e-3 (WS2 default)": dict(opt="optim.Adam(model.parameters(), lr=0.001)"),
    "Adam 1e-3 MSE": dict(opt="optim.Adam(model.parameters(), lr=0.001)", loss="nn.MSELoss()"),
    "Adam 3e-3 BN cosine": dict(opt="optim.Adam(model.parameters(), lr=0.003)", bn=True, sched="cosine"),
}


def code_for(s):
    sched = ""
    if s.get("sched") == "cosine":
        sched = f"scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={2 * STEPS})\n"
    return CODE.format(
        model=MODEL.format(bn2="nn.LazyBatchNorm2d(), " if s.get("bn") else "",
                           bn1="nn.LazyBatchNorm1d(), " if s.get("bn") else ""),
        opt=s["opt"], sched=sched, loss=s.get("loss", "nn.CrossEntropyLoss()"), batch=128,
        sched_arg=", scheduler=scheduler" if sched else "")


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    env = {**os.environ, "MPNN_WORKSPACE": "ws3"}
    for name, s in SETUPS.items():
        accs, secs = [], []
        for seed in seeds:
            run = OUT / f"{name.replace(' ', '_').replace(':', '')}_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "model.py").write_text(code_for(s))
            t = time.time()
            subprocess.run([sys.executable, str(REPO / "shared/harness.py"),
                            str(run / "model.py"), "--run-dir", str(run), "--seed", str(seed)],
                           env=env, capture_output=True, text=True)
            secs.append(time.time() - t)
            pred_file = run / "practice_pred.npy"
            accs.append(float((VISIBLE_GROUP[np.load(pred_file)] == VISIBLE_GROUP[LABELS]).mean())
                        if pred_file.exists() else float("nan"))
        print(json.dumps({"setup": name, "visible": [round(a, 4) for a in accs],
                          "seconds": [round(x) for x in secs]}), flush=True)


if __name__ == "__main__":
    main()
