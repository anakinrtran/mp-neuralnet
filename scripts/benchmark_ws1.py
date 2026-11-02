"""Workspace 1 baselines: training time and accuracy of the linear and ReLU models.

  python scripts/benchmark_ws1.py timing                      seconds per epoch, per model and thread count
  python scripts/benchmark_ws1.py accuracy --epochs 5 --seeds 5   val and practice accuracy over several seeds

Practice accuracy is reported two ways:
  raw      the predicted class must be exactly right
  visible  look-alike confusions (0/O, 1/I, ...) count as right, matching what the decoded image shows
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from shared.palette import VISIBLE_GROUP  # noqa: E402

FIXTURES = REPO / "fixtures"
LABELS = REPO / "data" / "processed" / "private" / "practice_ws1.npz"


def model_code(hidden: int | None, epochs: int) -> str:
    """Code like the blocks generate: linear if hidden is None, else one ReLU hidden layer."""
    if hidden is None:
        code = (FIXTURES / "ws1_linear.py").read_text()
    else:
        code = (FIXTURES / "ws1_relu.py").read_text().replace("LazyLinear(128)", f"LazyLinear({hidden})")
    return re.sub(r"epochs=\d+", f"epochs={epochs}", code)


def run(code: str, workdir: Path, seed: int, threads: int) -> dict:
    """Run code through the harness in a subprocess; return its events and wall time."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.py").write_text(code)
    cmd = [sys.executable, str(REPO / "shared" / "harness.py"), str(workdir / "model.py"),
           "--run-dir", str(workdir), "--seed", str(seed), "--threads", str(threads)]
    start = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    wall = time.perf_counter() - start
    events = [json.loads(line[2:]) for line in out.splitlines() if line.startswith("@@")]
    by_type = {e["event"]: e for e in events}
    return {"wall": wall, "events": events, **by_type}


def practice_accuracy(pred: np.ndarray) -> tuple[float, float]:
    labels = np.load(LABELS)["y"]
    raw = (pred == labels).mean()
    visible = (VISIBLE_GROUP[pred] == VISIBLE_GROUP[labels]).mean()
    return float(raw), float(visible)


def timing(args, tmp: Path):
    print(f"{'model':<10}{'threads':>8}{'s/epoch':>9}{'overhead s':>12}")
    for name, hidden in [("linear", None), ("relu128", 128), ("relu512", 512)]:
        for threads in args.threads:
            r = run(model_code(hidden, 2), tmp / f"t_{name}_{threads}", seed=0, threads=threads)
            per_epoch = r["train_done"]["seconds"] / 2
            print(f"{name:<10}{threads or 'all':>8}{per_epoch:>9.2f}{r['wall'] - 2 * per_epoch:>12.2f}")


def accuracy(args, tmp: Path):
    print(f"epochs={args.epochs}, threads={args.threads[0] or 'all'}")
    print(f"{'model':<10}{'seed':>5}{'wall s':>8}{'val':>8}{'practice raw':>14}{'visible':>9}")
    summary = {}
    for name, hidden in [("linear", None), ("relu128", 128)]:
        rows = []
        for seed in range(args.seeds):
            workdir = tmp / f"a_{name}_{seed}"
            r = run(model_code(hidden, args.epochs), workdir, seed=seed, threads=args.threads[0])
            pred = np.load(workdir / "practice_pred.npy")
            raw, visible = practice_accuracy(pred)
            rows.append((r["wall"], r["evaluate"]["accuracy"], raw, visible))
            print(f"{name:<10}{seed:>5}{r['wall']:>8.1f}{r['evaluate']['accuracy']:>8.3f}{raw:>14.3f}{visible:>9.3f}")
        summary[name] = np.array(rows)
        if name == "relu128":
            check_reload(workdir, pred)

    print("\nmean (min-max)")
    for name, rows in summary.items():
        cells = [f"{rows[:, i].mean():.3f} ({rows[:, i].min():.3f}-{rows[:, i].max():.3f})" for i in range(4)]
        print(f"{name:<10} wall {cells[0]}  val {cells[1]}  raw {cells[2]}  visible {cells[3]}")


def check_reload(workdir: Path, pred: np.ndarray):
    """The grader path (load code + weights, no training) must reproduce the same predictions."""
    from shared import harness
    model = harness.load_trained_model(workdir / "model.py", workdir / "model.pt")
    same = np.array_equal(harness.predict_practice(model), pred)
    print(f"  reload check (grader path, no retraining): {'OK' if same else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["timing", "accuracy"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 0], help="0 = torch default")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        {"timing": timing, "accuracy": accuracy}[args.mode](args, Path(tmp))


if __name__ == "__main__":
    main()
