"""PrairieLearn external grader for every workspace (MPNN_WORKSPACE picks which).

Reads the submitted model.py and model.pt from /grade/student, decodes the workspace's graded
grid without retraining, and writes /grade/results/results.json. The score is 1 if the visible
accuracy meets the workspace's threshold (and the model has any layers the workspace requires,
e.g. a Conv layer in Workspace 2), else 0.

The graded grid (shared/workspaces.py, graded_grid) is the practice grid in Workspaces 1-3,
the hidden class image in Workspace 4, and a held-out grid the student never sees in the
unused workspace (ws_unused). Feedback shows the decoded image next to the original, except for the hidden
image: there it shows only the accuracy, and results.json also stores the predicted grid
under "class_vote" (not shown; PrairieLearn keeps it in the submission's feedback, which the
assessment's submissions CSV downloads include) for scripts/class_vote.py.

Student code runs in a separate process as an unprivileged user (when started as root,
as in PrairieLearn), after the labels are read and locked away.

    GRADE_DIR=/grade python grade.py
"""

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from shared.grids import classes_to_image, upscale  # noqa: E402
from shared.harness import DATA_DIR, WORKSPACE  # noqa: E402
from shared.palette import CLASSES, VISIBLE_GROUP, load_palette  # noqa: E402

GRADE = Path(os.environ.get("GRADE_DIR", "/grade"))
STUDENT = GRADE / "student"
TESTS = GRADE / "tests"
RESULTS = GRADE / "results" / "results.json"
SANDBOX_USER = "ag"  # created in the grader Dockerfile
PREDICT_TIMEOUT = 45
SCALE = 8
AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def ws1_hint(layers: list[str]) -> str:
    if sum(name.startswith("Linear") for name in layers) == 1:
        return ("A linear model can't reach this. Add a hidden layer: "
                "Flatten → Linear 128 → ReLU → Linear 36, run it, then Save & Grade again.")
    return ("Check that a ReLU comes right after your hidden Linear layer, "
            "or try more epochs. Then run it and Save & Grade again.")


def ws2_hint(layers: list[str]) -> str:
    if sum(name.startswith("Conv") for name in layers) < 2:
        return "Try a second convolution stage, or more epochs. Then run it and Save & Grade again."
    return "Try more filters or more epochs. Then run it and Save & Grade again."


def ws3_hint(layers: list[str]) -> str:
    return ("Look at your loss curve in the workspace: is it flat, bumpy, or still dropping at the end? "
            "Try another optimizer or learning rate, BatchNorm or a scheduler, run it, then Save & Grade again.")


def ws_unused_hint(layers: list[str]) -> str:
    return ("Look at your training and validation accuracy in the workspace: how far apart do they end up? "
            "Which ways of keeping a model from memorizing have you combined? Run it, then Save & Grade again.")


def ws4_hint(layers: list[str]) -> str:
    if not any(name.startswith("Conv") for name in layers):
        return "Which kind of layer did best on images in Workspaces 2 and 3? Run it, then Save & Grade again."
    return ("Look at your training and validation accuracy in the workspace: how far apart do they end up? "
            "Which fixes raised the top of your validation curve, and have you combined them? "
            "Run it, then Save & Grade again.")


# What to suggest when a submission falls short, per workspace, given the model's layer names.
FAIL_HINTS = {"ws1": ws1_hint, "ws2": ws2_hint, "ws3": ws3_hint, "ws4": ws4_hint, "ws_unused": ws_unused_hint}

# How feedback names the graded grid.
GRID_NAMES = {"practice": "the practice image", "heldout": "the held-out test image",
              "hidden": "the hidden class image"}

# Layers a submission must contain, per workspace: (layer name prefix, test name, message if missing).
REQUIRED_LAYERS = {
    "ws2": [("Conv", "The model has a Conv layer",
             "Your model has no Conv layer, and this workspace is about convolutional networks. "
             "Add a convolution stage, run it, then Save & Grade again.")],
}


def write_results(results: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results))


def not_gradable(message: str):
    write_results({"gradable": False, "score": 0.0, "format_errors": [message], "message": message})


def png_data_url(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_labels() -> tuple[np.ndarray, np.ndarray]:
    """Read the graded grid's labels and palette, check they belong to the baked-in characters,
    then lock them away."""
    with np.load(TESTS / WORKSPACE.labels_file) as f:
        labels, want = f["y"], str(f["x_sha256"])
        # a held-out or hidden grid has its own palette; the practice grid's is in shared/palettes
        palette = f["palette"] if "palette" in f else load_palette(WORKSPACE.id)
    chars = np.load(DATA_DIR / WORKSPACE.graded_file)["x"]
    if hashlib.sha256(np.ascontiguousarray(chars, dtype=np.uint8).tobytes()).hexdigest() != want:
        raise RuntimeError("Grader setup error: the labels don't match the graded characters "
                           "in the grader image. Rebuild both from data/make_practice.py.")
    if AS_ROOT:
        os.chmod(TESTS, 0o700)
    return labels, palette


def predict(code: Path, weights: Path) -> dict:
    """Run predict.py on copies of the student's files, as the sandbox user when we are root."""
    work = Path(tempfile.mkdtemp(prefix="submission-"))
    for f in (code, weights):
        shutil.copyfile(f, work / f.name)
    if AS_ROOT:
        shutil.chown(work, SANDBOX_USER)
        for f in work.iterdir():
            shutil.chown(f, SANDBOX_USER)
    env = {"PATH": os.environ.get("PATH", ""), "MPNN_DATA": str(DATA_DIR), "HOME": str(work),
           "MPNN_WORKSPACE": WORKSPACE.id}
    if "SYSTEMROOT" in os.environ:  # Windows (local testing) can't start Python without it
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    try:
        proc = subprocess.run(
            [sys.executable, str(HERE / "predict.py"), str(work / code.name), str(work / weights.name)],
            capture_output=True, text=True, timeout=PREDICT_TIMEOUT, cwd=work, env=env,
            user=SANDBOX_USER if AS_ROOT else None, group=SANDBOX_USER if AS_ROOT else None,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Loading the model took longer than {PREDICT_TIMEOUT}s."}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("@@")]
    if not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return {"error": "The submitted model could not be loaded.\n" + "\n".join(tail)}
    return json.loads(lines[-1][2:])


def main():
    code, weights = STUDENT / "model.py", STUDENT / "model.pt"
    if not (code.exists() and weights.exists()):
        not_gradable("No trained model was found. In the workspace, run your model until it finishes, "
                     "then come back here and click Save & Grade.")
        return
    info_path = STUDENT / "submission.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    labels, palette = load_labels()
    result = predict(code, weights)
    if "error" in result:
        not_gradable(result["error"])
        return
    if result.get("workspace") != WORKSPACE.id:  # it must decode the same grid
        raise RuntimeError(f"predict.py decoded {result.get('workspace')}'s grid, not {WORKSPACE.id}'s.")
    pred = np.array(result["pred"])
    if pred.shape != labels.shape:
        not_gradable("The submitted model gave the wrong number of predictions.")
        return

    grid = GRID_NAMES[WORKSPACE.graded_grid]
    wrong = VISIBLE_GROUP[pred] != VISIBLE_GROUP[labels]
    visible = 1 - wrong.mean()
    raw = (pred == labels).mean()
    passed = visible >= WORKSPACE.threshold
    needed = WORKSPACE.threshold
    summary = (f"{visible:.1%} of {grid}'s pixels have the right color "
               f"(you need {needed:.1%}). {int(wrong.sum())} of {wrong.size} pixels are wrong.")
    if WORKSPACE.graded_grid == "heldout":
        summary += " Your model never saw these characters in the workspace."
    layers = result["layers"]
    tests = [{"name": f"{grid[4:].capitalize()}: at least {needed:.1%} of pixels right",
              "points": int(passed), "max_points": 1, "message": summary}]
    missing = []
    for prefix, name, why in REQUIRED_LAYERS.get(WORKSPACE.id, []):
        has = any(layer.startswith(prefix) for layer in layers)
        tests.append({"name": name, "points": int(has), "max_points": 1, "message": "" if has else why})
        if not has:
            missing.append(why)
    if missing:
        message = "Not yet. " + " ".join(missing) + " (" + summary + ")"
    elif passed:
        message = "Passed. " + summary
    else:
        message = "Not yet. " + summary + " " + FAIL_HINTS[WORKSPACE.id](layers)
    passed = passed and not missing

    model_line = " → ".join(layers)
    val = info.get("val_accuracy")
    name = grid[4:].capitalize()
    output = (f"Model: {model_line}\n"
              + (f"Validation accuracy in the workspace: {val:.1%}\n" if val is not None else "")
              + f"{name}, visible accuracy: {visible:.1%} (look-alikes such as 0/O count as right)\n"
              + f"{name}, exact accuracy: {raw:.1%}\n\n"
              + "Submitted code:\n" + code.read_text(encoding="utf-8", errors="replace"))

    results = {"gradable": True, "score": 1.0 if passed else 0.0, "message": message, "output": output,
               "tests": tests}
    if WORKSPACE.graded_grid == "hidden":
        # never show the hidden image; keep the predictions (one string of classes per row) for the class vote
        results["class_vote"] = {"workspace": WORKSPACE.id,
                                 "grid": ["".join(CLASSES[c] for c in row) for row in pred]}
    else:
        mask = np.where(wrong[..., None], [220, 40, 40], [245, 245, 245]).astype(np.uint8)
        results["images"] = [
            {"label": "Your decoded image", "url": png_data_url(upscale(classes_to_image(pred, palette), SCALE))},
            {"label": "The original image", "url": png_data_url(upscale(classes_to_image(labels, palette), SCALE))},
            {"label": "Wrong pixels (red)", "url": png_data_url(upscale(mask, SCALE))},
        ]
    write_results(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never leave PrairieLearn without results
        not_gradable(f"The grader hit an error. Please tell course staff. ({type(e).__name__}: {e})")
        raise
