"""PrairieLearn external grader for Workspace 1.

Reads the submitted model.py and model.pt from /grade/student, decodes the practice grid
without retraining, and writes /grade/results/results.json. The score is 1 if the visible
practice accuracy meets PRACTICE_THRESHOLD, else 0. Feedback shows the student's decoded
image next to the original.

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

from shared import config  # noqa: E402
from shared.grids import classes_to_image, upscale  # noqa: E402
from shared.harness import DATA_DIR  # noqa: E402
from shared.palette import VISIBLE_GROUP  # noqa: E402

GRADE = Path(os.environ.get("GRADE_DIR", "/grade"))
STUDENT = GRADE / "student"
TESTS = GRADE / "tests"
RESULTS = GRADE / "results" / "results.json"
SANDBOX_USER = "ag"  # created in the grader Dockerfile
PREDICT_TIMEOUT = 45
SCALE = 8
AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def write_results(results: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results))


def not_gradable(message: str):
    write_results({"gradable": False, "score": 0.0, "format_errors": [message], "message": message})


def png_data_url(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_labels() -> np.ndarray:
    """Read the labels, check they belong to the baked-in characters, then lock them away."""
    with np.load(TESTS / "practice_labels.npz") as f:
        labels, want = f["y"], str(f["x_sha256"])
    chars = np.load(DATA_DIR / "practice_ws1.npz")["x"]
    if hashlib.sha256(np.ascontiguousarray(chars, dtype=np.uint8).tobytes()).hexdigest() != want:
        raise RuntimeError("Grader setup error: practice labels don't match the practice characters "
                           "in the grader image. Rebuild both from data/make_practice.py.")
    if AS_ROOT:
        os.chmod(TESTS, 0o700)
    return labels


def predict(code: Path, weights: Path) -> dict:
    """Run predict.py on copies of the student's files, as the sandbox user when we are root."""
    work = Path(tempfile.mkdtemp(prefix="submission-"))
    for f in (code, weights):
        shutil.copyfile(f, work / f.name)
    if AS_ROOT:
        shutil.chown(work, SANDBOX_USER)
        for f in work.iterdir():
            shutil.chown(f, SANDBOX_USER)
    env = {"PATH": os.environ.get("PATH", ""), "MPNN_DATA": str(DATA_DIR), "HOME": str(work)}
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
        not_gradable("No trained model was submitted. In the workspace, run your model, click Submit, "
                     "then come back here and click Save & Grade.")
        return
    info_path = STUDENT / "submission.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    labels = load_labels()
    result = predict(code, weights)
    if "error" in result:
        not_gradable(result["error"])
        return
    pred = np.array(result["pred"])
    if pred.shape != labels.shape:
        not_gradable("The submitted model gave the wrong number of predictions.")
        return

    wrong = VISIBLE_GROUP[pred] != VISIBLE_GROUP[labels]
    visible = 1 - wrong.mean()
    raw = (pred == labels).mean()
    passed = visible >= config.PRACTICE_THRESHOLD
    needed = config.PRACTICE_THRESHOLD
    summary = (f"{visible:.1%} of the practice image's pixels have the right color "
               f"(you need {needed:.0%}). {int(wrong.sum())} of {wrong.size} pixels are wrong.")
    layers = result["layers"]
    if passed:
        message = "Passed. " + summary
    elif sum(name.startswith("Linear") for name in layers) == 1:
        message = ("Not yet. " + summary + " A linear model can't reach this. Add a hidden layer: "
                   "Flatten → Linear 128 → ReLU → Linear 36, run it, Submit, and grade again.")
    else:
        message = ("Not yet. " + summary + " Check that a ReLU comes right after your hidden Linear layer, "
                   "or try more epochs. Then run, Submit, and grade again.")

    model_line = " → ".join(layers)
    val = info.get("val_accuracy")
    output = (f"Model: {model_line}\n"
              + (f"Validation accuracy in the workspace: {val:.1%}\n" if val is not None else "")
              + f"Practice image, visible accuracy: {visible:.1%} (look-alikes such as 0/O count as right)\n"
              + f"Practice image, exact accuracy: {raw:.1%}\n\n"
              + "Submitted code:\n" + code.read_text(encoding="utf-8", errors="replace"))

    mask = np.where(wrong[..., None], [220, 40, 40], [245, 245, 245]).astype(np.uint8)
    write_results({
        "gradable": True,
        "score": 1.0 if passed else 0.0,
        "message": message,
        "output": output,
        "images": [
            {"label": "Your decoded image", "url": png_data_url(upscale(classes_to_image(pred), SCALE))},
            {"label": "The original image", "url": png_data_url(upscale(classes_to_image(labels), SCALE))},
            {"label": "Wrong pixels (red)", "url": png_data_url(upscale(mask, SCALE))},
        ],
        "tests": [{
            "name": f"Practice image: at least {needed:.0%} of pixels right",
            "points": 1 if passed else 0,
            "max_points": 1,
            "message": summary,
        }],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never leave PrairieLearn without results
        not_gradable(f"The grader hit an error. Please tell course staff. ({type(e).__name__}: {e})")
        raise
