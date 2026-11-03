"""Rebuild the submitted model, load its weights and print its predictions for the workspace's
graded grid (the practice grid, or a held-out/hidden grid baked only into the grader image).

    python predict.py model.py model.pt

grade.py runs this as an unprivileged user that can't read the labels, because model.py
is student code. The last line of stdout is "@@" + JSON:
{"pred": [[...]], "layers": [...], "workspace": "ws2"} or {"error": "..."}.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import harness  # noqa: E402


def main():
    code, weights = sys.argv[1:3]
    harness.torch.set_num_threads(harness._cpu_limit())
    try:
        model = harness.load_trained_model(code, weights)
        pred = harness.predict_grid(model, harness.DATA_DIR / harness.WORKSPACE.graded_file)
        result = {"pred": pred.tolist(), "layers": harness.describe_layers(model),
                  "workspace": harness.WORKSPACE.id}
    except harness.StudentError as e:
        result = {"error": str(e)}
    except RuntimeError as e:
        if "state_dict" not in str(e):
            result = {"error": f"The submitted model could not be loaded (RuntimeError: {e})."}
        else:
            result = {"error": "The saved weights don't fit the submitted model code. In the workspace, "
                               "run your model again, then Save & Grade."}
    except Exception as e:
        result = {"error": f"The submitted model could not be loaded ({type(e).__name__}: {e})."}
    print("@@" + json.dumps(result))


if __name__ == "__main__":
    main()
