"""Rebuild the submitted model, load its weights and print its practice-grid predictions.

    python predict.py model.py model.pt

grade.py runs this as an unprivileged user that can't read the labels, because model.py
is student code. The last line of stdout is "@@" + JSON: {"pred": [[...]], "layers": [...]}
or {"error": "..."}.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import harness  # noqa: E402


def main():
    code, weights = sys.argv[1:3]
    try:
        model = harness.load_trained_model(code, weights)
        result = {"pred": harness.predict_practice(model).tolist(), "layers": harness.describe_layers(model)}
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
