"""The class vote: reading predictions from a PrairieLearn submissions CSV, and the per-pixel vote."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import class_vote  # noqa: E402
from shared.palette import CLASSES, VISIBLE_GROUP  # noqa: E402


def feedback(rows: list[str]) -> str:
    # PrairieLearn nests the grader's results.json under "results"
    return json.dumps({"succeeded": True, "results": {"score": 1, "class_vote": {"workspace": "ws4", "grid": rows}}})


def test_latest_submission_per_student_and_lookalikes_vote_together(tmp_path):
    path = tmp_path / "subs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["UID", "Submission date", "Feedback"])
        w.writeheader()
        w.writerow({"UID": "a", "Submission date": "2026-12-01T10:00", "Feedback": feedback(["AA", "AA"])})
        w.writerow({"UID": "a", "Submission date": "2026-12-01T11:00", "Feedback": feedback(["0B", "AA"])})
        w.writerow({"UID": "b", "Submission date": "2026-12-01T10:00", "Feedback": feedback(["O8", "AC"])})
        w.writerow({"UID": "c", "Submission date": "2026-12-01T10:00", "Feedback": feedback(["XX", "CC"])})
        w.writerow({"UID": "d", "Submission date": "2026-12-01T10:00", "Feedback": json.dumps({"results": {}})})
        w.writerow({"UID": "e", "Submission date": "2026-12-01T10:00", "Feedback": feedback(["AAA", "AAA"])})
    votes = class_vote.read_votes([path], (2, 2))
    assert sorted(votes) == ["a", "b", "c"]
    assert CLASSES[votes["a"][0, 0]] == "0"  # the later of a's two submissions
    winner, agreement = class_vote.majority(list(votes.values()))
    assert winner[0, 0] == VISIBLE_GROUP[CLASSES.index("O")]  # 0 and O are one vote: 2 of 3
    assert winner[0, 1] == VISIBLE_GROUP[CLASSES.index("B")] and np.isclose(agreement[0, 1], 2 / 3)
    assert CLASSES[winner[1, 1]] == "C"
