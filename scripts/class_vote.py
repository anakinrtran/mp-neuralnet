"""The class vote on Workspace 4's hidden image (staff only: reads private labels and palette).

Each graded Workspace 4 submission stores the student's predicted grid under "class_vote" in its
feedback (grader/grade.py). PrairieLearn keeps the whole feedback with the submission, and the
assessment's Downloads page exports it in the "Feedback" column of
<assessment>_final_submissions.csv / _best_submissions.csv / _all_submissions.csv.

This script takes each student's latest graded submission (one vote per student), votes per pixel
on color groups (look-alike classes such as 0 and O count as the same vote, as in grading), and
renders the class image with the hidden image's palette. Ties go to the color group listed first
in the class order 0-9, A-Z (any rule would do; ties are rare with many votes).

    python scripts/class_vote.py SUBMISSIONS.csv [more.csv] [--out class_image.png] [--scale 8]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from shared import workspaces  # noqa: E402
from shared.grids import upscale  # noqa: E402
from shared.palette import CLASSES, VISIBLE_GROUP  # noqa: E402

WS = workspaces.get("ws4")
PRIVATE = REPO / "data" / "processed" / "private" / WS.graded_file


def find_vote(obj):
    """The "class_vote" entry anywhere in a feedback object (PrairieLearn nests results.json)."""
    if isinstance(obj, dict):
        if "class_vote" in obj:
            return obj["class_vote"]
        return next((v for v in map(find_vote, obj.values()) if v is not None), None)
    return None


def read_votes(paths: list[Path], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Each student's latest predicted grid, as class indices, keyed by UID."""
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    latest: dict[str, tuple[str, np.ndarray]] = {}
    lookup = {c: i for i, c in enumerate(CLASSES)}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    vote = find_vote(json.loads(row.get("Feedback") or "null"))
                except json.JSONDecodeError:
                    continue
                if not vote or vote.get("workspace") != WS.id:
                    continue
                grid = np.array([[lookup[c] for c in line] for line in vote["grid"]])
                if grid.shape != shape:
                    print(f"skipped a submission with a {grid.shape} grid (the hidden grid is {shape})")
                    continue
                who = row.get("UID") or row.get("Usernames") or row.get("Username") or "?"
                when = row.get("Submission date") or ""
                if who not in latest or when >= latest[who][0]:
                    latest[who] = (when, grid)
    return {who: grid for who, (_, grid) in latest.items()}


def majority(grids: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: the winning color group, and the share of votes it got."""
    groups = VISIBLE_GROUP[np.stack(grids)]  # (students, H, W)
    counts = np.stack([(groups == g).sum(0) for g in range(len(CLASSES))])  # (36, H, W)
    winner = counts.argmax(0)  # ties: the lowest class index
    return winner, counts.max(0) / len(grids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("class_image.png"))
    parser.add_argument("--scale", type=int, default=8)
    args = parser.parse_args()

    private = np.load(PRIVATE)
    labels, palette = private["y"], private["palette"]
    votes = read_votes(args.csv, labels.shape)
    if not votes:
        raise SystemExit("No Workspace 4 submissions with predictions found in those files.")
    winner, agreement = majority(list(votes.values()))
    right = VISIBLE_GROUP[labels]
    students = sorted(float((VISIBLE_GROUP[g] == right).mean()) for g in votes.values())
    print(f"votes: {len(votes)} students")
    print(f"class image accuracy: {(winner == right).mean():.1%} of pixels right")
    print(f"single students: median {np.median(students):.1%}, best {students[-1]:.1%}, worst {students[0]:.1%}")
    print(f"pixels where most students agree (>50%): {(agreement > 0.5).mean():.1%}")
    Image.fromarray(upscale(palette[winner], args.scale)).save(args.out)
    print("class image:", args.out)


if __name__ == "__main__":
    main()
