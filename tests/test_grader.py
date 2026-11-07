"""Grader decisions, with the sandboxed prediction step stubbed out."""

import json

import numpy as np
import pytest

import grade
from shared import workspaces

LABELS = np.zeros((32, 32), dtype=int)


@pytest.fixture
def submission(tmp_path, monkeypatch):
    student = tmp_path / "student"
    student.mkdir()
    (student / "model.py").write_text("")
    (student / "model.pt").write_bytes(b"")
    monkeypatch.setattr(grade, "STUDENT", student)
    monkeypatch.setattr(grade, "RESULTS", tmp_path / "results" / "results.json")
    monkeypatch.setattr(grade, "load_labels", lambda: (LABELS, grade.load_palette("ws1")))
    return tmp_path


def _predicted(monkeypatch, workspace, pred=LABELS):
    monkeypatch.setattr(grade, "predict", lambda code, weights: {
        "pred": pred.tolist(), "layers": ["Flatten", "Linear(36)"], "workspace": workspace})


def test_refuses_predictions_for_another_workspace(submission, monkeypatch):
    _predicted(monkeypatch, "ws9")
    with pytest.raises(RuntimeError, match="grid"):
        grade.main()


def test_scores_a_correct_prediction(submission, monkeypatch):
    _predicted(monkeypatch, grade.WORKSPACE.id)
    grade.main()
    results = json.loads(grade.RESULTS.read_text())
    assert results["score"] == 1.0 and results["gradable"]


def test_missing_files_are_not_gradable(submission, monkeypatch):
    (grade.STUDENT / "model.pt").unlink()
    grade.main()
    results = json.loads(grade.RESULTS.read_text())
    assert results["gradable"] is False and results["score"] == 0.0


def test_hidden_grid_shows_only_accuracy_and_keeps_the_vote(submission, monkeypatch):
    monkeypatch.setattr(grade, "WORKSPACE", workspaces.get("ws5"))
    _predicted(monkeypatch, "ws5")
    grade.main()
    results = json.loads(grade.RESULTS.read_text())
    assert "images" not in results and results["score"] == 1.0
    assert results["class_vote"]["grid"][0] == "0" * 32
