import re

from shared.workspaces import WORKSPACES


def test_question_pass_marks_match_thresholds():
    """question.html is static, so its pass mark is typed in; it must match `threshold`."""
    for ws in WORKSPACES.values():
        html = (ws.tests_dir.parent / "question.html").read_text(encoding="utf-8")
        assert re.findall(r"<b>([\d.]+)%</b>", html) == [f"{ws.threshold * 100:g}"], ws.id
