"""The five workspaces and their per-workspace settings.

One codebase and one pair of Docker images serve every workspace. The environment
variable MPNN_WORKSPACE (set per question in info.json) picks which one is running.
Settings that never change between workspaces live in config.py.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Workspace:
    id: str  # "ws1"; also names the practice files and the editor module
    title: str
    question: str  # directory under questions/
    practice_image: str  # file under practice_images/ (git-ignored answer key)
    # Minimum "visible" practice accuracy to pass: look-alike confusions (0/O, 1/I, ...)
    # count as correct, since they barely show in the decoded image.
    threshold: float
    default_epochs: int
    max_epochs: int

    @property
    def practice_file(self) -> str:
        """Practice characters (public data dir) and labels (private data dir) share this name."""
        return f"practice_{self.id}.npz"

    @property
    def tests_dir(self) -> Path:
        return REPO / "questions" / self.question / "tests"


WORKSPACES = {ws.id: ws for ws in [
    Workspace(
        id="ws1",
        title="Basic network",
        question="ws1-basic-network",
        practice_image="workspace1_dog.png",
        # Baselines (scripts/benchmark_ws1.py, 5 seeds, fitted palette): linear 85.6-87.3%, ReLU-128 89.9-93.8%;
        # 0.885 is about halfway between the best linear and the worst ReLU seed.
        threshold=0.885,
        default_epochs=5,
        max_epochs=10,
    ),
]}


def get(ws_id: str) -> Workspace:
    try:
        return WORKSPACES[ws_id]
    except KeyError:
        raise ValueError(f"Unknown workspace {ws_id!r}. Known: {', '.join(WORKSPACES)}.") from None


def current() -> Workspace:
    """The workspace named by MPNN_WORKSPACE (default ws1)."""
    return get(os.environ.get("MPNN_WORKSPACE", "ws1"))
