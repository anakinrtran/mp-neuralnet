"""The workspaces (1-4 assigned, plus the unused old Workspace 4) and their per-workspace settings.

One codebase and one pair of Docker images serve every workspace. The environment
variable MPNN_WORKSPACE (set per question in info.json) picks which one is running.
Settings that never change between workspaces live in config.py.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from shared import config

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
    # Torch threads for training; 0 = torch's default (all cores). One thread is fastest
    # for small fully connected models; convolutions gain from more.
    threads: int = 1
    # Training characters per class: the default, and the most the Load data block allows
    # (Workspace 4 trains on a small set; grading refuses code that asks for more).
    train_per_class: int = config.TRAIN_PER_CLASS
    # Report training accuracy next to validation accuracy (Workspace 4 on).
    train_accuracy: bool = False
    # Which grid the grader scores: "practice" (the one decoded in the workspace), "heldout"
    # (a test grid the student never sees; feedback shows it) or "hidden" (the class image;
    # feedback shows only the accuracy). heldout/hidden characters are only in the grader image.
    graded_grid: str = "practice"
    graded_image: str = ""  # source image under practice_images/ for a heldout/hidden grid

    @property
    def practice_file(self) -> str:
        """Practice characters (public data dir) and labels (private data dir) share this name."""
        return f"practice_{self.id}.npz"

    @property
    def graded_file(self) -> str:
        """Characters the grader decodes: public data dir for practice, grader data dir otherwise."""
        return self.practice_file if self.graded_grid == "practice" else f"{self.graded_grid}_{self.id}.npz"

    @property
    def labels_file(self) -> str:
        """The graded grid's labels, in tests_dir."""
        return f"{self.graded_grid}_labels.npz"

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
    Workspace(
        id="ws2",
        title="Convolutional network",
        question="ws2-cnn",
        practice_image="workspace2_sea.png",
        # Baselines (sea, 2 epochs, 3 seeds): WS1 model 88.9-90.3%, one Conv stage 92.1-93.5%,
        # two Conv stages 94.3-96.2% (one stage: 92.0% under PrairieLearn's CPU cap). The grader also
        # requires a Conv layer, so 0.90 only has to catch a CNN that didn't train properly.
        threshold=0.90,
        default_epochs=2,
        max_epochs=4,
        threads=0,
    ),
    Workspace(
        id="ws3",
        title="Training methods",
        question="ws3-training",
        practice_image="workspace3_mountains.png",  # 64x64
        # Baselines (mountains, 2 epochs, seeds 0-2; students always get seed 0, listed first):
        # start (SGD lr 10) 2.6/0/0%, SGD 0.1 88.5/89.5/87.9%, Adam 1e-3 + MSE 90.5/90.0/90.1%,
        # Adam 1e-3 91.5/91.9/91.9%, SGD + momentum 0.05 92.5/94.0/93.3%,
        # Adam 3e-3 + BatchNorm + cosine 95.6/95.1/95.5%. Grid (benchmark_ws3.py --grid, seed 0; BatchNorm
        # after the Conv): SGD 0.1 88.5%, + BatchNorm 93.1%; SGD+mom 0.1 92.7%; Adam 0.001 91.5%, + BatchNorm
        # 93.1%, + cosine 89.7%; Adam 0.01 92.5%, + cosine 94.0%; SGD+mom 0.1 + cosine 94.5%. 0.92: following
        # the steps (factors of ten, then SGD + momentum 0.1) passes by the 4th run. The 2-epoch budget is
        # enforced by train().
        threshold=0.92,
        default_epochs=2,
        max_epochs=2,
        threads=0,
    ),
    Workspace(
        id="ws_unused",  # the old Workspace 4, folded into ws4; not assigned
        title="Generalization and evaluation",
        question="ws_unused-generalization",
        practice_image="workspace4_underwater.png",  # 64x64
        # Graded on the held-out grid. Baselines (scripts/benchmark_ws45.py ws_unused: Conv 32 -> Linear 256, Adam
        # 1e-3, 100 per class, 20 epochs, seeds 0-2; students get seed 0, listed first): start 85.9/86.2/84.8%,
        # MLP 512-512 84.4/84.1/85.8%, + weight decay 1e-3 86.2/85.8/85.0%, + early stopping 3
        # 84.6/85.1/83.3%, + flip 79.3/80.8/79.7%, + dropout 0.5 87.8/86.9/86.2%, + rotate+shift
        # 89.2/89.9/90.2%, dropout + rotate+shift 89.9/87.9/87.5% (40 epochs, early stopping 8:
        # 93.5/91.3/91.7%). 0.88: no unregularized setup passes (best 86.2%); augmentation does.
        threshold=0.88,
        default_epochs=20,
        max_epochs=60,
        threads=0,
        train_per_class=100,
        train_accuracy=True,
        graded_grid="heldout",
        # The practice picture again, written with 4,096 other test characters. The planned flower is
        # 72% background (0/O), so its score measured 0/O reading, not overfitting (benchmark_ws45.py).
        graded_image="workspace4_underwater.png",
    ),
    Workspace(
        id="ws4",
        title="Generalization and evaluation",
        question="ws4-generalization",
        practice_image="workspace4_underwater.png",  # ws_unused's picture
        # The old Workspace 4's content plus shipping; graded on the hidden 128x88 grid, which also does the held-out
        # grid's job. Baselines (benchmark_ws45.py ws4, seed 0, 100 per class, 20 epochs; hidden / validation):
        # linear 75.8/76.0%, MLP 84.2/84.6%, start CNN 85.4/85.8%, + dropout 0.3/0.5/0.7 86.8/87.8/86.4%
        # (val 86.9/87.4/86.5%), + weight decay 1e-3 85.5%, + early stopping 5 85.2%, + rotate 87.8/88.9%,
        # + shift 90.9%, + rotate+shift 89.9/90.9%, + rotate+shift+mirror 79.0%, dropout + rotate+shift 40 epochs
        # 92.5/92.2%. Earlier seeds 1-2: start 86.4/85.4%, dropout 0.5 87.2/86.7%. 0.87: the start fails; dropout
        # 0.5 or the Augment block's default (rotate) passes on the 2nd run. Validation tracks hidden within ~1 point.
        threshold=0.87,
        default_epochs=20,
        max_epochs=60,
        threads=0,
        train_per_class=100,  # ws_unused's small set, so regularization still matters
        train_accuracy=True,
        graded_grid="hidden",
        graded_image="ws5_ClassImage_LARGE.png",  # shrunk by make_practice.py; never shown
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
