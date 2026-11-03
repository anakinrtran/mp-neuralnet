"""The One step tab's computation, on an untrained model (it misreads most characters)."""

import pytest
import torch
from torch import nn

from server.onestep import one_step
from shared import harness
from shared.palette import CLASSES, VISIBLE_GROUP

pytestmark = pytest.mark.skipif(not (harness.DATA_DIR / "val.npz").exists(),
                                reason="needs data/processed (run data/prepare_emnist.py)")

CODE = """from torch import nn
from harness import load_data, train
train_data, val_data = load_data()
model = nn.Sequential(nn.Flatten(), nn.LazyLinear(36))
train(model, train_data, val_data, epochs=1)
"""


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "model.py").write_text(CODE)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(36))
    model(torch.zeros(1, 1, 28, 28))
    torch.save(model.state_dict(), tmp_path / "model.pt")
    return tmp_path


def test_one_step_takes_apart_a_mistake(run_dir):
    weights = (run_dir / "model.pt").read_bytes()
    r = one_step(run_dir, 0)
    guess = r["before"][0][0]
    assert VISIBLE_GROUP[CLASSES.index(guess)] != VISIBLE_GROUP[CLASSES.index(r["label"])]
    assert [layer["name"] for layer in r["layers"]] == ["Linear(36)"]
    assert len(r["image"]) == len(r["saliency"]) == 28
    assert (run_dir / "model.pt").read_bytes() == weights  # the run's model is unchanged


def test_another_mistake_moves_on(run_dir):
    first = one_step(run_dir, 0)
    assert one_step(run_dir, first["index"] + 1)["index"] > first["index"]
