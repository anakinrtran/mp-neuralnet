"""Workspace 3's training options in the harness, on small random data (no data/processed needed)."""

import json

import pytest
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset

from shared import harness


def dataset(n=64):
    x = torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    return TensorDataset(x, torch.arange(n) % 36)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(harness._state, "mode", "run")
    monkeypatch.setitem(harness._state, "run_dir", tmp_path)
    return tmp_path


def events(capsys):
    return [json.loads(line[2:]) for line in capsys.readouterr().out.splitlines() if line.startswith("@@")]


def test_optimizer_loss_scheduler_and_update_sizes(run_dir, capsys):
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(36))
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    harness.train(model, dataset(), dataset(), epochs=1, batch_size=16,
                  optimizer=optimizer, loss_fn=nn.MSELoss(), scheduler=scheduler)
    found = events(capsys)
    settings = next(e["text"] for e in found if e["event"] == "settings")
    assert settings == "SGD + momentum, lr 0.1 · cosine · MSE · batch 16"
    steps = [e for e in found if e["event"] == "step"]
    assert steps and all(len(e["updates"]) == 1 and e["updates"][0] > 0 for e in steps)
    assert optimizer.param_groups[0]["lr"] < 0.1  # the scheduler is stepped every batch
    assert (run_dir / "model.pt").exists()


def test_batchnorm_passes_the_single_image_check(run_dir):
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(32), nn.LazyBatchNorm1d(), nn.ReLU(), nn.LazyLinear(36))
    harness.train(model, dataset(), dataset(), epochs=1, batch_size=16)
    assert harness.describe_layers(model) == ["Flatten", "Linear(32)", "BatchNorm", "ReLU", "Linear(36)"]


def test_exploding_loss_stops_with_a_plain_message(run_dir):
    class Exploding(nn.Module):
        def forward(self, out, y):
            return out.sum() * float("nan")

    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(36))
    with pytest.raises(harness.StudentError, match="the loss exploded"):
        harness.train(model, dataset(), dataset(), epochs=1, batch_size=16, loss_fn=Exploding())
    assert not (run_dir / "model.pt").exists()  # so it never becomes the graded run


def test_rejects_a_batch_size_the_blocks_cant_make(run_dir):
    with pytest.raises(harness.StudentError, match="Batch size"):
        harness.train(nn.Sequential(nn.Flatten(), nn.LazyLinear(36)), dataset(), dataset(), batch_size=100)


# ---------- Workspace 4 ----------

def test_early_stopping_stops_and_keeps_the_best_epoch(run_dir, capsys, monkeypatch):
    # validation accuracy at each epoch (measured 4 times an epoch; the last one counts)
    per_epoch = [0.5, 0.7, 0.6, 0.6, 0.9]
    calls = iter(v for v in per_epoch for _ in range(4))
    monkeypatch.setattr(harness, "accuracy", lambda model, data: next(calls))
    harness.train(nn.Sequential(nn.Flatten(), nn.LazyLinear(36)), dataset(), dataset(), epochs=5,
                  batch_size=16, early_stopping=2)
    found = {e["event"]: e for e in events(capsys)}
    assert found["early_stop"]["epoch"] == 4 and found["early_stop"]["best_epoch"] == 2
    assert found["best_weights"]["epoch"] == 2 and found["best_weights"]["accuracy"] == 0.7
    assert "early stopping 2" in found["settings"]["text"]


def test_augment_keeps_shape_and_flip_only_mirrors():
    x = torch.rand(8, 1, 28, 28)
    torch.manual_seed(0)
    flipped = harness._augment(x, ["flip"])
    assert all(torch.equal(f, a) or torch.equal(f, a.flip(-1)) for f, a in zip(flipped, x))
    moved = harness._augment(x, ["rotate", "shift"])
    assert moved.shape == x.shape and not torch.equal(moved, x)
    with pytest.raises(harness.StudentError, match="Augmentations"):
        harness.train(nn.Sequential(nn.Flatten(), nn.LazyLinear(36)), dataset(), dataset(), augment=["zoom"])


def test_small_training_set_limit_is_checked_in_grading_too(monkeypatch):
    from shared import workspaces
    monkeypatch.setattr(harness, "WORKSPACE", workspaces.get("ws4"))
    monkeypatch.setitem(harness._state, "mode", "load")
    with pytest.raises(harness.StudentError, match="10 to 100"):
        harness.load_data(train_per_class=2368)
    assert harness.load_data(train_per_class=100) == (None, None)


def test_confusion_matrix_counts_every_validation_character(run_dir, capsys):
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(36))
    model(torch.zeros(1, 1, 28, 28))
    matrix = harness.confusion_matrix(model, dataset(72))
    assert matrix.shape == (36, 36) and matrix.sum() == 72 and (matrix.sum(1) == 2).all()
    assert any(e["event"] == "confusion" for e in events(capsys))
