"""Harness logic that is easy to break silently: CPU limits and the filter timelapse."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from shared import harness
from shared.palette import CLASSES


@pytest.mark.parametrize("files, expected", [
    ({"cpu.max": "90000 100000\n"}, 1),                 # cgroup v2, PrairieLearn's local cap
    ({"cpu.max": "250000 100000\n"}, 2),
    ({"cpu.max": "max 100000\n"}, None),                # v2, no limit
    ({"cpu/cpu.cfs_quota_us": "90000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 1),  # v1 (Docker Desktop)
    ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, None),  # v1, no limit
    ({}, None),                                          # no cgroup files (Windows)
])
def test_cpu_limit(tmp_path, files, expected):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(text)
    assert harness._cpu_limit(tmp_path) == (expected or harness.os.cpu_count())


def _val_data():
    labels = torch.arange(len(CLASSES))  # one image per class, so the sample "A" exists
    return TensorDataset(torch.rand(len(CLASSES), 1, 28, 28), labels)


def _record(model, path, total_steps=100):
    model(torch.zeros(1, 1, 28, 28))  # materialize lazy layers, as check_model does
    snapshot = harness._timelapse_recorder(model, _val_data(), total_steps)
    for step in range(1, total_steps + 1):
        snapshot(step, step / 10)
    snapshot.save(path)


def test_timelapse_records_first_conv(tmp_path):
    model = nn.Sequential(nn.LazyConv2d(4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.LazyLinear(36))
    _record(model, tmp_path / "t.npz")
    data = np.load(tmp_path / "t.npz")
    assert data["progress"].shape == (harness.TIMELAPSE_SNAPSHOTS,)
    assert data["progress"][0] == 0 and data["progress"][-1] == 10  # before training, and the end
    assert data["filters"].shape == (harness.TIMELAPSE_SNAPSHOTS, 4, 3, 3)
    assert data["outputs"].shape == (harness.TIMELAPSE_SNAPSHOTS, 4, 28, 28)


def test_timelapse_skips_models_without_conv(tmp_path):
    _record(nn.Sequential(nn.Flatten(), nn.LazyLinear(36)), tmp_path / "t.npz")
    assert not (tmp_path / "t.npz").exists()
