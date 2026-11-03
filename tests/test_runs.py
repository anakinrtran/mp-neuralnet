"""How the server explains a run that died without an error of its own."""

from server.runs import crash_message


def test_out_of_memory_kill_is_explained(tmp_path):
    (tmp_path / "model.py").write_text("train(model, train_data, val_data, epochs=2, batch_size=1024)")
    assert "ran out of memory" in crash_message(-9, tmp_path)
    assert "smaller batch size" in crash_message(-9, tmp_path)
    (tmp_path / "model.py").write_text("train(model, train_data, val_data, epochs=2)")  # no batch size yet (WS1-2)
    assert "batch" not in crash_message(137, tmp_path)


def test_other_crashes_keep_the_exit_code(tmp_path):
    (tmp_path / "model.py").write_text("")
    assert crash_message(1, tmp_path) == "The run crashed (exit code 1)."
