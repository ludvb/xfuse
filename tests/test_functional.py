"""Functional tests"""

import os

import pytest


@pytest.mark.parametrize(
    "arguments", [["--patch-size=32", "--batch-size=1", "--epochs=1"]]
)
def test_train_exit_status(shared_datadir, script_runner, tmp_path, arguments):
    """Test CLI invocation"""
    save_path = tmp_path / "output_dir"
    arguments = [
        f"--save-path={save_path}",
        "train",
        str(shared_datadir / "test1" / "design.csv"),
        *arguments,
    ]
    ret = script_runner.run("hssl", *arguments)
    assert ret.success
    assert "final.session" in os.listdir(save_path)
    assert "log" in os.listdir(save_path)
    assert "stats" in os.listdir(save_path)
