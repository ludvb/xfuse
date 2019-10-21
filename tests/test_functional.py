r"""Functional tests"""

import os

import pyro
import pytest

from hssl.session import Session, Unset, get


@pytest.mark.parametrize(
    "arguments",
    [
        ["--patch-size=32", "--batch-size=1", "--epochs=1", "--lazy"],
        ["--patch-size=32", "--batch-size=1", "--epochs=1", "--non-lazy"],
    ],
)
def test_train_exit_status(shared_datadir, script_runner, tmp_path, arguments):
    r"""Test CLI invocation"""
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


def test_restore_session(shared_datadir, script_runner, mocker, tmp_path):
    r"""Test session restore"""
    subcmd = [
        "train",
        str(shared_datadir / "test1" / "design.csv"),
        "--patch-size=32",
        "--batch-size=1",
        "--epochs=1",
    ]

    script_runner.run("hssl", f"--save-path={tmp_path}", *subcmd)

    params = [*pyro.get_param_store().values()]
    pyro.clear_param_store()

    def _mock_run_training(*_args, **_kwargs):
        with Session(panic=Unset):
            assert int(get("global_step")) > 1
            assert all(
                (a == b).all()
                for a, b in zip(params, get("param_store").values())
            )

    mocker.patch("hssl.__main__.run_training", _mock_run_training)

    assert script_runner.run(
        "hssl", f"--session={tmp_path / 'final.session'}", *subcmd
    ).success
