r"""Functional tests"""

import os

import pytest

from hssl.session import Session, Unset, get
from hssl.utility.modules import get_state_dict, load_state_dict


@pytest.mark.parametrize(
    "test_case", ["test_train_exit_status.1.toml"],
)
def test_train_exit_status(shared_datadir, script_runner, tmp_path, test_case):
    r"""Test CLI invocation"""
    save_path = tmp_path / "output_dir"
    arguments = [
        "run",
        f"--save-path={save_path}",
        str(shared_datadir / test_case),
    ]
    ret = script_runner.run("hssl", *arguments)
    assert ret.success
    assert "final.session" in os.listdir(save_path)
    assert "log" in os.listdir(save_path)
    assert "stats" in os.listdir(save_path)


def test_restore_session(shared_datadir, script_runner, mocker, tmp_path):
    r"""Test session restore"""

    script_runner.run(
        "hssl",
        "run",
        f"--save-path={tmp_path}",
        str(shared_datadir / "test_restore_session.toml"),
    )

    module_state = get_state_dict()
    load_state_dict({})

    def _mock_run(*_args, **_kwargs):
        with Session(panic=Unset):
            assert get("training_data").step > 1
            new_module_state = get_state_dict()
            assert all(
                (new_module_state[k] == v).all()
                for k, v in module_state.items()
            )

    mocker.patch("hssl.__main__._run", _mock_run)

    ret = script_runner.run(
        "hssl",
        "run",
        f"--save-path={tmp_path}",
        str(shared_datadir / "test_restore_session.toml"),
    )
    assert ret.success
