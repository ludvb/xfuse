r"""Functional tests"""

import os

import pytest

from xfuse.session import Session, Unset, get
from xfuse.utility.modules import get_state_dict, reset_state


@pytest.mark.parametrize("test_case", ["test_train_exit_status.1.toml"])
def test_train_exit_status(shared_datadir, script_runner, tmp_path, test_case):
    r"""Test CLI invocation"""
    save_path = tmp_path / "output_dir"
    arguments = [
        "run",
        f"--save-path={save_path}",
        str(shared_datadir / test_case),
    ]
    ret = script_runner.run("xfuse", *arguments)
    assert ret.success
    assert "final.session" in os.listdir(save_path)
    assert "log" in os.listdir(save_path)
    assert "stats" in os.listdir(save_path)


def test_restore_session(shared_datadir, script_runner, mocker, tmp_path):
    r"""Test session restore"""

    script_runner.run(
        "xfuse",
        "run",
        f"--save-path={tmp_path}",
        str(shared_datadir / "test_restore_session.toml"),
    )

    state_dict = get_state_dict()
    reset_state()

    def _mock_run(*_args, **_kwargs):
        with Session(panic=Unset()):
            assert get("training_data").step > 1
            new_state_dict = get_state_dict()
            assert all(
                (
                    new_state_dict.modules[module_name][param_name]
                    == param_value
                ).all()
                for module_name, module_state in state_dict.modules.items()
                for param_name, param_value in module_state.items()
            )
            assert all(
                (new_state_dict.params[param_name] == param_value).all()
                for param_name, param_value in state_dict.params.items()
            )

    mocker.patch("xfuse.__main__._run", _mock_run)

    ret = script_runner.run(
        "xfuse",
        "run",
        f"--save-path={tmp_path}",
        str(shared_datadir / "test_restore_session.toml"),
    )
    assert ret.success
