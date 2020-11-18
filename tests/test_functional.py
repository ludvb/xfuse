r"""Functional tests"""

import os
from glob import glob

import pytest

from xfuse.__main__ import construct_default_config_toml
from xfuse.session import Session, Unset, get
from xfuse.session.items.training_data import TrainingData
from xfuse.utility.state import get_state_dict, reset_state


@pytest.mark.script_launch_mode("subprocess")
@pytest.mark.parametrize("test_case", ["test_train_exit_status.1.toml"])
def test_train_exit_status(shared_datadir, script_runner, tmp_path, test_case):
    r"""Test CLI run invocation"""
    save_path = tmp_path / "output_dir"
    arguments = [
        "run",
        str(shared_datadir / test_case),
        f"--save-path={save_path}",
        "--checkpoint-interval=1",
        "--purge-interval=1",
    ]
    ret = script_runner.run("xfuse", *arguments)
    assert ret.success
    assert "final.session" in os.listdir(save_path)
    assert "log" in os.listdir(save_path)
    checkpoint_pattern = os.path.join(save_path, "checkpoints", "*.session")
    assert len(glob(checkpoint_pattern)) > 0


@pytest.mark.script_launch_mode("subprocess")
def test_train_stats_filewriter(shared_datadir, script_runner, tmp_path):
    r"""Test CLI invocation"""
    save_path = tmp_path / "output_dir"
    arguments = [
        "run",
        str(shared_datadir / "test_train_exit_status.1.toml"),
        f"--save-path={save_path}",
        "--no-tensorboard",
        "--stats",
        "--stats-elbo-interval=1",
        "--stats-image-interval=1",
        "--stats-latent-interval=1",
        "--stats-loglikelihood-interval=1",
        "--stats-metagenefullsummary-interval=1",
        "--stats-metagenehistogram-interval=1",
        "--stats-metagenemean-interval=1",
        "--stats-metagenesummary-interval=1",
        "--stats-rmse-interval=1",
        "--stats-scale-interval=1",
    ]
    script_runner.run("xfuse", *arguments)
    assert os.path.exists(
        os.path.join(save_path, "stats", "accuracy", "rmse.csv.gz")
    )
    assert os.path.exists(
        os.path.join(save_path, "stats", "loss", "elbo.csv.gz")
    )
    assert os.path.exists(
        os.path.join(
            save_path, "stats", "loss", "loglikelihood", "ST", "xsg.csv.gz"
        )
    )
    assert os.path.exists(
        os.path.join(
            save_path, "stats", "loss", "loglikelihood", "ST", "image.csv.gz"
        )
    )
    assert os.path.exists(
        os.path.join(save_path, "stats", "metagene-mean", "metagene-1.csv.gz")
    )
    assert os.path.exists(
        os.path.join(
            save_path,
            "stats",
            "image",
            "metagene-1",
            "profile",
            "ST",
            "meansort-1-1.png",
        )
    )
    assert os.path.exists(
        os.path.join(
            save_path,
            "stats",
            "image",
            "metagene-1",
            "profile",
            "ST",
            "invcvsort-1-1.png",
        )
    )
    assert os.path.exists(
        os.path.join(save_path, "stats", "image", "image", "sample-1-1.png")
    )
    assert os.path.exists(
        os.path.join(save_path, "stats", "image", "image", "mean-1-1.png")
    )
    assert os.path.exists(
        os.path.join(
            save_path, "stats", "image", "image", "ground_truth-1-1.png"
        )
    )
    assert os.path.exists(
        os.path.join(save_path, "stats", "image", "scale-1-1.png")
    )
    assert os.path.exists(
        os.path.join(save_path, "stats", "image", "z", "ST-1-1-1.png")
    )
    assert os.path.exists(
        os.path.join(save_path, "stats", "image", "z", "ST-0-1-1.png")
    )


@pytest.mark.script_launch_mode("subprocess")
def test_train_stats_tensorboardwriter(
    shared_datadir, script_runner, tmp_path
):
    r"""Test CLI invocation"""
    save_path = tmp_path / "output_dir"
    arguments = [
        "run",
        str(shared_datadir / "test_train_exit_status.1.toml"),
        f"--save-path={save_path}",
        "--tensorboard",
        "--no-stats",
        "--stats-elbo-interval=1",
        "--stats-image-interval=1",
        "--stats-latent-interval=1",
        "--stats-loglikelihood-interval=1",
        "--stats-metagenefullsummary-interval=1",
        "--stats-metagenehistogram-interval=1",
        "--stats-metagenemean-interval=1",
        "--stats-metagenesummary-interval=1",
        "--stats-rmse-interval=1",
        "--stats-scale-interval=1",
    ]
    script_runner.run("xfuse", *arguments)
    log_file_pattern = os.path.join(
        save_path, "stats", "tb", "events.out.tfevents.*"
    )
    assert len(glob(log_file_pattern)) > 0


def test_init(shared_datadir, script_runner, tmp_path, mocker):
    r"""Test CLI init invocation"""

    def _construct_default_config_toml():
        default_config = construct_default_config_toml()
        default_config["optimization"]["epochs"] = 1
        default_config["analyses"] = {}
        return default_config

    mocker.patch(
        "xfuse.__main__.construct_default_config_toml",
        _construct_default_config_toml,
    )

    ret = script_runner.run(
        "xfuse",
        "init",
        str(tmp_path / "config.toml"),
        str(shared_datadir / "files" / "toydata.h5"),
    )
    assert ret.success

    ret = script_runner.run(
        "xfuse",
        "run",
        str(tmp_path / "config.toml"),
        "--save-path={}".format(str(tmp_path / "output_dir")),
    )
    assert ret.success


@pytest.mark.parametrize(
    "config", ["test_restore_session.1.toml", "test_restore_session.2.toml"]
)
@pytest.mark.script_launch_mode("inprocess")
def test_restore_session(
    config, shared_datadir, script_runner, mocker, tmp_path
):
    r"""Test session restore"""
    with Session(training_data=TrainingData()):
        script_runner.run(
            "xfuse",
            "run",
            "--debug",
            f"--save-path={tmp_path}",
            str(shared_datadir / config),
        )

    state_dict = get_state_dict()
    reset_state()

    def _mock_run(*_args, **_kwargs):
        with Session(panic=Unset()):
            assert get("training_data").step > 0
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
                (
                    new_state_dict.params[param_name].data == param_value.data
                ).all()
                for param_name, param_value in state_dict.params.items()
                if param_value.data.nelement() > 0
            )

    mocker.patch("xfuse.__main__._run", _mock_run)

    ret = script_runner.run(
        "xfuse",
        "run",
        f"--save-path={tmp_path}",
        str(shared_datadir / config),
        "--session=" + str(tmp_path / "final.session"),
    )
    assert ret.success


@pytest.mark.parametrize("extra_args", [[], ["--no-mask", "--scale=0.5"]])
def test_convert_image(extra_args, shared_datadir, script_runner, tmp_path):
    r"""Test convert image data"""

    ret = script_runner.run(
        "xfuse",
        "convert",
        "image",
        "--image=" + str(shared_datadir / "files" / "st" / "image.jpg"),
        "--save-path=" + str(tmp_path),
        *extra_args,
    )
    assert ret.success
    assert os.path.exists(tmp_path / "data.h5")


@pytest.mark.parametrize("extra_args", [[], ["--no-mask", "--scale=0.5"]])
def test_convert_st(extra_args, shared_datadir, script_runner, tmp_path):
    r"""Test convert Spatial Transcriptomics Pipeline run"""

    ret = script_runner.run(
        "xfuse",
        "convert",
        "st",
        "--counts=" + str(shared_datadir / "files" / "st" / "counts.tsv"),
        "--image=" + str(shared_datadir / "files" / "st" / "image.jpg"),
        "--spots=" + str(shared_datadir / "files" / "st" / "spots.tsv"),
        "--save-path=" + str(tmp_path),
        *extra_args,
    )
    assert ret.success
    assert os.path.exists(tmp_path / "data.h5")


@pytest.mark.parametrize("extra_args", [[], ["--no-mask", "--scale=0.5"]])
def test_convert_visium(extra_args, shared_datadir, script_runner, tmp_path):
    r"""Test convert Space Ranger run"""

    ret = script_runner.run(
        "xfuse",
        "convert",
        "visium",
        "--image=" + str(shared_datadir / "files" / "visium" / "image.jpg"),
        "--bc-matrix=" + str(shared_datadir / "files" / "visium" / "data.h5"),
        "--tissue-positions="
        + str(shared_datadir / "files" / "visium" / "tissue_positions.csv"),
        "--scale-factors="
        + str(shared_datadir / "files" / "visium" / "scale_factors.json"),
        "--save-path=" + str(tmp_path),
        *extra_args,
    )
    assert ret.success
    assert os.path.exists(tmp_path / "data.h5")
