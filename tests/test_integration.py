r"""Integration tests"""

import pyro.optim
from torch.utils.tensorboard import SummaryWriter

import pytest
from xfuse.handlers.stats import RMSE
from xfuse.model import XFuse
from xfuse.model.experiment.st import ST, MetageneDefault
from xfuse.model.experiment.st.metagene_eval import purge_metagenes
from xfuse.model.experiment.st.metagene_expansion_strategy import (
    Extra,
    DropAndSplit,
)
from xfuse.session import Session, get
from xfuse.train import train
from xfuse.utility.design import extract_covariates


@pytest.mark.fix_rng
@pytest.mark.slow
def test_toydata(tmp_path, mocker, toydata):
    r"""Integration test on toy dataset"""
    st_experiment = ST(
        depth=2,
        num_channels=4,
        metagenes=[MetageneDefault(0.0, None) for _ in range(3)],
    )
    xfuse = XFuse(experiments=[st_experiment])
    summary_writer = SummaryWriter(tmp_path)
    rmse = RMSE(summary_writer)
    rmse.add_scalar = mocker.MagicMock()
    with Session(
        model=xfuse,
        optimizer=pyro.optim.Adam({"lr": 0.0001}),
        dataloader=toydata,
        covariates=extract_covariates(toydata.dataset.data.design),
    ), rmse:
        train(100 + get("training_data").epoch)
    rmses = [x[1][1] for x in rmse.add_scalar.mock_calls]
    assert rmses[-1] < 6.0


@pytest.mark.fix_rng
@pytest.mark.parametrize(
    "expansion_strategies,compute_expected_metagenes",
    [
        ((Extra(5),), lambda n: (n + 5, n)),
        ((DropAndSplit(),) * 2, lambda n: (2 * n, n)),
    ],
)
def test_metagene_expansion(
    # pylint: disable=redefined-outer-name
    toydata,
    pretrained_toy_model,
    expansion_strategies,
    compute_expected_metagenes,
):
    r"""Test metagene expansion dynamics"""
    st_experiment = pretrained_toy_model.get_experiment("ST")
    num_start_metagenes = len(st_experiment.metagenes)

    for expansion_strategy, expected_metagenes in zip(
        expansion_strategies, compute_expected_metagenes(num_start_metagenes)
    ):
        with Session(
            metagene_expansion_strategy=expansion_strategy,
            dataloader=toydata,
            covariates=extract_covariates(toydata.dataset.data.design),
        ):
            purge_metagenes(pretrained_toy_model, num_samples=10)
        assert len(st_experiment.metagenes) == expected_metagenes
