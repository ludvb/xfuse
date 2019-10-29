r"""Integration tests"""

import pyro.optim
from torch.utils.tensorboard import SummaryWriter

import pytest
from hssl.handlers.stats import RMSE
from hssl.model import XFuse
from hssl.model.experiment.st import (
    ST,
    ExtraBaselines,
    FactorDefault,
    RetractAndSplit,
    purge_factors,
)
from hssl.session import Session
from hssl.train import train


@pytest.mark.fix_rng
@pytest.mark.slow
def test_toydata(tmp_path, mocker, toydata):
    r"""Integration test on toy dataset"""
    st_experiment = ST(
        n=len(toydata),
        depth=2,
        num_channels=4,
        factors=[FactorDefault(0.0, None) for _ in range(3)],
    )
    xfuse = XFuse(experiments=[st_experiment], latent_size=16)
    summary_writer = SummaryWriter(tmp_path)
    rmse = RMSE(summary_writer)
    rmse.add_scalar = mocker.MagicMock()
    with Session(model=xfuse, optimizer=pyro.optim.Adam({"lr": 0.01})), rmse:
        train(toydata, 100)
    rmses = [x[1][1] for x in rmse.add_scalar.mock_calls]
    assert rmses[0] > rmses[19]
    assert rmses[19] > rmses[-1]
    assert rmses[-1] < 10.0


@pytest.fixture
def pretrained_toy_model(toydata):
    r"""Pretrained toy model"""
    st_experiment = ST(
        n=len(toydata),
        depth=2,
        num_channels=4,
        factors=[FactorDefault(0.0, None) for _ in range(1)],
    )
    xfuse = XFuse(experiments=[st_experiment], latent_size=16)
    with Session(model=xfuse, optimizer=pyro.optim.Adam({"lr": 0.01})):
        train(toydata, 100)
    return xfuse


@pytest.mark.fix_rng
@pytest.mark.parametrize(
    "expansion_strategies,compute_expected_factors",
    [
        ((ExtraBaselines(5), ExtraBaselines(0)), lambda n: (n + 5, n)),
        ((RetractAndSplit(),) * 2, lambda n: (2 * n, n)),
    ],
)
def test_factor_expansion(
    # pylint: disable=redefined-outer-name
    toydata,
    pretrained_toy_model,
    expansion_strategies,
    compute_expected_factors,
):
    r"""Test factor expansion dynamics"""
    st_experiment = pretrained_toy_model.get_experiment("ST")
    num_start_factors = len(st_experiment.factors)

    for expansion_strategy, expected_factors in zip(
        expansion_strategies, compute_expected_factors(num_start_factors)
    ):
        with Session(factor_expansion_strategy=expansion_strategy):
            purge_factors(pretrained_toy_model, toydata, num_samples=10)
        assert len(st_experiment.factors) == expected_factors
