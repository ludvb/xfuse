"""Integration tests"""

import pandas as pd
import pyro.optim
import pytest
from torch.utils.tensorboard import SummaryWriter

from hssl.data import Dataset
from hssl.data.slide.full_slide import FullSlide
from hssl.data.utility.misc import make_dataloader
from hssl.handlers.stats import RMSE
from hssl.model import XFuse
from hssl.model.experiment.st import ST, FactorDefault
from hssl.session import Session
from hssl.train import train
from hssl.utility import design_matrix_from


@pytest.mark.fix_rng
@pytest.mark.slow
def test_toydata(tmp_path, mocker, toydata):
    """Integration test on toy dataset"""
    counts, image, label = toydata
    design_matrix = design_matrix_from(pd.DataFrame({"sample": [1]}))
    slide = FullSlide(counts, image, label)
    dataset = Dataset([slide], design_matrix)
    dataloader = make_dataloader(dataset)
    st_experiment = ST(
        n=len(slide),
        depth=2,
        num_channels=4,
        factors=[FactorDefault(0.0, None) for _ in range(3)],
    )
    xfuse = XFuse(experiments=[st_experiment], latent_size=16)
    summary_writer = SummaryWriter(tmp_path)
    rmse = RMSE(summary_writer)
    rmse.add_scalar = mocker.MagicMock()
    with Session(model=xfuse, optimizer=pyro.optim.Adam({"lr": 0.01})), rmse:
        train(dataloader, 100)
    rmses = [x[1][1] for x in rmse.add_scalar.mock_calls]
    assert rmses[0] > rmses[19]
    assert rmses[19] > rmses[-1]
    assert rmses[-1] < 10.0
