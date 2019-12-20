import itertools as it
import os
from contextlib import ExitStack
from typing import List

import numpy as np
import pyro
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .handlers import Checkpointer, stats
from .logging import DEBUG, log
from .model.experiment.st import FactorPurger
from .session import get, require
from .utility import to_device


def test_convergence():
    r"""
    Tests if the model has converged according to a heuristic stopping
    criterion
    """
    training_data = get("training_data")
    rmean_long = np.mean(training_data.elbos[-len(training_data.elbos) // 2 :])
    rmean_short = np.mean(
        training_data.elbos[-len(training_data.elbos) // 4 :]
    )
    log(DEBUG, "Running mean ELBO (long):  %.2e", rmean_long)
    log(DEBUG, "Running mean ELBO (short): %.2e", rmean_short)
    return rmean_long > rmean_short


def train(epochs: int = -1):
    """Trains the session model"""
    optim = require("optimizer")
    model = require("model")
    dataloader = require("dataloader")
    training_data = get("training_data")

    messengers: List[Messenger] = [
        FactorPurger(
            period=lambda e: (
                e % 100 == 0 and (epochs < 0 or e <= epochs - 100)
            ),
            num_samples=3,
        )
    ]

    if get("save_path") is not None:
        messengers.append(Checkpointer(period=1000))

        def _every(n):
            def _predicate(**_msg):
                if training_data.step % n == 0:
                    return True
                return False

            return _predicate

        writer = SummaryWriter(os.path.join(get("save_path"), "stats"))

        messengers.extend(
            [
                stats.ELBO(writer, _every(1)),
                stats.FactorActivationHistogram(writer, _every(10)),
                stats.FactorActivationMaps(writer, _every(100)),
                stats.FactorActivationMean(writer, _every(1)),
                stats.FactorActivationSummary(writer, _every(100)),
                stats.FactorActivationFullSummary(writer, _every(1000)),
                stats.Image(writer, _every(100)),
                stats.Latent(writer, _every(100)),
                stats.LogLikelihood(writer, _every(1)),
                stats.RMSE(writer, _every(1)),
                stats.Scale(writer, _every(100)),
            ]
        )

    @effectful(type="step")
    def _step(*, x):
        loss = pyro.infer.Trace_ELBO()
        return -pyro.infer.SVI(model.model, model.guide, optim, loss).step(x)

    @effectful(type="epoch")
    def _epoch(*, epoch):
        progress = tqdm(dataloader, position=0, dynamic_ncols=True)
        elbo = []
        for x in progress:
            training_data.step += 1
            elbo.append(_step(x=to_device(x)))
            progress.set_description(
                f"Epoch {epoch:05d} :: Mean ELBO {np.mean(elbo):+.3e}"
            )
        return np.mean(elbo)

    with ExitStack() as stack:
        for messenger in messengers:
            stack.enter_context(messenger)

        for epoch in tqdm(
            (
                it.count(training_data.epoch + 1)
                if epochs < 0
                else range(training_data.epoch + 1, epochs + 1)
            ),
            position=1,
            desc="Optimizing model",
            unit="epoch",
            dynamic_ncols=True,
        ):
            training_data.epoch = epoch
            training_data.elbos.append(_epoch(epoch=epoch))

            if epochs < 0 and test_convergence():
                log(DEBUG, "Model has converged, stopping")
                break
