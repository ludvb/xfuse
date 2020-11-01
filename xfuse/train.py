import itertools as it
import os
from contextlib import ExitStack
from typing import List

import numpy as np
import pyro
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful
from torch.utils.tensorboard.writer import SummaryWriter

from .handlers import Checkpointer, stats
from .logging import DEBUG, INFO, Progressbar, log
from .model.experiment.st.metagene_eval import MetagenePurger
from .session import get, require
from .utility.tensor import to_device


def test_convergence():
    r"""
    Tests if the model has converged according to a heuristic stopping
    criterion
    """
    training_data = get("training_data")
    return (
        training_data.epoch > 1000
        and training_data.elbo_long > training_data.elbo_short
    )


def train(epochs: int = -1):
    """Trains the session model"""
    optim = require("optimizer")
    model = require("model")
    dataloader = require("dataloader")
    training_data = get("training_data")

    messengers: List[Messenger] = [
        MetagenePurger(
            period=lambda e: (
                e % 1000 == 0 and (epochs < 0 or e <= epochs - 1000)
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
                stats.MetageneHistogram(writer, _every(100)),
                stats.MetageneMean(writer, _every(100)),
                stats.MetageneSummary(writer, _every(1000)),
                stats.MetageneFullSummary(writer, _every(5000)),
                stats.Image(writer, _every(1000)),
                stats.Latent(writer, _every(1000)),
                stats.LogLikelihood(writer, _every(1)),
                stats.RMSE(writer, _every(1)),
                stats.Scale(writer, _every(1000)),
            ]
        )

    @effectful(type="step")
    def _step(*, x):
        loss = pyro.infer.Trace_ELBO()
        return -pyro.infer.SVI(model.model, model.guide, optim, loss).step(x)

    @effectful(type="epoch")
    def _epoch(*, epoch):
        if isinstance(optim, pyro.optim.PyroLRScheduler):
            optim.step(epoch=epoch)
        with Progressbar(
            dataloader, desc=f"Epoch {epoch:05d}", leave=False,
        ) as iterator:
            elbo = []
            for x in iterator:
                training_data.step += 1
                elbo.append(_step(x=to_device(x)))
        return np.mean(elbo)

    with ExitStack() as stack:
        for messenger in messengers:
            stack.enter_context(messenger)

        with Progressbar(
            (
                it.count(training_data.epoch + 1)
                if epochs < 0
                else range(training_data.epoch + 1, epochs + 1)
            ),
            desc="Optimizing model",
            unit="epoch",
            dynamic_ncols=True,
            leave=False,
        ) as iterator:
            for epoch in iterator:
                training_data.epoch = epoch
                elbo = _epoch(epoch=epoch)
                log(
                    INFO,
                    " | ".join(
                        [
                            "Epoch %05d",
                            "ELBO %+.3e",
                            "Running ELBO %+.4e",
                            "Running RMSE %.3f",
                        ]
                    ),
                    epoch,
                    elbo,
                    training_data.elbo_long or 0.0,
                    training_data.rmse or 0.0,
                )

                if epochs < 0 and test_convergence():
                    log(DEBUG, "Model has converged, stopping")
                    break
