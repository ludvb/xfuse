import itertools as it
import os
from contextlib import ExitStack
from functools import reduce
from typing import List

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import pyro
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful

from .handlers import Checkpointer, stats
from .model.experiment.st import FactorPurger
from .session import get, require
from .utility import to_device


def train(epochs: int = -1):
    """Trains the session model"""
    optim = require("optimizer")
    model = require("model")
    dataloader = require("dataloader")

    messengers: List[Messenger] = [
        FactorPurger(
            frequency=lambda e: (
                e % 100 == 0 and (epochs < 0 or e <= epochs - 100)
            ),
            num_samples=3,
        ),
    ]

    if get("save_path") is not None:
        messengers.append(Checkpointer(frequency=100000))

        def _every(n):
            def _predicate(**_msg):
                if int(get("global_step")) % n == 0:
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
        return pyro.infer.SVI(model.model, model.guide, optim, loss).step(x)

    @effectful(type="epoch")
    def _epoch(*, epoch):
        progress = tqdm(dataloader, dynamic_ncols=True)
        elbo = []
        for x in progress:
            elbo.append(_step(x=to_device(x)))
            progress.set_description(
                f"epoch {epoch:05d} / mean ELBO {np.mean(elbo):.3e}"
            )
            global_step = get("global_step")
            global_step += 1

    with ExitStack() as stack:
        for messenger in messengers:
            stack.enter_context(messenger)
        for epoch in it.takewhile(
            lambda x: epochs < 0 or x <= epochs, it.count(1)
        ):
            _epoch(epoch=epoch)
