import itertools as it
from typing import Optional

import numpy as np
import pyro
from pyro.poutine.runtime import effectful
from tqdm import tqdm

from .session import get, require
from .utility import to_device


def train(epochs: Optional[int] = None):
    """Trains the session model"""
    optim = require("optimizer")
    model = require("model")
    dataloader = require("dataloader")

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

    for epoch in it.takewhile(
        lambda x: epochs is None or x <= epochs, it.count(1)
    ):
        _epoch(epoch=epoch)
