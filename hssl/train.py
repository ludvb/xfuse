import itertools as it
from typing import Optional

import numpy as np
import pyro
from pyro.poutine.runtime import effectful
from torch.utils.data import DataLoader
from tqdm import tqdm

from .session import get
from .utility import to_device


def train(dataloader: DataLoader, epochs: Optional[int] = None):
    """Trains the session model"""

    @effectful(type="step")
    def _step(*, x):
        optim = get("optimizer")
        loss = pyro.infer.Trace_ELBO()
        model = get("model")
        return pyro.infer.SVI(model.model, model.guide, optim, loss).step(x)

    @effectful(type="epoch")
    def _epoch(*, data, epoch):
        progress = tqdm(data, dynamic_ncols=True)
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
        _epoch(data=dataloader, epoch=epoch)
