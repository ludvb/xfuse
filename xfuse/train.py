import itertools as it

import pyro
from pyro.poutine.runtime import effectful

from .logging import DEBUG, INFO, Progressbar, log
from .session import get, require
from .utility.pyro import TraceWithDuplicates
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

    @effectful(type="step")
    def _step(*, x):
        loss = pyro.infer.Trace_ELBO()
        with TraceWithDuplicates() as trace:
            pyro.infer.SVI(model.model, model.guide, optim, loss).step(x)
        return trace.trace

    @effectful(type="epoch")
    def _epoch(*, epoch):
        if isinstance(optim, pyro.optim.PyroLRScheduler):
            optim.step(epoch=epoch)
        with Progressbar(
            dataloader, desc=f"Epoch {epoch:05d}", leave=False,
        ) as iterator:
            for x in iterator:
                training_data.step += 1
                _step(x=to_device(x))

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
            _epoch(epoch=epoch)
            log(
                INFO,
                " | ".join(
                    ["Epoch %05d", "Running ELBO %+.4e", "Running RMSE %.3f"]
                ),
                epoch,
                training_data.elbo_long or 0.0,
                training_data.rmse or 0.0,
            )

            if epochs < 0 and test_convergence():
                log(DEBUG, "Model has converged, stopping")
                break
