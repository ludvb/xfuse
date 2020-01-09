from typing import List

import pyro
import torch

from ...session import Session


def compare(data, guide, *models) -> List[float]:
    r"""
    Returns the ELBO of given models on provided data using the same guide
    trace
    """

    def _evaluate(model):
        with pyro.poutine.trace() as trace:
            with pyro.poutine.replay(trace=guide):
                model(data)
        return (
            trace.trace.log_prob_sum().item()
            - guide.log_prob_sum(
                site_filter=lambda name, site: name in trace.trace.nodes
            ).item()
        )

    with torch.no_grad(), Session(eval=True):
        result = [_evaluate(model) for model in models]
    return result
