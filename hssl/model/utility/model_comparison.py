import numpy as np

import pyro as p


def compare(data, guide, *models, num_samples=1):
    r"""
    Returns the ELBO of given models on provided data using the same guide
    trace
    """

    def _once():
        def _evaluate(model):
            with p.poutine.trace() as trace:
                with p.poutine.replay(trace=guide):
                    model(data)
            return trace.trace.log_prob_sum().item() - guide.log_prob_sum(
                site_filter=lambda name, site: name in trace.trace.nodes
            )

        return [_evaluate(model) for model in models]

    return np.mean([_once() for _ in range(num_samples)], 0)
