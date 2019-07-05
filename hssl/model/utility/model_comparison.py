import numpy as np

import pyro as p


def compare(data, guide, *models, num_samples=1):
    def _once():
        def _evaluate(model):
            with p.poutine.trace() as tr:
                with p.poutine.block(hide_fn=lambda x: not x['is_observed']):
                    with p.poutine.replay(trace=guide):
                        model(data)
            return tr.trace.log_prob_sum().item()
        return [_evaluate(model) for model in models]
    return np.mean([_once() for _ in range(num_samples)], 0)
