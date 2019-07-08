from copy import deepcopy

from typing import Any, Iterable, Optional

import numpy as np

import pyro as p
from pyro.poutine.messenger import Messenger

import torch as t

from ... import XFuse
from ...utility import compare
from ....handlers import Noop
from ....logging import INFO, WARNING, log
from ....session import Session, get_model
from ....utility import to_device


def purge_factors(
        xfuse: XFuse,
        data: Iterable[Any],
        extra_factors: int = 0,
        baseline: Optional[t.Tensor] = None,
        **kwargs: Any,
) -> None:
    log(INFO, 'evaluating factors')

    def _xfuse_without(n):
        reduced_xfuse = deepcopy(xfuse)
        reduced_xfuse._XFuse__experiment_store['ST'].remove_factor(n)
        return reduced_xfuse

    with Session(log_level=WARNING):
        reduced_models, ns = zip(*[
            (_xfuse_without(n).model, n)
            for n in xfuse._get_experiment('ST').factors
        ])

        def _eval_on(x):
            guide = p.poutine.trace(xfuse.guide).get_trace(x)

            full, *reduced = compare(
                x, guide, xfuse.model, *reduced_models, **kwargs)
            scores = [x - full for x in reduced]

            with p.poutine.trace() as tr:
                with p.poutine.replay(trace=guide):
                    xfuse.model(x)
            min_activation = (
                tr.trace.nodes['rim_raw']['fn'].mean.mean(0).min()
                .detach().cpu()
            )

            return scores, min_activation

        scores, min_activation = (
            np.stack(xs).mean(0)
            for xs in zip(*[_eval_on(to_device(x)) for x in data])
        )

    noncontrib = [
        n for res, n in reversed(sorted(zip(scores, ns)))
        if res >= 0
    ]
    contrib = [n for n in ns if n not in noncontrib]
    log(
        INFO,
        'contributing factors: %s',
        ', '.join(map(str, contrib)) if contrib != [] else '-',
    )
    log(
        INFO,
        'non-contributing factors: %s',
        ', '.join(map(str, noncontrib)) if noncontrib != [] else '-',
    )
    if noncontrib == []:
        for _ in range(extra_factors):
            xfuse._get_experiment('ST').add_factor((min_activation, baseline))
    else:
        for n in noncontrib[:-extra_factors]:
            xfuse._get_experiment('ST').remove_factor(n)


class FactorPurger(Messenger):
    def __new__(cls, *args, **kwargs):
        try:
            xfuse = get_model()
            _ = xfuse._get_experiment('ST')
        except (AttributeError, KeyError):
            log(WARNING, 'session model does not have an ST experiment.'
                         f' {cls.__name__} will be disabled.')
            return Noop()
        instance = super().__new__(cls)
        instance._model = xfuse
        return instance

    def __init__(
            self,
            data: Iterable[Any],
            frequency: int = 1,
            **kwargs: Any,
    ):
        self._data = data
        self._freq = frequency
        self._kwargs = kwargs

    def _handle(self, **msg) -> None:
        raise RuntimeError('unreachable code path')

    def _select_msg(self, **msg) -> bool:
        return False

    def _pyro_post_epoch(self, msg) -> None:
        if msg['kwargs']['epoch'] % self._freq == 0:
            with Session(pyro_stack=[]):
                purge_factors(self._model, self._data, **self._kwargs)
